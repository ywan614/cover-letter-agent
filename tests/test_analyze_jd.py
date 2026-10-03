"""离线验证真实 structured output 解析及 LangGraph state 更新。"""
import json
import unittest
from unittest.mock import patch

import httpx
from langchain_openai import ChatOpenAI
from openai import APITimeoutError, AuthenticationError, BadRequestError, LengthFinishReasonError, RateLimitError
from pydantic import ValidationError

from src.state import create_initial_state
from src.nodes.llm_call import analyze_jd
from test_graph.graph import build_test_graph


class AnalyzeJDTest(unittest.TestCase):
    def test_retries_are_bounded_and_only_for_retryable_errors(self):
        expected = {
            "job_title": None, "company": None, "summary": "开发 API。",
            "responsibilities": [], "requirements": [], "nice_to_haves": [],
        }
        for failure, exception, retryable in (
            ("timeout", APITimeoutError, True),
            ("length", LengthFinishReasonError, True),
            ("context", BadRequestError, True),
            ("rate_limit", RateLimitError, True),
            ("invalid", BadRequestError, False),
            ("auth", AuthenticationError, False),
        ):
            for recover in (True, False):
                with self.subTest(failure=failure, recover=recover):
                    calls = []

                    def respond(request):
                        calls.append(request)
                        fail = not recover or len(calls) == 1
                        if fail and failure == "timeout":
                            raise httpx.ReadTimeout("timeout", request=request)
                        if fail and failure not in ("timeout", "length"):
                            status, code = {
                                "context": (400, "context_length_exceeded"),
                                "rate_limit": (429, "rate_limit_exceeded"),
                                "invalid": (400, "invalid_parameter"),
                                "auth": (401, "invalid_api_key"),
                            }[failure]
                            return httpx.Response(status, json={"error": {
                                "message": "test error", "type": "test_error", "code": code,
                            }})
                        return httpx.Response(200, json={
                            "id": "chatcmpl-test", "object": "chat.completion", "created": 0,
                            "model": "gpt-4o", "choices": [{
                                "index": 0, "finish_reason": "length" if fail else "stop",
                                "message": {"role": "assistant", "content": json.dumps(expected)},
                            }],
                        })

                    initial = create_initial_state(jd_text="Build APIs.")
                    with httpx.Client(transport=httpx.MockTransport(respond)) as client, \
                            patch.object(analyze_jd.retry, "sleep") as sleep:
                        llm = ChatOpenAI(model="gpt-4o", api_key="offline-test",
                                         http_client=client, max_retries=0)
                        graph = build_test_graph(llm)
                        if recover and retryable:
                            self.assertEqual(graph.invoke(initial, config={"configurable": {"thread_id": "test"}})["jd_analysis"], expected)
                            self.assertEqual(len(calls), 2)
                        else:
                            with self.assertRaises(exception):
                                graph.invoke(initial, config={"configurable": {"thread_id": "test"}})
                            self.assertEqual(len(calls), 3 if retryable else 1)
                        self.assertEqual([float(call.args[0]) for call in sleep.call_args_list],
                                         [1.0, 2.0][:len(calls) - 1])
                        self.assertIsNone(initial["jd_analysis"])

    def test_structured_output_and_state(self):
        expected = {
            "job_title": None, "company": None, "summary": "开发 API。",
            "responsibilities": ["开发 API"], "requirements": ["掌握 Python"],
            "nice_to_haves": [],
        }
        requests = []
        payload = expected

        def respond(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={
                "id": "chatcmpl-test", "object": "chat.completion", "created": 0,
                "model": "gpt-4o", "choices": [{
                    "index": 0, "finish_reason": "stop",
                    "message": {"role": "assistant", "content": json.dumps(payload)},
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            })

        with httpx.Client(transport=httpx.MockTransport(respond)) as client:
            llm = ChatOpenAI(model="gpt-4o", api_key="offline-test", http_client=client)
            graph = build_test_graph(llm)
            initial = create_initial_state(jd_text="Build APIs. Python required.")
            result = graph.invoke(initial, config={"configurable": {"thread_id": "test"}})
            self.assertEqual(result, {**initial, "jd_analysis": expected})
            self.assertIsNone(initial["jd_analysis"])
            response_format = requests[0]["response_format"]
            self.assertEqual(response_format["type"], "json_schema")
            self.assertTrue(response_format["json_schema"]["strict"])
            self.assertCountEqual(response_format["json_schema"]["schema"]["required"], expected)
            self.assertEqual(requests[0]["messages"][-1]["content"], initial["jd_text"])

            # 缺字段或错误类型必须失败，不能写入不完整的分析。
            for payload in ({"summary": "incomplete"}, {**expected, "requirements": "Python"}):
                with self.subTest(payload=payload), self.assertRaises(ValidationError):
                    graph.invoke(initial, config={"configurable": {"thread_id": "test"}})
                self.assertIsNone(initial["jd_analysis"])

            count = len(requests)
            with self.assertRaisesRegex(ValueError, "非空字符串"):
                graph.invoke(create_initial_state(jd_text=" \n "),
                             config={"configurable": {"thread_id": "empty"}})
            self.assertEqual(len(requests), count)


if __name__ == "__main__":
    unittest.main()
