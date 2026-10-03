"""真实 SDK + MockTransport 验证配置、思考开关和超时原因，不请求外网。"""
import json
import tempfile
import unittest
from configparser import ConfigParser
from pathlib import Path
from unittest.mock import patch

import httpx
from langchain_openai import ChatOpenAI
from openai import RateLimitError

from src.llm import create_llm
from src.logger import exception_details
from src.nodes.llm_call import analyze_jd, evaluate, prepare, generate_output
from src.state import create_initial_state
from src.usage import TokenUsageTracker


class LLMDiagnosticsTest(unittest.TestCase):
    def config(self):
        config = ConfigParser()
        config.read_dict({"openai": {"openai_api_key": "offline"}, "model": {
            "model_name": "qwen3.7-plus", "model_base_url": "https://offline.invalid/v1",
            "timeout_seconds": "300",
        }})
        return config

    def test_timeout_retry_and_thinking_on_wire(self):
        requests = []
        outputs = iter([
            {"job_title": None, "company": None, "summary": "AI", "responsibilities": [],
             "requirements": [], "nice_to_haves": []},
            {"status": "ready", "summary": "足够", "questions": []},
            {"prepared_materials": "内部 AI 项目；来源 CV；不扩写。"},
            {"output_markdown": "Dear Hiring Manager,\nI build AI systems."},
        ])

        def respond(request):
            requests.append(json.loads(request.content))
            self.assertEqual(request.extensions["timeout"]["read"], 300)
            if len(requests) == 1:
                raise httpx.ReadTimeout("private request content", request=request)
            return httpx.Response(200, json={
                "id": "offline", "object": "chat.completion", "created": 0,
                "model": "qwen3.7-plus", "choices": [{"index": 0, "finish_reason": "stop",
                    "message": {"role": "assistant", "content": json.dumps(next(outputs))}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
                          "completion_tokens_details": {"reasoning_tokens": 2}},
            })

        with tempfile.TemporaryDirectory() as directory, \
                httpx.Client(transport=httpx.MockTransport(respond)) as client, \
                patch("src.llm.ChatOpenAI", side_effect=lambda **kw: ChatOpenAI(http_client=client, **kw)), \
                patch.object(analyze_jd.retry, "sleep"), \
                self.assertLogs("cover_letter_agent", level="INFO") as logs:
            tracker = TokenUsageTracker(Path(directory))
            llm = create_llm(self.config(), usage_tracker=tracker)
            state = create_initial_state(jd_text="Build AI")
            state["cv_text"] = "Built an internal AI assistant."
            state["template_text"] = "Write an evidence-based cover letter."
            for node in (analyze_jd, evaluate, prepare, generate_output):
                state.update(node(state, llm=llm))
            self.assertEqual([r["enable_thinking"] for r in requests], [False, False, False, True, False])
            report = json.loads(tracker.path.read_text())
            failed = report["calls"][0]
            self.assertEqual(failed["error_chain"][-1]["type"], "ReadTimeout")
            self.assertGreaterEqual(failed["elapsed_s"], 0)
            self.assertIn("finished_at", failed)
            self.assertNotIn("private request content", tracker.path.read_text())
            self.assertEqual(report["calls"][1]["usage"]["output_token_details"]["reasoning"], 2)
        logged = "\n".join(logs.output)
        for value in ("ReadTimeout", "attempt=2/3", "timeout_s=300", "elapsed_s=", "call_id="):
            self.assertIn(value, logged)
        self.assertNotIn("private request content", logged)

    def test_timeout_validation_and_http_diagnostics(self):
        config = self.config()
        config["model"].pop("timeout_seconds")
        self.assertEqual(create_llm(config).request_timeout, 300)
        config["model"]["timeout_seconds"] = "120"
        self.assertEqual(create_llm(config).request_timeout, 120)
        for value in ("0", "-1", "nan", "inf", "invalid"):
            config["model"]["timeout_seconds"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                create_llm(config)
        response = httpx.Response(429, request=httpx.Request("POST", "https://offline.invalid"),
                                  headers={"x-request-id": "request-123"})
        error = RateLimitError("private body", response=response, body={"code": "rate_limit_exceeded"})
        detail = exception_details(error)["error_chain"][0]
        self.assertEqual(detail["status_code"], "429")
        self.assertEqual(detail["request_id"], "request-123")
        self.assertEqual(detail["code"], "rate_limit_exceeded")
        self.assertNotIn("private body", str(detail))


if __name__ == "__main__":
    unittest.main()
