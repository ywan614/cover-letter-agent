"""离线验证 prepare 输出、资料传递及人工补充后的真实图路由。"""
import json
import unittest
from unittest.mock import patch

import httpx
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from pydantic import ValidationError

from src.nodes.llm_call import PreparedMaterialsOutput, prepare
from src.state import create_initial_state
from test_graph.graph import build_test_graph


class PrepareTest(unittest.TestCase):
    def test_prepare_after_human_input(self):
        for answers in ([], [""], [" \n "], ["我负责部署", "更正：未负责预算"]):
            with self.subTest(answers=answers):
                requests = []
                output = {"prepared_materials": "## 写作主线\n突出内部 AI 项目交付。\n## 段落方案\n引用 CV 项目，不扩展管理职责。"}

                def respond(request):
                    requests.append(json.loads(request.content))
                    return httpx.Response(200, json={
                        "id": "chatcmpl-test", "object": "chat.completion", "created": 0,
                        "model": "gpt-4o", "choices": [{"index": 0, "finish_reason": "stop",
                            "message": {"role": "assistant", "content": json.dumps(output)}}],
                    })

                evaluation = {"status": "missing_info" if answers else "ready",
                              "summary": "现有资料", "questions": ["个人职责？"] if answers else []}
                state = create_initial_state(jd_text="Build AI")
                state["cv_text"] = "Built an internal AI assistant."
                state["template_text"] = "Write an evidence-based cover letter."
                state["scenes_text"] = ""
                with httpx.Client(transport=httpx.MockTransport(respond)) as client, \
                        patch("src.nodes.llm_call.analyze_jd", return_value={"jd_analysis": {"summary": "AI"}}), \
                        patch("src.nodes.llm_call.evaluate", return_value={"evaluation": evaluation}):
                    llm = ChatOpenAI(model="gpt-4o", api_key="offline", http_client=client)
                    graph = build_test_graph(llm, stop_after="prepare")
                    config = {"configurable": {"thread_id": "prepare-test"}}
                    result = graph.invoke(state, config)
                    for answer in answers:
                        self.assertIn("__interrupt__", result)
                        result = graph.invoke(Command(resume=answer), config)
                self.assertNotIn("__interrupt__", result)
                self.assertEqual(result["prepared_materials"], output["prepared_materials"])
                self.assertIsNone(result["output_markdown"])
                self.assertEqual(result["evaluation"], evaluation)
                self.assertIsNone(state["prepared_materials"])
                self.assertEqual(len(requests), 1)
                self.assertTrue(requests[0]["response_format"]["json_schema"]["strict"])
                materials = json.loads(requests[0]["messages"][-1]["content"])
                self.assertEqual(set(materials), {"jd_text", "jd_analysis", "cv_text", "scenes_text",
                                                  "template_text", "evaluation", "human_inputs"})
                for key in materials:
                    self.assertEqual(materials[key], result[key])
                self.assertEqual([item["answer"] for item in materials["human_inputs"]], answers)

    def test_invalid_inputs_and_blank_output(self):
        state = create_initial_state(jd_text="JD")
        state["cv_text"] = "Built an internal AI assistant."
        state["template_text"] = "Write an evidence-based cover letter."
        state["jd_analysis"] = {"summary": "AI"}
        state["evaluation"] = {"status": "ready", "summary": "足够", "questions": []}
        for key, value in (("jd_text", " "), ("cv_text", ""), ("template_text", None),
                           ("jd_analysis", None), ("evaluation", None)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                prepare({**state, key: value}, llm=object())
        for value in ("", " \n "):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                PreparedMaterialsOutput(prepared_materials=value)


if __name__ == "__main__":
    unittest.main()
