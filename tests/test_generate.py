"""离线验证生成节点的真实模型适配、图内传递和失败边界。"""
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import httpx
from langchain_openai import ChatOpenAI
from pydantic import ValidationError
from tenacity import wait_none

from src.nodes.llm_call import generate_output
from src.state import create_initial_state
from test_graph.graph import build_test_graph


class GenerateTest(unittest.TestCase):
    def state(self):
        state = create_initial_state(jd_text="Build AI applications")
        state["cv_text"] = "Built an internal AI assistant."
        state["template_text"] = "Write an evidence-based cover letter."
        state["prepared_materials"] = "引用 CV 的 AI 项目，保留部署职责，不虚构预算管理。"
        state["human_inputs"] = [{"summary": "职责", "questions": ["负责什么？"],
                                  "answer": "更正：未负责预算"}]
        return state

    def test_graph_generation_and_transient_retry(self):
        requests = []
        letter = "Dear Hiring Manager,\n\nI built AI applications.\n\nKind regards,\nAlex Example"

        def respond(request):
            requests.append(json.loads(request.content))
            if len(requests) == 1:
                return httpx.Response(503, json={"error": {"message": "Unavailable"}})
            return httpx.Response(200, json={
                "id": "chatcmpl-test", "object": "chat.completion", "created": 0,
                "model": "gpt-4o", "choices": [{"index": 0, "finish_reason": "stop",
                    "message": {"role": "assistant", "content": json.dumps({"output_markdown": letter})}}],
            })

        state = self.state()
        with httpx.Client(transport=httpx.MockTransport(respond)) as client, \
                patch("src.nodes.llm_call.analyze_jd", return_value={"jd_analysis": {"summary": "AI"}}), \
                patch("src.nodes.llm_call.evaluate", return_value={"evaluation": {
                    "status": "ready", "summary": "足够", "questions": []}}), \
                patch("src.nodes.llm_call.prepare", return_value={"prepared_materials": state["prepared_materials"]}), \
                patch.object(generate_output.retry, "wait", wait_none()):
            llm = ChatOpenAI(model="gpt-4o", api_key="offline", http_client=client, max_retries=0)
            graph = build_test_graph(llm, stop_after="generate_output")
            result = graph.invoke(state, {"configurable": {"thread_id": "generate-test"}})
        self.assertEqual(result["output_markdown"], letter)
        self.assertIsNone(state["output_markdown"])
        self.assertEqual(result["output_files"], state["output_files"])
        self.assertEqual(len(requests), 2)
        self.assertTrue(requests[-1]["response_format"]["json_schema"]["strict"])
        system = requests[-1]["messages"][0]["content"]
        self.assertIn("# Humanizer: remove AI writing patterns", system)
        self.assertIn("Embedded mode", system)
        self.assertIn("格式约定：只使用纯文本段落与 Markdown 硬换行", system)
        materials = json.loads(requests[-1]["messages"][-1]["content"])
        for key in ("prepared_materials", "template_text", "jd_text", "cv_text", "scenes_text", "human_inputs"):
            self.assertEqual(materials[key], state[key])
        self.assertEqual(materials["current_date"], datetime.now(ZoneInfo("Pacific/Auckland")).date().isoformat())

    def test_invalid_inputs_and_unavailable_skill(self):
        for key in ("prepared_materials", "template_text", "jd_text", "cv_text"):
            for value in (None, "", " \n "):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    generate_output({**self.state(), key: value}, llm=object())
        with tempfile.TemporaryDirectory() as directory, patch("src.nodes.llm_call.PROJECT_ROOT", Path(directory)):
            with self.assertRaises(FileNotFoundError):
                generate_output(self.state(), llm=object())
            skill = Path(directory) / "skills/humanizer/SKILL.md"
            skill.parent.mkdir(parents=True)
            skill.write_text(" \n ")
            with self.assertRaises(ValueError):
                generate_output(self.state(), llm=object())

    def test_blank_model_output_fails_without_retry(self):
        requests = []

        def respond(request):
            requests.append(request)
            return httpx.Response(200, json={
                "id": "chatcmpl-test", "object": "chat.completion", "created": 0,
                "model": "gpt-4o", "choices": [{"index": 0, "finish_reason": "stop",
                    "message": {"role": "assistant", "content": '{"output_markdown": "   "}'}}],
            })

        state = self.state()
        with httpx.Client(transport=httpx.MockTransport(respond)) as client:
            llm = ChatOpenAI(model="gpt-4o", api_key="offline", http_client=client, max_retries=0)
            with self.assertRaises(ValidationError):
                generate_output(state, llm=llm)
        self.assertEqual(len(requests), 1)
        self.assertIsNone(state["output_markdown"])


if __name__ == "__main__":
    unittest.main()
