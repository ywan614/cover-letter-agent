"""用真实 LangChain 回调链离线验证统计，不请求模型服务。"""
import json
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import LLMResult

from src.usage import TokenUsageTracker


class TokenUsageTest(unittest.TestCase):
    def test_each_call_and_totals(self):
        with tempfile.TemporaryDirectory() as directory:
            tracker = TokenUsageTracker(Path(directory))
            llm = FakeMessagesListChatModel(callbacks=[tracker], responses=[
                AIMessage(content="private response", usage_metadata={
                    "input_tokens": 10, "output_tokens": 4, "total_tokens": 14,
                    "input_token_details": {"cache_read": 2},
                }),
                AIMessage(content="private response", usage_metadata={
                    "input_tokens": 20, "output_tokens": 6, "total_tokens": 26,
                }),
            ])
            for node in ("analyze_jd", "evaluate"):
                llm.invoke("private prompt", config={"metadata": {"langgraph_node": node}})
                self.assertTrue(tracker.path.exists())
            report = json.loads(tracker.path.read_text())
            self.assertEqual(report["call_count"], 2)
            self.assertEqual([c["node"] for c in report["calls"]], ["analyze_jd", "evaluate"])
            self.assertEqual(report["total_usage"], {"input_tokens": 30, "output_tokens": 10, "total_tokens": 40})
            self.assertEqual(report["calls"][0]["usage"]["input_token_details"]["cache_read"], 2)
            self.assertNotIn("private", tracker.path.read_text())

    def test_missing_usage_and_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            tracker = TokenUsageTracker(Path(directory))
            llm = FakeMessagesListChatModel(callbacks=[tracker], responses=[AIMessage(content="no usage")])
            llm.invoke("test")
            run_id = uuid4()
            tracker.on_chat_model_start({}, [], run_id=run_id)
            tracker.on_llm_error(RuntimeError("secret error"), run_id=run_id)
            report = json.loads(tracker.path.read_text())
            self.assertFalse(report["usage_complete"])
            self.assertIsNone(report["total_usage"])
            self.assertEqual(report["calls"][1]["status"], "failed")
            self.assertNotIn("secret", tracker.path.read_text())

    def test_provider_usage_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            tracker = TokenUsageTracker(Path(directory))
            run_id = uuid4()
            tracker.on_chat_model_start({}, [], run_id=run_id)
            tracker.on_llm_end(LLMResult(generations=[], llm_output={"token_usage": {
                "prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5,
            }}), run_id=run_id)
            report = json.loads(tracker.path.read_text())
            self.assertEqual(report["total_usage"]["total_tokens"], 5)
            self.assertTrue(report["usage_complete"])


if __name__ == "__main__":
    unittest.main()
