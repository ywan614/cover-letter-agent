"""在真实图中验证停止边界、状态传递及测试入口输出。"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

import test as runner
from src.state import create_initial_state
from test_graph.graph import STOP_NODES, build_test_graph


class TestGraphTest(unittest.TestCase):
    def test_stop_boundaries(self):
        for stop in STOP_NODES:
            with self.subTest(stop=stop):
                calls = []

                def analyze(state, *, llm):
                    calls.append("analyze_jd")
                    return {"jd_analysis": {"summary": "analysis"}}

                def evaluate(state, *, llm):
                    self.assertEqual(state["jd_analysis"]["summary"], "analysis")
                    calls.append("evaluate")
                    return {"evaluation": {"status": "ready", "summary": "ok", "questions": []}}

                def prepare(state, *, llm):
                    self.assertEqual(state["evaluation"]["status"], "ready")
                    self.assertIsNone(state["prepared_materials"])
                    calls.append("prepare")
                    return {"prepared_materials": "填写建议"}

                def generate(state, *, llm):
                    self.assertEqual(state["prepared_materials"], "填写建议")
                    calls.append("generate_output")
                    return {"output_markdown": "Letter"}

                def save(state, *, output_dir):
                    self.assertEqual(state["output_markdown"], "Letter")
                    self.assertEqual(output_dir, Path("example"))
                    calls.append("save_output")
                    return {}

                with patch.multiple("src.nodes.llm_call", analyze_jd=analyze,
                                    evaluate=evaluate, prepare=prepare, generate_output=generate), \
                        patch("src.nodes.tool_call.save_output", save):
                    graph = build_test_graph(object(), stop_after=stop, output_dir=Path("example"))
                    graph.invoke(create_initial_state(jd_text="JD"),
                                 config={"configurable": {"thread_id": "test"}})
                self.assertEqual(calls, list(STOP_NODES[:STOP_NODES.index(stop) + 1]))
        with self.assertRaises(ValueError):
            build_test_graph(object(), stop_after="invalid")

    def test_entrypoint_outputs_and_usage(self):
        def fake_llm(config, *, usage_tracker):
            return FakeMessagesListChatModel(callbacks=[usage_tracker], responses=[
                AIMessage(content="analysis", usage_metadata={
                    "input_tokens": 3, "output_tokens": 2, "total_tokens": 5,
                }),
            ])

        def analyze(state, *, llm):
            return {"jd_analysis": {"summary": llm.invoke(state["jd_text"]).content}}

        with tempfile.TemporaryDirectory() as directory, \
                patch.object(runner, "PROJECT_ROOT", Path(directory)), \
                patch.object(runner, "setup_logging"), \
                patch.object(runner, "create_llm", fake_llm), \
                patch("src.nodes.llm_call.analyze_jd", analyze):
            jd_file = Path(directory) / "job.txt"
            jd_file.write_text("Example JD")
            self.assertEqual(runner.main(["--jd-file", str(jd_file)]), 0)
            output_dir, = (Path(directory) / "output").iterdir()
            state = json.loads((output_dir / "final_state.json").read_text())
            usage = json.loads((output_dir / "token_usage.json").read_text())
            self.assertEqual(state["jd_analysis"]["summary"], "analysis")
            self.assertEqual(usage["calls"][0]["node"], "analyze_jd")
            self.assertEqual(usage["total_usage"]["total_tokens"], 5)
            with patch("src.nodes.llm_call.analyze_jd", side_effect=NotImplementedError("not ready")):
                self.assertEqual(runner.main(["--jd-file", str(jd_file)]), 1)
            self.assertEqual(len(list((Path(directory) / "output").glob("*/final_state.json"))), 1)


if __name__ == "__main__":
    unittest.main()
