"""离线检查日志轮转和工作流数据传递。"""
import logging
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.config import load_config
from src.graph import build_graph
from src.llm import create_llm
from src.logger import setup_logging
from src.state import create_initial_state
from langgraph.types import Command


class FrameworkTest(unittest.TestCase):
    def test_logging(self):
        config = load_config()
        with tempfile.TemporaryDirectory() as directory:
            config["logging"].update({"log_dir": directory, "log_console": "false",
                                      "log_max_bytes": "200", "log_backup_count": "2"})
            logger = setup_logging(config)
            try:
                setup_logging(config)
                self.assertEqual(len(logger.handlers), 1)
                for i in range(20):
                    logging.getLogger("cover_letter_agent.test").info("中文日志 %s", i)
                self.assertTrue((Path(directory) / "app.log.1").exists())
                self.assertIn("中文日志", (Path(directory) / "app.log").read_text())
                self.assertLessEqual(len(list(Path(directory).iterdir())), 3)
                previous = Path.cwd()
                try:
                    os.chdir(directory)
                    self.assertTrue(load_config().has_section("model"))
                finally:
                    os.chdir(previous)
            finally:
                for handler in logger.handlers[:]:
                    logger.removeHandler(handler)
                    handler.close()

    def test_graph_and_missing_key(self):
        config = load_config()
        config["openai"]["openai_api_key"] = ""
        with self.assertRaisesRegex(ValueError, "openai_api_key"):
            create_llm(config)
        calls = []

        def analyze(state, *, llm):
            self.assertEqual(state["jd_text"], "Test JD")
            calls.append("analyze")
            return {"jd_analysis": {"summary": "test"}}

        def evaluate(state, *, llm):
            self.assertEqual(state["jd_analysis"]["summary"], "test")
            calls.append("evaluate")
            return {"evaluation": {"status": "missing_info", "summary": "缺信息", "questions": []}}

        def prepare(state, *, llm):
            self.assertEqual(state["evaluation"]["status"], "missing_info")
            self.assertEqual(state["human_inputs"][-1]["answer"], "")
            self.assertIsNone(state["prepared_materials"])
            calls.append("prepare")
            return {"prepared_materials": "基于已有事实的填写建议"}

        def generate(state, *, llm):
            self.assertEqual(state["evaluation"]["status"], "missing_info")
            self.assertEqual(state["prepared_materials"], "基于已有事实的填写建议")
            calls.append("generate")
            return {"output_markdown": "Letter"}

        def save(state):
            self.assertEqual(state["output_markdown"], "Letter")
            calls.append("save")
            return {}

        with patch.multiple("src.graph", analyze_jd=analyze, evaluate=evaluate,
                            prepare=prepare, generate_output=generate, save_output=save):
            graph = build_graph(object())
            config = {"configurable": {"thread_id": "test"}}
            paused = graph.invoke(create_initial_state(jd_text="Test JD"), config=config)
            self.assertIn("__interrupt__", paused)
            graph.invoke(Command(resume=""), config=config)
        self.assertEqual(calls, ["analyze", "evaluate", "prepare", "generate", "save"])


if __name__ == "__main__":
    unittest.main()
