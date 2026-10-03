"""离线验证结构化评估、真实 interrupt/resume 及补充上限。"""
import json
import unittest
from unittest.mock import patch

import httpx
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from pydantic import ValidationError

from src.nodes.llm_call import EvaluationOutput, evaluate
from src.runner import run_interactive
from src.state import create_initial_state
from test_graph.graph import build_test_graph


READY = {"status": "ready", "summary": "资料足够", "questions": []}
MISSING = {"status": "missing_info", "summary": "需澄清职责", "questions": ["项目中你具体负责什么？"]}


class EvaluateTest(unittest.TestCase):
    def test_structured_evaluation_and_inputs(self):
        requests = []

        def respond(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={
                "id": "chatcmpl-test", "object": "chat.completion", "created": 0,
                "model": "gpt-4o", "choices": [{"index": 0, "finish_reason": "stop",
                    "message": {"role": "assistant", "content": json.dumps(MISSING)}}],
            })

        state = create_initial_state(jd_text="Build AI")
        state["cv_text"] = "Built an internal AI assistant."
        state["template_text"] = "Write an evidence-based cover letter."
        state["jd_analysis"] = {"summary": "AI"}
        state["human_inputs"] = [{"summary": "此前问题", "questions": ["职责？"], "answer": "开发"}]
        with httpx.Client(transport=httpx.MockTransport(respond)) as client:
            llm = ChatOpenAI(model="gpt-4o", api_key="offline", http_client=client)
            self.assertEqual(evaluate(state, llm=llm), {"evaluation": MISSING})
            self.assertIsNone(state["evaluation"])
            self.assertTrue(requests[0]["response_format"]["json_schema"]["strict"])
            materials = json.loads(requests[0]["messages"][-1]["content"])
            for key in ("jd_text", "jd_analysis", "cv_text", "scenes_text", "template_text", "human_inputs"):
                self.assertEqual(materials[key], state[key])
            state["cv_text"] = " "
            with self.assertRaises(ValueError):
                evaluate(state, llm=llm)
            self.assertEqual(len(requests), 1)
        for invalid in ({**READY, "questions": ["问题"]}, {**MISSING, "questions": []},
                        {**MISSING, "questions": [" "]}, {**READY, "summary": " "}):
            with self.subTest(invalid=invalid), self.assertRaises(ValidationError):
                EvaluationOutput.model_validate(invalid)

    def test_interrupt_routes(self):
        for answers, decisions in (([], [READY]), ([""], [MISSING]),
                                   ([" \n "], [MISSING]), (["我负责部署"], [MISSING, READY]),
                                   (["第一次补充", "第二次补充"], [MISSING, MISSING, MISSING])):
            with self.subTest(answers=answers):
                seen = []

                def assess(state, *, llm):
                    seen.append(list(state["human_inputs"]))
                    return {"evaluation": decisions[len(seen) - 1]}

                with patch("src.nodes.llm_call.analyze_jd", return_value={"jd_analysis": {"summary": "AI"}}) as analyze, \
                        patch("src.nodes.llm_call.evaluate", side_effect=assess):
                    graph = build_test_graph(object(), stop_after="evaluate")
                    config = {"configurable": {"thread_id": "test"}}
                    result = graph.invoke(create_initial_state(jd_text="JD"), config)
                    for index, answer in enumerate(answers):
                        self.assertEqual(result["__interrupt__"][0].value["questions"], MISSING["questions"])
                        self.assertEqual(len(result["human_inputs"]), index)
                        self.assertEqual(result["evaluation"], MISSING)
                        result = graph.invoke(Command(resume=answer), config)
                    self.assertNotIn("__interrupt__", result)
                    self.assertEqual(len(seen), len(decisions))
                    self.assertEqual([item["answer"] for item in result["human_inputs"]], answers)
                    self.assertEqual(result["evaluation"], decisions[-1])
                    self.assertIsNone(result["prepared_materials"])
                    analyze.assert_called_once()

    def test_terminal_loop_exit(self):
        for decision, answers, expected_calls in ((READY, [], 1), (MISSING, ["补充一", "补充二"], 3)):
            with self.subTest(decision=decision), \
                    patch("src.nodes.llm_call.analyze_jd", return_value={"jd_analysis": {"summary": "AI"}}), \
                    patch("src.nodes.llm_call.evaluate", return_value={"evaluation": decision}) as assess, \
                    patch("builtins.print"), patch("builtins.input", side_effect=answers) as read:
                graph = build_test_graph(object(), stop_after="evaluate")
                result = run_interactive(graph, create_initial_state(jd_text="JD"), thread_id="exit")
                self.assertEqual(len(result["human_inputs"]), len(answers))
                self.assertEqual(read.call_count, len(answers))
                self.assertEqual(assess.call_count, expected_calls)
        with patch("src.nodes.llm_call.analyze_jd", side_effect=RuntimeError("模型失败")), \
                patch("builtins.input") as read:
            graph = build_test_graph(object())
            with self.assertRaisesRegex(RuntimeError, "模型失败"):
                run_interactive(graph, create_initial_state(jd_text="JD"), thread_id="failure")
            read.assert_not_called()

    def test_terminal_resume_and_eof(self):
        with patch("src.nodes.llm_call.analyze_jd", return_value={"jd_analysis": {"summary": "AI"}}), \
                patch("src.nodes.llm_call.evaluate", return_value={"evaluation": MISSING}) as assess, \
                patch("builtins.print"), patch("builtins.input", return_value=""):
            graph = build_test_graph(object(), stop_after="evaluate")
            result = run_interactive(graph, create_initial_state(jd_text="JD"), thread_id="terminal")
            self.assertEqual(result["human_inputs"][0]["answer"], "")
            assess.assert_called_once()
            with patch("builtins.input", side_effect=EOFError), self.assertRaisesRegex(ValueError, "输入已断开"):
                run_interactive(graph, create_initial_state(jd_text="JD"), thread_id="eof")
            self.assertEqual(graph.get_state({"configurable": {"thread_id": "eof"}}).values["human_inputs"], [])
            with self.assertRaisesRegex(ValueError, "字符串"):
                graph.invoke(Command(resume=123), {"configurable": {"thread_id": "eof"}})


if __name__ == "__main__":
    unittest.main()
