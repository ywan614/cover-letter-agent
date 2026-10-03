"""保存评估后再中断，恢复时不重复模型调用。"""
from langgraph.types import interrupt

from ..state import AgentState, MAX_HUMAN_INPUTS


def human_input(state: AgentState) -> dict:
    evaluation = state["evaluation"]
    answer = interrupt({
        "summary": evaluation["summary"],
        "questions": evaluation["questions"],
        "round": len(state["human_inputs"]) + 1,
        "max_rounds": MAX_HUMAN_INPUTS,
    })
    if not isinstance(answer, str):
        raise ValueError("人工补充必须是字符串；空白字符串表示使用已有材料继续")
    return {"human_inputs": [*state["human_inputs"], {
        "summary": evaluation["summary"],
        "questions": list(evaluation["questions"]),
        "answer": answer,
    }]}
