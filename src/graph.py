"""支持人工补充的流程，模型实例不进入业务 state。"""
import logging
from functools import partial
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from .nodes.llm_call import analyze_jd, evaluate, prepare, generate_output
from .nodes.tool_call import save_output
from .state import AgentState, MAX_HUMAN_INPUTS
from .nodes.human_input import human_input

logger = logging.getLogger("cover_letter_agent.graph")


def needs_human_input(state: AgentState) -> bool:
    return (
        state["evaluation"]["status"] == "missing_info"
        and len(state["human_inputs"]) < MAX_HUMAN_INPUTS
    )


def build_graph(llm: BaseChatModel, *, output_dir: Path | None = None):
    builder = StateGraph(AgentState)
    builder.add_node("analyze_jd", partial(analyze_jd, llm=llm))
    builder.add_node("evaluate", partial(evaluate, llm=llm))
    builder.add_node("human_input", human_input)
    builder.add_node("prepare", partial(prepare, llm=llm))
    builder.add_node("generate_output", partial(generate_output, llm=llm))
    builder.add_node("save_output", partial(save_output, output_dir=output_dir) if output_dir else save_output)
    builder.add_edge(START, "analyze_jd")
    builder.add_edge("analyze_jd", "evaluate")
    builder.add_conditional_edges(
        "evaluate", needs_human_input,
        {True: "human_input", False: "prepare"},
    )
    builder.add_conditional_edges(
        "human_input", lambda state: bool(state["human_inputs"][-1]["answer"].strip()),
        {True: "evaluate", False: "prepare"},
    )
    builder.add_edge("prepare", "generate_output")
    builder.add_edge("generate_output", "save_output")
    builder.add_edge("save_output", END)
    graph = builder.compile(checkpointer=InMemorySaver())
    logger.info("工作流编译完成")
    return graph
