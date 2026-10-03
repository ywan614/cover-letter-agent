"""复用正式节点，从入口执行到指定节点后结束。"""
from functools import partial
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from src.nodes import llm_call, tool_call
from src.state import AgentState
from src.nodes.human_input import human_input
from src.graph import needs_human_input

STOP_NODES = ("analyze_jd", "evaluate", "prepare", "generate_output", "save_output")


def build_test_graph(llm: BaseChatModel, *, stop_after: str = "analyze_jd",
                     output_dir: Path | None = None):
    if stop_after not in STOP_NODES:
        raise ValueError(f"未知结束节点：{stop_after}")
    builder = StateGraph(AgentState)
    previous = START
    for name in STOP_NODES[:STOP_NODES.index(stop_after) + 1]:
        if name == "save_output":
            node = partial(tool_call.save_output, output_dir=output_dir)
        else:
            node = partial(getattr(llm_call, name), llm=llm)
        builder.add_node(name, node)
        if previous != "evaluate":
            builder.add_edge(previous, name)
        previous = name
    if previous != "evaluate":
        builder.add_edge(previous, END)
    if stop_after != "analyze_jd":
        destination = END if stop_after == "evaluate" else "prepare"
        builder.add_node("human_input", human_input)
        builder.add_conditional_edges(
            "evaluate", needs_human_input,
            {True: "human_input", False: destination},
        )
        builder.add_conditional_edges(
            "human_input", lambda state: bool(state["human_inputs"][-1]["answer"].strip()),
            {True: "evaluate", False: destination},
        )
    return builder.compile(checkpointer=InMemorySaver())
