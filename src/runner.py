"""终端内处理 LangGraph 暂停和恢复；EOF 不视为空白回答。"""
import logging

from langgraph.types import Command

logger = logging.getLogger("cover_letter_agent.runner")


def run_interactive(graph, initial_state, *, thread_id: str):
    """每轮执行到结束或中断；无中断则返回，有中断则读取回答并恢复。"""
    config = {"configurable": {"thread_id": thread_id}}
    next_input = initial_state
    while True:
        pending = ()  # 每次恢复后重新收集，不能沿用上一轮的中断。
        for update in graph.stream(next_input, config=config, stream_mode="updates"):
            for name in update:
                if name == "__interrupt__":
                    pending = update[name]
                else:
                    logger.info("节点完成：%s", name)
        if not pending:
            return dict(graph.get_state(config).values)
        payload = pending[0].value
        print(f"\n补充材料（{payload['round']}/{payload['max_rounds']}）")
        print(payload["summary"])
        for index, question in enumerate(payload["questions"], 1):
            print(f"{index}. {question}")
        try:
            answer = input("请输入补充材料（可在一行回答多个问题；直接回车则使用现有材料继续）：\n")
        except EOFError as error:
            raise ValueError("输入已断开，未提交补充回答；本次运行结束") from error
        next_input = Command(resume=answer)
