"""自动读取 JD；--check 仅检查框架，不调用模型。"""
import argparse
import json
import logging
from src.config import PROJECT_ROOT, load_config
from src.graph import build_graph
from src.llm import create_llm
from src.logger import setup_logging, exception_details
from src.state import create_initial_state
from src.runner import run_interactive
from src.usage import TokenUsageTracker
from src.run_files import add_run_arguments, saved_run


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="自动读取 data/jd.py")
    parser.add_argument("--check", action="store_true", help="检查初始化，不执行业务节点")
    add_run_arguments(parser)
    args = parser.parse_args(argv)
    logger = logging.getLogger("cover_letter_agent.main")
    try:
        if args.check:
            config = load_config()
            setup_logging(config)
            llm = create_llm(config)
            build_graph(llm)
            logger.info("框架检查通过；未调用模型、未执行业务节点")
            return 0
        with saved_run(args, PROJECT_ROOT / "output") as (output_dir, jd_text):
            config = load_config()
            setup_logging(config, output_dir=output_dir)
            logger.info("启动工作流；输出目录：%s", output_dir)
            tracker = TokenUsageTracker(output_dir)
            llm = create_llm(config, usage_tracker=tracker)
            graph = build_graph(llm, output_dir=output_dir)
            state = create_initial_state(jd_text=jd_text)
            result = run_interactive(graph, state, thread_id=output_dir.name)
            snapshot = output_dir / "final_state.json"
            snapshot.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            logger.info("工作流完成")
            print(f"最终 state：{snapshot}\nToken 统计：{tracker.path}")
        return 0
    except Exception as error:
        # 不将第三方异常正文或个人资料写入日志。
        logger.error("运行失败：%s", exception_details(error))
        if isinstance(error, (ValueError, NotImplementedError)):
            print(str(error))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
