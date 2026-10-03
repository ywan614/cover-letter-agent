"""在真实 graph 内运行指定节点及其上游：uv run test.py --stop-after evaluate。"""
import argparse
import json
import logging
from src.config import PROJECT_ROOT, load_config
from src.llm import create_llm
from src.logger import setup_logging, exception_details
from src.state import create_initial_state
from src.runner import run_interactive
from src.usage import TokenUsageTracker
from src.run_files import add_run_arguments, saved_run
from test_graph.graph import STOP_NODES, build_test_graph


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="在 graph 中逐节点测试，自动执行目标节点的上游")
    parser.add_argument("--stop-after", choices=STOP_NODES, default="analyze_jd",
                        help="执行此节点后结束，默认 analyze_jd")
    add_run_arguments(parser)
    args = parser.parse_args(argv)
    logger = logging.getLogger("cover_letter_agent.test")
    try:
        with saved_run(args, PROJECT_ROOT / "output", stop_after=args.stop_after) as (output_dir, jd_text):
            config = load_config()
            setup_logging(config, output_dir=output_dir)
            logger.info("测试至 %s；输出目录：%s", args.stop_after, output_dir)
            tracker = TokenUsageTracker(output_dir)
            llm = create_llm(config, usage_tracker=tracker)
            graph = build_test_graph(llm, stop_after=args.stop_after, output_dir=output_dir)
            result = run_interactive(graph, create_initial_state(jd_text=jd_text),
                                     thread_id=output_dir.name)
            snapshot = output_dir / "final_state.json"
            snapshot.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            logger.info("测试完成：%s", args.stop_after)
            print(f"最终 state：{snapshot}\nToken 统计：{tracker.path}")
        return 0
    except Exception as error:
        logger.error("测试失败：%s", exception_details(error))
        if isinstance(error, (ValueError, NotImplementedError)):
            print(str(error))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
