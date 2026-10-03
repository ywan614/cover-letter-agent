"""启动前存档 JD；每次尝试使用独立目录。"""
from contextlib import contextmanager
from datetime import datetime
import json
from pathlib import Path
import re
from tempfile import mkdtemp


def add_run_arguments(parser):
    parser.add_argument("--name", help="手动命名，例如 ExampleCo__AI-Engineer；默认 untitled")
    parser.add_argument("--jd-file", type=Path, help="读取已保存的 UTF-8 JD 文件，相对路径基于当前目录")


@contextmanager
def saved_run(args, output_root, *, stop_after=None):
    if args.jd_file is None:
        from data.jd import JD_TEXT
        jd_text = JD_TEXT
        source = str(Path(__file__).resolve().parent.parent / "data" / "jd.py")
    else:
        jd_text = args.jd_file.read_text(encoding="utf-8")
        source = str(args.jd_file.resolve())
    if not jd_text.strip():
        raise ValueError("JD 不能为空，请填写 data/jd.py 或指定 --jd-file")
    name = args.name or "untitled"
    # 保留中文和双下划线；限制字节长度，避免超出文件系统目录名上限。
    label = re.sub(r"[^\w-]+", "-", name).strip("-_") or "untitled"
    label = label.encode("utf-8")[:160].decode("utf-8", errors="ignore")
    output_root.mkdir(parents=True, exist_ok=True)
    output_dir = Path(mkdtemp(prefix=f"{datetime.now():%Y%m%d_%H%M%S}__{label}__", dir=output_root))
    (output_dir / "jd.txt").write_text(jd_text, encoding="utf-8")
    record = {"name": name, "jd_source": source, "stop_after": stop_after,
              "started_at": datetime.now().astimezone().isoformat(), "status": "running"}

    def save_record():
        temporary = output_dir / "run.json.tmp"
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output_dir / "run.json")

    save_record()
    print(f"本次输出目录：{output_dir}", flush=True)
    try:
        yield output_dir, jd_text
    except BaseException as error:
        record.update(status="interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
                      error_type=type(error).__name__)
        raise
    else:
        record["status"] = "completed"
    finally:
        record["finished_at"] = datetime.now().astimezone().isoformat()
        save_record()
