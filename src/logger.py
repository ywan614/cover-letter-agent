"""应用日志写入 log/，支持轮转和控制台输出。"""
import logging
from configparser import ConfigParser
from logging.handlers import RotatingFileHandler
import traceback

from .config import PROJECT_ROOT


def exception_details(error: BaseException) -> dict:
    """保留异常链、HTTP 标识和栈位置，不收集正文、请求、局部变量或源码行。"""
    chain = []
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        item = {"type": type(error).__name__}
        for key in ("status_code", "request_id", "code"):
            value = getattr(error, key, None)
            if isinstance(value, (str, int)):
                item[key] = str(value)[:200]
        item["stack"] = [
            f"{frame.filename}:{frame.lineno} in {frame.name}"
            for frame in traceback.extract_tb(error.__traceback__)
        ]
        chain.append(item)
        error = error.__cause__ or (None if error.__suppress_context__ else error.__context__)
    return {"error_chain": chain}


def setup_logging(config: ConfigParser, *, output_dir=None) -> logging.Logger:
    settings = config["logging"]
    log_dir = output_dir if output_dir is not None else PROJECT_ROOT / (settings.get("log_dir") or "log")
    max_bytes = int(settings.get("log_max_bytes") or 10485760)
    backups = int(settings.get("log_backup_count") or 5)
    if max_bytes <= 0 or backups < 1:
        raise ValueError("日志轮转大小和备份数量必须大于 0")
    logger = logging.getLogger("cover_letter_agent")
    logger.setLevel((settings.get("log_level") or "INFO").upper())
    logger.propagate = False
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        log_dir / ("run.log" if output_dir is not None else settings.get("log_file_name") or "app.log"),
        maxBytes=max_bytes, backupCount=backups, encoding="utf-8",
    )
    for old_handler in logger.handlers[:]:
        logger.removeHandler(old_handler)
        old_handler.close()
    logger.addHandler(handler)
    if settings.get("log_console", "").strip().lower() not in ("false", "no", "0", "off"):
        logger.addHandler(logging.StreamHandler())
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    for handler in logger.handlers:
        handler.setFormatter(formatter)
    return logger
