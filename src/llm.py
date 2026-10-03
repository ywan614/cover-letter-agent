"""初始化模型，不发送请求；显式传入 INI 中的模型配置。"""
import logging
import math
import os
from configparser import ConfigParser

from langchain_openai import ChatOpenAI

from .config import load_config
from .usage import TokenUsageTracker

logger = logging.getLogger("cover_letter_agent.llm")


def create_llm(config: ConfigParser | None = None, *, usage_tracker: TokenUsageTracker | None = None) -> ChatOpenAI:
    if config is None:
        config = load_config()
    values = {}
    for section, key, parameter in (
        ("openai", "openai_api_key", "api_key"),
        ("model", "model_name", "model"),
        ("model", "model_base_url", "base_url"),
    ):
        value = config.get(section, key, fallback="").strip()
        if not value:
            raise ValueError(f"config.ini 缺少 [{section}] {key}")
        values[parameter] = value
    tracing_key = config.get("langsmith", "langsmith_api_key", fallback="").strip()
    if tracing_key:
        os.environ["LANGSMITH_API_KEY"] = tracing_key
    # 仅加载凭据；追踪开关由 LANGSMITH_TRACING 控制。
    # 重试由节点统一控制，避免 SDK 重试与节点重试叠加。
    timeout = float(config.get("model", "timeout_seconds", fallback="").strip() or 300)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("[model] timeout_seconds 必须是有限正数")
    llm = ChatOpenAI(**values, timeout=timeout, max_retries=0,
                     callbacks=[usage_tracker] if usage_tracker is not None else None)
    logger.info("模型客户端初始化完成：model=%s timeout_s=%s sdk_retries=0", values["model"], timeout)
    return llm
