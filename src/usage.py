"""每次运行独立记录服务端 token usage，不记录提示词和模型正文。"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from time import perf_counter
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from .logger import exception_details

logger = logging.getLogger("cover_letter_agent.usage")


class TokenUsageTracker(BaseCallbackHandler):
    # 写盘失败应使调用报错，避免静默丢失统计。
    raise_error = True
    run_inline = True

    def __init__(self, output_dir: Path):
        self.path = output_dir / "token_usage.json"
        self.calls: dict[str, dict] = {}
        self.started: dict[str, float] = {}
        self.lock = RLock()
        output_dir.mkdir(parents=True, exist_ok=True)
        self._save()

    def on_chat_model_start(self, serialized, messages, *, run_id: UUID,
                            metadata=None, **kwargs):
        metadata = metadata or {}
        with self.lock:
            self.started[str(run_id)] = perf_counter()
            self.calls[str(run_id)] = {
                "call_id": str(run_id),
                "node": metadata.get("langgraph_node"),
                "model": metadata.get("ls_model_name"),
                "started_at": datetime.now(timezone.utc).isoformat(),
                "status": "running", "usage": None,
            }
            self._save()

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs):
        # 当前框架每次 invoke/ainvoke 一个输入。优先取标准 usage_metadata。
        usage = None
        if response.generations and response.generations[0]:
            message = getattr(response.generations[0][0], "message", None)
            usage = getattr(message, "usage_metadata", None)
        if usage is None:
            raw = (response.llm_output or {}).get("token_usage")
            if raw:
                usage = {
                    "input_tokens": raw.get("prompt_tokens"),
                    "output_tokens": raw.get("completion_tokens"),
                    "total_tokens": raw.get("total_tokens"),
                }
        with self.lock:
            call = self.calls[str(run_id)]
            call.update(status="completed", usage=usage,
                        elapsed_s=round(perf_counter() - self.started.pop(str(run_id)), 3),
                        finished_at=datetime.now(timezone.utc).isoformat())
            logger.info("模型响应完成：node=%s call_id=%s elapsed_s=%s usage=%s",
                        call["node"], run_id, call["elapsed_s"], usage)
            self._save()

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs):
        with self.lock:
            call = self.calls[str(run_id)]
            call.update(status="failed", error_type=type(error).__name__,
                        elapsed_s=round(perf_counter() - self.started.pop(str(run_id)), 3),
                        finished_at=datetime.now(timezone.utc).isoformat(),
                        **exception_details(error))
            logger.error("模型响应失败：node=%s call_id=%s elapsed_s=%s details=%s",
                         call["node"], run_id, call["elapsed_s"], call["error_chain"])
            self._save()

    def _save(self):
        calls = list(self.calls.values())
        known = {key: 0 for key in ("input_tokens", "output_tokens", "total_tokens")}
        complete = True
        for call in calls:
            usage = call["usage"] or {}
            for key in known:
                value = usage.get(key)
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    known[key] += value
                else:
                    complete = False
        report = {
            "call_count": len(calls),
            "usage_complete": complete,
            "total_usage": known if complete else None,
            "known_usage": known,
            "calls": calls,
        }
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.path)
