"""Logging helpers shared by discovery workers and CLI commands."""

from __future__ import annotations

import logging


class WorkerLogAdapter(logging.LoggerAdapter):
    """Prefix worker messages with their name and optional batch ID."""

    def __init__(
        self,
        logger: logging.Logger,
        worker_name: str,
        batch_id: str | None = None,
    ) -> None:
        super().__init__(logger, {"worker_name": worker_name, "batch_id": batch_id})

    def set_batch(self, batch_id: str | None) -> None:
        """Update the batch ID for subsequent log messages."""
        self.extra["batch_id"] = batch_id

    def process(self, msg: str, kwargs: dict) -> tuple[str, dict]:
        worker = self.extra.get("worker_name", "")
        batch = self.extra.get("batch_id")
        batch_str = f" [batch={batch}]" if batch else ""
        return f"{worker}{batch_str}: {msg}", kwargs


def log_worker_error(
    logger: logging.Logger | logging.LoggerAdapter,
    *,
    worker_name: str,
    signal_id: str | None = None,
    error: Exception,
    error_type: str = "application",
    retry_count: int = 0,
    max_retries: int = 0,
    batch_id: str | None = None,
) -> None:
    """Log a worker error with consistent context and severity."""
    parts = [worker_name]
    if batch_id:
        parts.append(f"batch={batch_id}")
    if signal_id:
        parts.append(f"signal={signal_id}")
    parts.append(f"type={error_type}")
    if max_retries > 0:
        parts.append(f"retry={retry_count}/{max_retries}")

    context = " ".join(parts)
    level = logging.WARNING if error_type == "infrastructure" else logging.ERROR
    logger.log(level, "%s: %s", context, error)
