"""
Centralized logging for LLM request/response tracing.

Goals:
- Always write to absolute `{repo_root}/logs` (resolved at runtime).
- Avoid import-time side effects (no file creation on module import).
- Ensure the file is never empty (write a header immediately when configured).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Union

from .log_paths import get_logs_dir


LLM_LOGGER_NAME = "llm_responses"


def configure_llm_responses_logger(
    *,
    repo_root: Optional[Union[str, Path]] = None,
    log_file: Optional[Union[str, Path]] = None,
    timestamp: Optional[str] = None,
    level: int = logging.DEBUG,
) -> Tuple[logging.Logger, Path]:
    """
    Configure and return the `llm_responses` logger.

    If `log_file` is not provided, creates a timestamped file in `{repo_root}/logs`.
    """
    logger = logging.getLogger(LLM_LOGGER_NAME)
    logger.setLevel(level)

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if log_file is None:
        logs_dir = get_logs_dir(repo_root)
        log_path = logs_dir / f"llm_responses_{timestamp}.log"
    else:
        log_path = Path(log_file).expanduser().resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)

    # If already configured for this exact path, keep it (avoid duplicate handlers).
    for h in list(logger.handlers):
        if isinstance(h, logging.FileHandler) and Path(getattr(h, "baseFilename", "")).resolve() == log_path:
            logger.propagate = False
            _write_llm_log_header(log_path, timestamp)
            logger.debug("LLM logger already configured (reusing handler).")
            return logger, log_path

    # Remove old handlers (we want exactly one authoritative destination).
    for h in list(logger.handlers):
        logger.removeHandler(h)

    class ImmediateFlushingFileHandler(logging.FileHandler):
        def emit(self, record):
            super().emit(record)
            self.flush()

    handler = ImmediateFlushingFileHandler(log_path)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)

    # Keep LLM logs out of root handlers unless the caller explicitly wants them.
    logger.propagate = False

    _write_llm_log_header(log_path, timestamp)
    logger.debug("LLM logger configured.")
    return logger, log_path


def ensure_llm_logger_configured(repo_root: Optional[Union[str, Path]] = None) -> logging.Logger:
    """Idempotently ensure `llm_responses` logger has a file handler."""
    logger = logging.getLogger(LLM_LOGGER_NAME)
    if any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        return logger
    configure_llm_responses_logger(repo_root=repo_root)
    return logging.getLogger(LLM_LOGGER_NAME)


def _write_llm_log_header(log_path: Path, timestamp: str) -> None:
    # Guarantee the file is non-empty even if no LLM call occurs.
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            if f.tell() == 0:
                f.write(f"LLM Responses Log initialized at {timestamp}\n")
                f.write(f"PID: {os.getpid()}\n")
                f.write(f"CWD: {os.getcwd()}\n")
                f.write("=" * 80 + "\n")
                f.flush()
    except Exception:
        # Never raise from logging utilities.
        pass

