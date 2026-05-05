"""
logger.py
---------
Lightweight logging setup shared across all training scripts.

Provides a module-level singleton logger with a console handler, plus helpers
to attach a file handler at runtime once the log path is known.

Usage::

    from logger import logger, add_file_handlers

    add_file_handlers("logger/my_run.log")
    logger.info("training started")
"""

import json
import logging
import os

# ── Constants ─────────────────────────────────────────────────────────────────

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logger")
os.makedirs(LOG_DIR, exist_ok=True)

_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

# ── Singleton logger ──────────────────────────────────────────────────────────

def build_logger() -> logging.Logger:
    """Create (or retrieve) the module logger with a console handler.

    Safe to call multiple times — the console handler is only added once.

    Returns:
        The configured :class:`logging.Logger` instance.
    """
    log = logging.getLogger("train")
    if log.handlers:
        return log

    log.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    log.addHandler(console_handler)
    return log


logger = build_logger()

# ── File handler ──────────────────────────────────────────────────────────────

def add_file_handlers(log_path: str) -> None:
    """Attach a file handler to the global logger.

    Args:
        log_path: target log file path (absolute path recommended).
    """
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    logger.addHandler(file_handler)

# ── Utilities ─────────────────────────────────────────────────────────────────

def print_formated_args(args) -> None:
    """Log all argparse arguments as formatted JSON."""
    formatted_args = json.dumps(vars(args), ensure_ascii=False, indent=4)
    logger.info(f"Args: {formatted_args}")
