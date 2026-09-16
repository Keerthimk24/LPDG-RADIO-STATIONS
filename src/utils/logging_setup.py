"""
Logging setup — structured logging configuration.

Provides a consistent logging format across all modules.
"""

from __future__ import annotations

import logging
import os


def setup_logging(level: str | None = None) -> None:
    """Configure structured logging for the application.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR).
               Defaults to LOG_LEVEL env var or INFO.
    """
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO")

    log_level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    # Suppress noisy third-party loggers
    logging.getLogger("lightgbm").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.info("Logging configured: level=%s", level.upper())
