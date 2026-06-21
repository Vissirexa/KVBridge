from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logger(
    level: str = "info",
    log_file: str | None = None,
    console: bool = True,
) -> logging.Logger:
    logger = logging.getLogger("kvbridge")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger("kvbridge")
