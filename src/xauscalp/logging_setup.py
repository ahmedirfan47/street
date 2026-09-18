"""Logging configuration: readable console, full-detail rotating file."""
from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path


def setup_logging(log_path: str = "logs/xauscalp.log", level: str = "INFO") -> None:
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for h in list(root.handlers):
        root.removeHandler(h)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(getattr(logging, level.upper(), logging.INFO))
    console.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s",
                                           datefmt="%Y-%m-%d %H:%M:%S"))
    root.addHandler(console)

    fileh = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=20 * 1024 * 1024, backupCount=6, encoding="utf-8"
    )
    fileh.setLevel(logging.DEBUG)
    fileh.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-7s [%(name)s:%(lineno)d] %(message)s"
    ))
    root.addHandler(fileh)

    for noisy in ("urllib3", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)