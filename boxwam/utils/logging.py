"""Logging helpers: a configured logger, a CSV writer and running averages."""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path

FORMAT = "[%(asctime)s][%(levelname)s] %(message)s"


def get_logger(name: str = "boxwam", *, rank: int = 0) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(FORMAT, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    return logger


class CSVLogger:
    """Append rows to a CSV file, writing the header on first use."""

    def __init__(self, path: str | Path, columns: list[str]):
        self.path = Path(path)
        self.columns = columns
        if not self.path.exists():
            with self.path.open("w", newline="") as handle:
                csv.writer(handle).writerow(columns)

    def log(self, **values) -> None:
        with self.path.open("a", newline="") as handle:
            csv.writer(handle).writerow([values.get(column, "") for column in self.columns])


def append_jsonl(path: str | Path, record: dict) -> None:
    with open(path, "a") as handle:
        handle.write(json.dumps(record) + "\n")


class Average:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def mean(self) -> float:
        return self.total / max(self.count, 1)
