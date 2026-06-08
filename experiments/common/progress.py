from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Iterable, Iterator, Optional, TypeVar

T = TypeVar("T")


def setup_logging(run_dir: Path, level: str = "INFO", quiet: bool = False) -> None:
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%H:%M:%S")

    file_handler = logging.FileHandler(log_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.addHandler(file_handler)

    if not quiet:
        stream_handler = TqdmLoggingHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
        root.addHandler(stream_handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


class TqdmLoggingHandler(logging.StreamHandler):
    def __init__(self) -> None:
        super().__init__(sys.stdout)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            try:
                from tqdm.auto import tqdm

                tqdm.write(message, file=sys.stdout)
            except Exception:
                self.stream.write(message + self.terminator)
                self.flush()
        except Exception:
            self.handleError(record)


def progress(
    iterable: Iterable[T],
    *,
    total: Optional[int] = None,
    desc: str = "",
    unit: str = "it",
    enabled: bool = True,
    leave: bool = False,
) -> Iterable[T]:
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm

        return tqdm(
            iterable,
            total=total,
            desc=desc,
            unit=unit,
            leave=leave,
            dynamic_ncols=True,
            mininterval=0.5,
            smoothing=0.1,
        )
    except Exception:
        return iterable
