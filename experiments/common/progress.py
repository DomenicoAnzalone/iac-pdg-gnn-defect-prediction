from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path
from typing import Iterable, Optional, TypeVar

T = TypeVar("T")


def setup_logging(run_dir: Path, level: str = "INFO", quiet: bool = False, console_mode: str = "standard") -> None:
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
        if console_mode == "compact":
            stream_handler.addFilter(CompactConsoleFilter())
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


class CompactConsoleFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return bool(getattr(record, "console", False))


def progress(
    iterable: Iterable[T],
    *,
    total: Optional[int] = None,
    desc: str = "",
    unit: str = "it",
    enabled: bool = True,
    leave: bool = False,
    position: Optional[int] = None,
    dynamic_ncols: bool = True,
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
            position=position,
            file=sys.stdout,
            dynamic_ncols=dynamic_ncols,
            mininterval=0.5,
            smoothing=0.1,
        )
    except Exception:
        return iterable


class CompactStatusLine:
    """Single-line progress display that tolerates terminal resizing."""

    def __init__(self, total_splits: int, max_width: int = 100, enabled: bool = True):
        self.total_splits = max(total_splits, 1)
        self.max_width = max_width
        self.enabled = enabled
        self.last_len = 0

    def update(
        self,
        *,
        split_index: int,
        completed_splits: int | None = None,
        epoch: int | None = None,
        total_epochs: int | None = None,
        loss: str | None = None,
        best: str | None = None,
        patience: str | None = None,
        mcc: str | None = None,
        avg: str | None = None,
        eta: str | None = None,
        status: str | None = None,
    ) -> None:
        if not self.enabled:
            return
        completed = split_index if completed_splits is None else completed_splits
        pct = 100.0 * min(max(completed, 0), self.total_splits) / self.total_splits
        parts = [f"Split {split_index}/{self.total_splits} done={completed} {pct:5.1f}%"]
        if epoch is not None and total_epochs is not None:
            parts.append(f"Epoch {epoch}/{total_epochs}")
        if loss is not None:
            parts.append(f"loss={loss}")
        if best is not None:
            parts.append(f"best={best}")
        if patience is not None:
            parts.append(f"pat={patience}")
        if mcc is not None:
            parts.append(f"mcc={mcc}")
        if avg is not None:
            parts.append(f"avg={avg}")
        if eta is not None:
            parts.append(f"eta={eta}")
        if status is not None:
            parts.append(status)
        self._write(" | ".join(parts))

    def close(self) -> None:
        if not self.enabled:
            return
        sys.stdout.write("\n")
        sys.stdout.flush()
        self.last_len = 0

    def _write(self, text: str) -> None:
        width = min(self.max_width, max(40, shutil.get_terminal_size(fallback=(self.max_width, 20)).columns - 1))
        if len(text) > width:
            text = text[: max(0, width - 3)] + "..."
        padded = text.ljust(max(self.last_len, len(text)))
        sys.stdout.write("\r" + padded)
        sys.stdout.flush()
        self.last_len = len(padded)
