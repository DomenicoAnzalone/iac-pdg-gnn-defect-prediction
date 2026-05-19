import logging
import sys
from typing import Optional


def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO):
    logger = logging.getLogger()
    logger.setLevel(level)

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")

    # console
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
