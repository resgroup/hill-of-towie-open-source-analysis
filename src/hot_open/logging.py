"""Helper functions relating to logging."""

import logging
from pathlib import Path


def setup_logger(log_fpath: Path | None = None, level: int = logging.INFO) -> None:
    """Initialize the logger with a file handler and a console handler."""
    log_formatter_file = logging.Formatter("%(asctime)s [%(levelname)-8s]  %(message)s")
    root_logger = logging.getLogger()

    # ensuring no previous handler is active
    while root_logger.hasHandlers():
        root_logger.handlers[0].close  # noqa: B018
        root_logger.removeHandler(root_logger.handlers[0])

    root_logger.setLevel(level)

    if log_fpath is not None:
        file_handler = logging.FileHandler(log_fpath, mode="w")
        file_handler.setFormatter(log_formatter_file)
        root_logger.addHandler(file_handler)

    log_formatter_console = logging.Formatter("%(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter_console)
    root_logger.addHandler(console_handler)
