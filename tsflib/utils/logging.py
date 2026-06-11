"""
Logging Utilities for TSFLib

This module provides logging utilities for the library.
"""

import logging
import sys
from typing import Optional

from lightning.fabric.utilities import rank_zero_only


class RankedLogger(logging.LoggerAdapter):
    """A logger adapter that handles rank information for distributed training.

    This is adapted from Lightning's RankedLogger to provide consistent
    logging across the library.

    Args:
        name: Logger name.
        rank_zero_only: If True, only log on rank 0.

    Example:
        >>> logger = RankedLogger(__name__, rank_zero_only=True)
        >>> logger.info("This will only print on rank 0")
    """

    def __init__(self, name: str = __name__, rank_zero_only: bool = False):
        logger = logging.getLogger(name)
        super().__init__(logger, {})
        self.rank_zero_only = rank_zero_only

    def log(self, level: int, msg: str, *args, **kwargs) -> None:
        """Log a message with the given level."""
        if self.rank_zero_only:
            return rank_zero_only(self.logger.log)(level, msg, *args, **kwargs)
        self.logger.log(level, msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs) -> None:
        """Log an info message."""
        self.log(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        """Log a warning message."""
        self.log(logging.WARNING, msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs) -> None:
        """Log an error message."""
        self.log(logging.ERROR, msg, *args, **kwargs)

    def debug(self, msg: str, *args, **kwargs) -> None:
        """Log a debug message."""
        self.log(logging.DEBUG, msg, *args, **kwargs)


def get_logger(
    name: str = "tsflib",
    level: int = logging.INFO,
    rank_zero_only: bool = True,
    format_string: Optional[str] = None,
) -> RankedLogger:
    """Get a logger with the specified configuration.

    Args:
        name: Logger name.
        level: Logging level.
        rank_zero_only: If True, only log on rank 0.
        format_string: Custom format string.

    Returns:
        Configured logger.

    Example:
        >>> logger = get_logger("my_module", level=logging.DEBUG)
        >>> logger.info("Hello, world!")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers = []

    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    # Set format
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(format_string)
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return RankedLogger(name, rank_zero_only=rank_zero_only)


# Default logger for the library
default_logger = get_logger("tsflib", rank_zero_only=True)


__all__ = [
    "RankedLogger",
    "get_logger",
    "default_logger",
]
