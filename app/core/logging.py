"""
TalkBuddy — Structured logging setup using loguru.
"""

import sys
from loguru import logger


def setup_logging(log_level: str = "INFO") -> None:
    """
    Configure loguru logger with structured output format.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    # Remove default handler
    logger.remove()

    # Add custom handler with structured format
    logger.add(
        sys.stdout,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
        colorize=True,
    )

    # Optional: file logging for debugging
    # logger.add(
    #     "logs/talkbuddy_{time:YYYY-MM-DD}.log",
    #     level="DEBUG",
    #     rotation="1 day",
    #     retention="7 days",
    #     compression="zip",
    # )


# Initialize logging on import
def get_logger():
    """Get the configured logger instance."""
    return logger
