"""
TalkBuddy — ThreadPoolExecutor pools for CPU-bound inference tasks.
Separate pools for STT and TTS prevent one from starving the other.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from app.core.logging import get_logger

logger = get_logger()

# Dedicated executor pools — lazily initialized
_stt_executor: Optional[ThreadPoolExecutor] = None
_tts_executor: Optional[ThreadPoolExecutor] = None


def get_stt_executor(max_workers: int = 2) -> ThreadPoolExecutor:
    """Get the STT thread pool executor (created on first call)."""
    global _stt_executor
    if _stt_executor is None:
        _stt_executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="stt")
        logger.debug(f"STT executor created with {max_workers} workers")
    return _stt_executor


def get_tts_executor(max_workers: int = 2) -> ThreadPoolExecutor:
    """Get the TTS thread pool executor (created on first call)."""
    global _tts_executor
    if _tts_executor is None:
        _tts_executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="tts")
        logger.debug(f"TTS executor created with {max_workers} workers")
    return _tts_executor


def shutdown_executors() -> None:
    """Shut down all executor pools. Called during app shutdown."""
    global _stt_executor, _tts_executor

    for name, executor in [("STT", _stt_executor), ("TTS", _tts_executor)]:
        if executor is not None:
            logger.debug(f"Shutting down {name} executor")
            executor.shutdown(wait=False, cancel_futures=True)

    _stt_executor = None
    _tts_executor = None
