"""Observability decorators for timing and tracing."""

from __future__ import annotations

import functools
import logging
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


def timed_gate(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator that logs gate execution time in milliseconds.

    Attaches ``_duration_ms`` to the return value if it has that attribute,
    otherwise logs only.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.time()
        result = func(*args, **kwargs)
        duration_ms = int((time.time() - start) * 1000)
        logger.info("Gate %s executed in %dms", func.__qualname__, duration_ms)
        if hasattr(result, "_duration_ms"):
            result._duration_ms = duration_ms
        return result

    return wrapper
