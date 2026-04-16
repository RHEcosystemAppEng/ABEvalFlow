"""Pluggable observer protocol for evaluation results.

Observers are notified after results are persisted to the database.
They may log to MLflow, push to Langfuse, emit OTel spans, post Slack
notifications, etc.

Observer failures are logged as warnings and never fail the pipeline.
"""

from __future__ import annotations

import logging
import uuid
from typing import Protocol

from abevalflow.report import AnalysisResult

logger = logging.getLogger(__name__)


class ResultsObserver(Protocol):
    """Interface for downstream result consumers."""

    def on_evaluation_stored(
        self,
        result: AnalysisResult,
        run_id: uuid.UUID,
    ) -> None:
        """Called after a successful DB commit.

        Args:
            result: The validated analysis result that was persisted.
            run_id: The UUID primary key of the stored evaluation run.
        """
        ...


def _discover_observers() -> list[ResultsObserver]:
    """Discover and instantiate observers based on environment variables.

    All matching observers are loaded (not first-match). Returns an empty
    list if no observer env vars are configured.
    """
    import os

    observers: list[ResultsObserver] = []

    if os.environ.get("MLFLOW_TRACKING_URI"):
        logger.info("MLflow observer requested but not yet implemented — skipping")

    if os.environ.get("LANGFUSE_PUBLIC_KEY"):
        logger.info("Langfuse observer requested but not yet implemented — skipping")

    return observers


def notify_observers(
    observers: list[ResultsObserver],
    result: AnalysisResult,
    run_id: uuid.UUID,
) -> None:
    """Invoke all observers, catching and logging any errors."""
    for obs in observers:
        try:
            obs.on_evaluation_stored(result, run_id)
        except Exception:
            logger.warning(
                "Observer %s.%s failed",
                type(obs).__module__,
                type(obs).__qualname__,
                exc_info=True,
            )
