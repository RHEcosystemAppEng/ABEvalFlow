"""Shared AEH scoring helpers used by the engine adapter and aggregate script.

Keeps pairwise win-rate / gate logic and numeric judge pass/fail heuristics
aligned across call sites (engine, aggregate_aeh, Tekton PARSE scripts).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_AEH_THRESHOLD = 0.5

# Import path for AEH HarborRunner --environment-import-path (emptyDir mounts).
OPENSHIFT_ENVIRONMENT_IMPORT_PATH = "abevalflow.harbor_extensions.openshift_environment:OpenShiftEnvironment"


def resolve_evaluation_threshold(
    metadata_path: Path | str | None = None,
    *,
    default: float = DEFAULT_AEH_THRESHOLD,
) -> float:
    """Read ``gate_policy.gates.evaluation.threshold`` from metadata.yaml.

    Returns ``default`` when metadata is missing or the threshold is unset.
    """
    if metadata_path is None:
        return default
    path = Path(metadata_path)
    if not path.is_file():
        return default
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError):
        return default
    if not isinstance(data, dict):
        return default
    gate_policy = data.get("gate_policy") or {}
    if not isinstance(gate_policy, dict):
        return default
    gates = gate_policy.get("gates") or {}
    if not isinstance(gates, dict):
        return default
    evaluation = gates.get("evaluation") or {}
    if not isinstance(evaluation, dict):
        return default
    threshold = evaluation.get("threshold")
    if threshold is None:
        return default
    try:
        return float(threshold)
    except (TypeError, ValueError):
        return default


def pairwise_outcome(
    wins_a: int,
    wins_b: int,
    ties: int = 0,
    errors: int = 0,
    *,
    threshold: float = DEFAULT_AEH_THRESHOLD,
) -> dict[str, Any]:
    """Compute honest pairwise win-rate and pass/fail.

    - Denominator includes ties and errors (errors are non-wins).
    - All-ties (no decisive outcomes, no errors) still passes.
    - All-errors does not get the all-ties exception.
    """
    wins_a = int(wins_a or 0)
    wins_b = int(wins_b or 0)
    ties = int(ties or 0)
    errors = int(errors or 0)
    total = wins_a + wins_b + ties + errors
    decisive = wins_a + wins_b
    win_rate = (wins_a / total) if total > 0 else 0.0
    all_ties = decisive == 0 and ties > 0 and errors == 0
    passed = all_ties or (win_rate >= threshold)
    return {
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
        "errors": errors,
        "total": total,
        "decisive": decisive,
        "win_rate": win_rate,
        "all_ties": all_ties,
        "passed": passed,
        "threshold": threshold,
        "recommendation": "pass" if passed else "fail",
    }


def numeric_judge_passes(value: int | float) -> bool:
    """Whether a numeric judge value counts as a pass.

    Scale detection:
    - Integer 1–5 → Likert; pass when ``>= 3``
    - Float in [0, 1] → normalized; pass when ``>= 0.5``
    - Float in (1, 5] → Likert; pass when ``>= 3``
    - Otherwise → normalized fallback (``>= 0.5``)

    Callers must handle booleans separately (``bool`` is a subclass of ``int``).
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and 1 <= value <= 5:
        return value >= 3
    v = float(value)
    if 0.0 <= v <= 1.0:
        return v >= 0.5
    if 1.0 < v <= 5.0:
        return v >= 3.0
    return v >= 0.5


def numeric_judge_is_low(value: int | float) -> bool:
    """Inverse of :func:`numeric_judge_passes` for finding extraction."""
    if isinstance(value, bool):
        return not value
    return not numeric_judge_passes(value)
