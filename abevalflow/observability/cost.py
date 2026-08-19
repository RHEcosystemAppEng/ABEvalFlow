"""Cost estimation from token counts and per-model pricing.

Reads model_costs.yaml for per-model rates (USD per 1K tokens).
Lookup order: exact match → prefix match → _default entry → zero.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_COSTS_PATH = Path(__file__).resolve().parents[2] / "config" / "observability" / "model_costs.yaml"


@lru_cache(maxsize=1)
def _load_model_costs(path: Path = _COSTS_PATH) -> dict[str, dict[str, float]]:
    if not path.is_file():
        logger.warning("Model costs file not found: %s", path)
        return {}
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    return {k: v for k, v in data.items() if isinstance(v, dict)}


def _resolve_rates(model_name: str, costs: dict[str, dict[str, float]]) -> dict[str, float]:
    if model_name in costs:
        return costs[model_name]

    for key in costs:
        if key == "_default":
            continue
        if model_name.startswith(key) or key.startswith(model_name):
            return costs[key]

    return costs.get("_default", {})


def calculate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    model_name: str | None,
    costs_path: Path | None = None,
) -> float | None:
    """Estimate USD cost from token counts and model name.

    Returns None if model_name is not provided or no pricing is available.
    """
    if not model_name:
        return None

    costs = _load_model_costs(costs_path or _COSTS_PATH)
    if not costs:
        return None

    rates = _resolve_rates(model_name, costs)
    if not rates:
        return None

    input_rate = rates.get("input_per_1k", 0)
    output_rate = rates.get("output_per_1k", 0)

    return round((prompt_tokens / 1000) * input_rate + (completion_tokens / 1000) * output_rate, 6)
