"""Quality gate registry for unified scorecard."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from abevalflow.gates.quality.base import QualityGate

_QUALITY_GATE_REGISTRY: dict[str, type[QualityGate]] = {}


def register_quality_gate(name: str):
    """Decorator to register a quality gate class."""

    def decorator(cls: type[QualityGate]) -> type[QualityGate]:
        _QUALITY_GATE_REGISTRY[name] = cls
        return cls

    return decorator


def get_quality_gate(name: str) -> QualityGate:
    """Get a quality gate instance by name.

    Args:
        name: Gate name (llm-review, etc.)

    Returns:
        Quality gate instance

    Raises:
        KeyError: If gate is not registered
    """
    if name not in _QUALITY_GATE_REGISTRY:
        available = ", ".join(_QUALITY_GATE_REGISTRY.keys())
        raise KeyError(f"Unknown quality gate '{name}'. Available: {available}")
    return _QUALITY_GATE_REGISTRY[name]()


def get_all_quality_gates() -> list[QualityGate]:
    """Get instances of all registered quality gates."""
    return [cls() for cls in _QUALITY_GATE_REGISTRY.values()]


def get_all_quality_gate_names() -> list[str]:
    """Get all registered quality gate names."""
    return list(_QUALITY_GATE_REGISTRY.keys())


from abevalflow.gates.quality.llm_review import LLMReviewGate

__all__ = [
    "register_quality_gate",
    "get_quality_gate",
    "get_all_quality_gates",
    "get_all_quality_gate_names",
    "LLMReviewGate",
]
