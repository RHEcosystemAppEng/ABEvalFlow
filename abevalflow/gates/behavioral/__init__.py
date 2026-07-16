"""Behavioral gate registry for unified scorecard."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from abevalflow.gates.behavioral.base import BehavioralGate

_BEHAVIORAL_GATE_REGISTRY: dict[str, type[BehavioralGate]] = {}


def register_behavioral_gate(name: str):
    """Decorator to register a behavioral gate class."""

    def decorator(cls: type[BehavioralGate]) -> type[BehavioralGate]:
        _BEHAVIORAL_GATE_REGISTRY[name] = cls
        return cls

    return decorator


def get_behavioral_gate(name: str) -> BehavioralGate:
    """Get a behavioral gate instance by name."""
    if name not in _BEHAVIORAL_GATE_REGISTRY:
        available = ", ".join(_BEHAVIORAL_GATE_REGISTRY.keys())
        raise KeyError(f"Unknown behavioral gate '{name}'. Available: {available}")
    return _BEHAVIORAL_GATE_REGISTRY[name]()


def get_all_behavioral_gates() -> list[BehavioralGate]:
    """Get instances of all registered behavioral gates."""
    return [cls() for cls in _BEHAVIORAL_GATE_REGISTRY.values()]


def get_all_behavioral_gate_names() -> list[str]:
    """Get all registered behavioral gate names."""
    return list(_BEHAVIORAL_GATE_REGISTRY.keys())


from abevalflow.gates.behavioral.edge_case import EdgeCaseGate

__all__ = [
    "register_behavioral_gate",
    "get_behavioral_gate",
    "get_all_behavioral_gates",
    "get_all_behavioral_gate_names",
    "EdgeCaseGate",
]
