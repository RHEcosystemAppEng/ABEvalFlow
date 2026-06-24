"""Security gate registry for unified scorecard."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from abevalflow.gates.security.base import SecurityGate

_SECURITY_GATE_REGISTRY: dict[str, type[SecurityGate]] = {}


def register_security_gate(name: str):
    """Decorator to register a security gate class."""

    def decorator(cls: type[SecurityGate]) -> type[SecurityGate]:
        _SECURITY_GATE_REGISTRY[name] = cls
        return cls

    return decorator


def get_security_gate(name: str) -> SecurityGate:
    """Get a security gate instance by name.

    Args:
        name: Gate name (cisco, snyk, etc.)

    Returns:
        Security gate instance

    Raises:
        KeyError: If gate is not registered
    """
    if name not in _SECURITY_GATE_REGISTRY:
        available = ", ".join(_SECURITY_GATE_REGISTRY.keys())
        raise KeyError(f"Unknown security gate '{name}'. Available: {available}")
    return _SECURITY_GATE_REGISTRY[name]()


def get_all_security_gates() -> list[SecurityGate]:
    """Get instances of all registered security gates."""
    return [cls() for cls in _SECURITY_GATE_REGISTRY.values()]


def get_all_security_gate_names() -> list[str]:
    """Get all registered security gate names."""
    return list(_SECURITY_GATE_REGISTRY.keys())


from abevalflow.gates.security.cisco import CiscoGate

__all__ = [
    "register_security_gate",
    "get_security_gate",
    "get_all_security_gates",
    "get_all_security_gate_names",
    "CiscoGate",
]
