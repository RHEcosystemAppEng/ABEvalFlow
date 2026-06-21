"""Client for pushing gate results to Red Hat Compass Facts API.

The Compass Soundcheck Facts API accepts evaluation results that can be
displayed in the Compass developer portal alongside other component metrics.

Usage:
    from abevalflow.compass_facts import push_gate_fact
    from abevalflow.gates.base import GateResult

    push_gate_fact(
        gate_result=result,
        endpoint="https://compass.stage.redhat.com/api/soundcheck/facts/",
        entity_ref="component:default/my-skill",
        fact_ref="catalog:default/abevalflow_harbor",
    )
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from abevalflow.gates.base import GateResult
    from abevalflow.schemas import PushFactsConfig

logger = logging.getLogger(__name__)


def _build_fact_payload(
    gate_result: "GateResult",
    entity_ref: str,
    fact_ref: str,
) -> dict[str, Any]:
    """Build Soundcheck fact payload from gate result.

    Args:
        gate_result: The gate evaluation result
        entity_ref: Compass entity reference (e.g. component:default/abevalflow)
        fact_ref: Unique fact identifier (e.g. catalog:default/abevalflow_harbor)

    Returns:
        Dict ready to be JSON-serialized and POSTed to Facts API
    """
    return {
        "facts": [
            {
                "factRef": fact_ref,
                "entityRef": entity_ref,
                "data": {
                    "gate_name": gate_result.gate_name,
                    "passed": gate_result.passed,
                    "score": gate_result.score,
                    "mode": gate_result.mode.value if gate_result.mode else "unknown",
                    "message": gate_result.message or "",
                    "details": gate_result.details or {},
                    "evaluated_at": (
                        gate_result.evaluated_at.isoformat()
                        if gate_result.evaluated_at
                        else datetime.now(timezone.utc).isoformat()
                    ),
                },
            }
        ]
    }


def push_gate_fact(
    gate_result: "GateResult",
    endpoint: str,
    entity_ref: str,
    fact_ref: str,
    timeout_sec: float = 30.0,
) -> bool:
    """Push a gate result to Compass Facts API.

    Args:
        gate_result: The gate evaluation result to push
        endpoint: Compass Facts API URL
        entity_ref: Compass entity reference
        fact_ref: Unique fact identifier for this gate
        timeout_sec: Request timeout in seconds

    Returns:
        True if push succeeded, False otherwise
    """
    payload = _build_fact_payload(gate_result, entity_ref, fact_ref)

    try:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            endpoint,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )

        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            status = response.status
            if status in (200, 201, 202):
                logger.info(
                    "Pushed fact %s for gate %s (status=%d)",
                    fact_ref,
                    gate_result.gate_name,
                    status,
                )
                return True
            else:
                logger.warning(
                    "Unexpected status %d pushing fact %s",
                    status,
                    fact_ref,
                )
                return False

    except urllib.error.HTTPError as e:
        logger.error(
            "HTTP error pushing fact %s: %d %s",
            fact_ref,
            e.code,
            e.reason,
        )
        return False
    except urllib.error.URLError as e:
        logger.error("URL error pushing fact %s: %s", fact_ref, e.reason)
        return False
    except Exception as e:
        logger.error("Error pushing fact %s: %s", fact_ref, e)
        return False


def push_gate_fact_from_config(
    gate_result: "GateResult",
    push_facts_config: "PushFactsConfig",
) -> bool:
    """Push a gate result using PushFactsConfig settings.

    Convenience wrapper that builds the fact_ref from the config prefix
    and gate name.

    Args:
        gate_result: The gate evaluation result to push
        push_facts_config: Configuration with endpoint, entity_ref, and prefix

    Returns:
        True if push succeeded, False otherwise
    """
    fact_ref = f"{push_facts_config.fact_ref_prefix}{gate_result.gate_name}"

    return push_gate_fact(
        gate_result=gate_result,
        endpoint=push_facts_config.endpoint,
        entity_ref=push_facts_config.entity_ref,
        fact_ref=fact_ref,
    )


def validate_push_facts_config(
    push_facts_config: "PushFactsConfig | None",
    gates_with_push_fact: list[str],
) -> None:
    """Validate push_facts configuration and warn if misconfigured.

    Logs a warning if endpoint is configured but no gates have push_fact=True.

    Args:
        push_facts_config: The push_facts configuration (may be None)
        gates_with_push_fact: List of gate names with push_fact=True
    """
    if push_facts_config is None:
        return

    if not gates_with_push_fact:
        logger.warning(
            "push_facts.endpoint is configured (%s) but no gates have push_fact=True. "
            "No facts will be pushed. Add push_fact: true to gate configurations.",
            push_facts_config.endpoint,
        )
