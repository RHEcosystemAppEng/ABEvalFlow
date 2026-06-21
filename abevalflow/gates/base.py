"""Base schemas for gate results.

Gates are evaluation checkpoints that produce standardized results.
Three types exist:
- Engine gates: evaluation engines (Harbor, ASE, A2A, MCPChecker)
- Security gates: security scanners (Cisco, Snyk)
- Quality gates: quality reviewers (LLM review)

Each gate produces a GateResult with a normalized score and pass/fail status.
"""

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class GateType(StrEnum):
    """Type of gate in the unified scorecard."""

    ENGINE = "engine"
    SECURITY = "security"
    QUALITY = "quality"


class GateMode(StrEnum):
    """Gate enforcement mode."""

    DISABLED = "disabled"
    WARN = "warn"
    BLOCK = "block"


class Severity(StrEnum):
    """Finding severity levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class Finding(BaseModel):
    """A single finding from a gate evaluation.

    Findings represent issues, warnings, or informational notes discovered
    during gate evaluation. Used by security and quality gates primarily.
    """

    severity: Severity = Field(description="Severity level of the finding")
    message: str = Field(description="Human-readable description of the finding")
    location: str | None = Field(
        default=None,
        description="File path or location where the finding was detected",
    )
    rule_id: str | None = Field(
        default=None,
        description="Identifier of the rule that triggered this finding",
    )
    details: dict[str, Any] | None = Field(
        default=None,
        description="Additional structured details about the finding",
    )


class GateResult(BaseModel):
    """Standardized result from any gate evaluation.

    All gates (engine, security, quality) produce this common format,
    enabling unified scorecard aggregation and policy enforcement.
    """

    gate_type: GateType = Field(description="Type of gate: engine, security, or quality")
    gate_name: str = Field(
        description="Unique name of the gate (e.g., 'harbor', 'cisco', 'llm-review')"
    )
    passed: bool = Field(description="Whether the gate passed based on its internal criteria")
    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Normalized score from 0.0 (worst) to 1.0 (best)",
    )
    mode: GateMode = Field(
        default=GateMode.WARN,
        description="Enforcement mode that was applied to this gate",
    )
    threshold: float | None = Field(
        default=None,
        description="Threshold that was used to determine pass/fail (if applicable)",
    )
    findings: list[Finding] = Field(
        default_factory=list,
        description="List of findings from the gate evaluation",
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Engine-specific payload (AnalysisResult, MCPCheckerResult, etc.)",
    )
    message: str | None = Field(
        default=None,
        description="Human-readable summary of the gate result",
    )
    evaluated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the gate was evaluated",
    )
