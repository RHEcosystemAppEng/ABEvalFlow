# ABEvalFlow

Automated Tekton-orchestrated pipeline on OpenShift for evaluating AI artifacts:

- **Skills** — Measures skill efficacy by comparing agent performance with and without skills (A/B "gap" testing)
- **MCP Servers** — Validates MCP server implementations via task-based verification
- **Agents** — Evaluates full agent behavior using Harbor (general agents) or A2A protocol (A2A-compliant agents)

Produces statistical reports with pass rates, uplift metrics, significance tests, and a unified scorecard.

## Pipelines

ABEvalFlow provides two pipeline variants:

| Pipeline | Purpose | Key Differences |
|----------|---------|-----------------|
| **CI Pipeline** | Full evaluation for new submissions | Includes security scan, quality review, artifact generation |
| **Monitoring Pipeline** | Regression detection for deployed artifacts | Includes degradation check against historical baseline, Slack alerts |

## How It Works

The pipeline executes in six main stages, with engine-specific steps within each:

### 1. Prepare
- Clone submission repository
- Validate structure and `metadata.yaml` schema
- AI-assisted generation of missing test artifacts (optional):
  - Harbor/A2A: generates `instruction.md` and `test_outputs.py` from `SKILL.md`
  - ASE: generates `evals.json` from `SKILL.md`

### 2. Test (CI Pipeline only)
- **Quality Review** — AI-powered review of skill/test coherence (advisory)
- **Security Scan** — Cisco AI Defense scan for prompt injection, data exfiltration risks

### 3. Evaluate
Four evaluation engines, each suited for different artifact types:

| Engine | Evaluates | Comparison Mode | Container Isolation |
|--------|-----------|-----------------|---------------------|
| **Harbor** | Skills, general agents | A/B (treatment vs control) | Yes |
| **ASE** | Skills only | A/B (treatment vs control) | No |
| **A2A** | A2A-protocol agents | A/B (treatment vs control) | Yes |
| **MCPChecker** | MCP servers | Single-agent task verification | No |

### 4. Analyze
- Compute pass rates, uplift (gap), statistical significance (p-value)
- Generate `report.json` and `report.md`
- Aggregate gate results into unified `scorecard.json`
- **Monitoring only:** Check for degradation against historical baseline

### 5. Store
- Upload reports and artifacts to MinIO
- Record results to PostgreSQL for historical analysis

### 6. Cleanup
- Remove temporary workspaces and artifacts

## Configuration

All flow configuration is defined in `metadata.yaml` within each submission:

```yaml
name: my-submission
eval_engine: harbor              # harbor, ase, a2a, or mcpchecker
persona: general                 # Agent persona for Harbor/A2A

experiment:
  n_trials: 20                   # Number of evaluation attempts

gate_policy:
  default_mode: warn
  combination: all_pass
  gates:
    evaluation:
      mode: block
      threshold: 0.0
    security:
      mode: warn
```

See [Gate Policy Configuration](#gate-policy-configuration) for full options.

## Repository Structure

```
ABEvalFlow/
├── Docs/                    # ADR, implementation plan, guides
├── pipeline/
│   ├── pipeline.yaml        # Main pipeline definition
│   ├── triggers/            # EventListener, TriggerTemplate, TriggerBinding
│   └── tasks/
│       ├── validate.yaml
│       ├── generate_tests.yaml
│       ├── test-quality-review.yaml
│       ├── security-scan.yaml
│       ├── scaffold.yaml
│       ├── build-push.yaml
│       ├── harbor-eval.yaml
│       ├── analyze-report.yaml
│       └── publish-store.yaml
├── templates/               # Jinja2 templates (Dockerfiles, test.sh, task.toml)
├── scripts/                 # Python scripts invoked by pipeline tasks
├── config/                  # K8s manifests (RBAC, PostgreSQL, LiteLLM)
└── tests/                   # Unit and integration tests
```

## Related Repositories

| Repository | Purpose |
|---|---|
| [skill-submissions](https://github.com/RHEcosystemAppEng/skill-submissions) | Submission intake — users push skills, MCP evals, and agent evals here |
| [skills_eval_corrections](https://github.com/RHEcosystemAppEng/skills_eval_corrections) | Harbor fork with OpenShift backend for ABEvalFlow |
| [All-Hands-AI/openhands-agent-monitor](https://github.com/All-Hands-AI/openhands-agent-monitor) | Harbor upstream — agent evaluation framework |
| [cisco-ai-defense/skill-scanner](https://github.com/cisco-ai-defense/skill-scanner) | Security scanner for prompt injection and data exfiltration detection |

## Evaluation Engines

The pipeline supports four evaluation engines, each suited for different artifact types:

| Engine | Artifact Type | Use Case | Comparison | Container Isolation |
|--------|---------------|----------|------------|---------------------|
| **Harbor** | Skills, Agents | Full evaluation with real tool execution | A/B (with vs without skill) | Yes |
| **ASE** | Skills only | Lightweight LLM-as-judge assertions | A/B (with vs without skill) | No |
| **A2A** | A2A Agents | A2A-protocol compliant agent evaluation | A/B (treatment vs control) | Yes |
| **MCPChecker** | MCP Servers | MCP server/tool verification | Single-agent task verification | No |

Engines are implemented in `abevalflow/engines/` using a registry pattern:

```
abevalflow/engines/
├── __init__.py      # Engine registry and factory
├── base.py          # EvalEngine abstract base class
├── harbor.py        # Harbor A/B evaluation
├── ase.py           # ASE LLM-as-judge evaluation
├── a2a.py           # A2A protocol evaluation
└── mcpchecker.py    # MCPChecker task verification
```

## Gates Architecture

Gates are evaluation checkpoints that produce standardized results. The unified scorecard aggregates all gate results to produce a final recommendation.

### Gate Types

| Category | Policy Key | Purpose | Implementation |
|----------|------------|---------|----------------|
| **evaluation** | `evaluation` | Results from the selected eval engine | Harbor, ASE, A2A, or MCPChecker |
| **security** | `security` | Security scanning results | Cisco AI Defense scanner |
| **quality** | `quality` | Quality review results | LLM-powered review |

### Gate Modes

Each gate operates in one of three modes:

| Mode | Behavior |
|------|----------|
| `disabled` | Gate is skipped entirely |
| `warn` | Gate runs; failures produce warnings but don't block |
| `block` | Gate runs; failures cause the scorecard to fail |

### GateResult Schema

All gates produce a standardized `GateResult`:

```python
class GateResult:
    gate_type: GateType      # engine, security, or quality
    gate_name: str           # Category name: "evaluation", "security", or "quality"
    policy_key: str          # Implementation: "harbor", "cisco", "llm-review", etc.
    passed: bool             # Whether the gate passed
    score: float             # Normalized score (0.0 to 1.0)
    mode: GateMode           # Mode that was applied (disabled/warn/block)
    threshold: float | None  # Threshold used for pass/fail
    findings: list[Finding]  # Issues discovered (security/quality gates)
    details: dict            # Implementation-specific data (e.g., {"engine": "harbor"})
    message: str             # Human-readable summary
```

The `gate_name` is the category used in policy configuration, while `policy_key` identifies the specific implementation.

### Existing Gates

#### Evaluation Gate (`evaluation`)

The primary gate that wraps the selected evaluation engine's results.

- **Location:** `abevalflow/engines/*.py` (each engine produces evaluation gate results)
- **Input:** Engine-specific report from `reports/{submission}/`
- **Engines:** Harbor, ASE, A2A, MCPChecker (selected via `eval_engine` in metadata.yaml)
- **Pass criteria:**
  - Harbor/ASE/A2A: `treatment_score - control_score >= threshold` (default threshold: 0.0)
  - MCPChecker: All tasks pass verification
- **Score:** Mean reward or pass rate depending on engine

#### Security Gate (`security`)

Reads `security-scan.json` produced by the Cisco AI Defense scanner.

- **Location:** `abevalflow/gates/security/cisco.py`
- **Input:** `reports/{submission}/security-scan.json`
- **Scanner:** Cisco AI Defense
- **Pass criteria:**
  - `warn` mode: Always passes (findings are advisory)
  - `block` mode: Fails if any HIGH or CRITICAL findings exist
- **Score:** Weighted average based on finding severities

#### Quality Gate (`quality`)

Reads `_ai_review.json` produced by the AI quality reviewer.

- **Location:** `abevalflow/gates/quality/llm_review.py`
- **Input:** `{workspace}/_ai_review.json`
- **Reviewer:** LLM-powered quality review
- **Dimensions evaluated:** coherence, coverage, clarity, feasibility, robustness
- **Pass criteria:**
  - `warn` mode: Passes unless recommendation is "fail"
  - `block` mode: Passes only if `overall_score >= threshold`
- **Default threshold:** 0.6

## Scorecard

The scorecard is the single source of truth for submission evaluation, aggregating all gate results with configurable policy.

### Scorecard Schema

```python
class Scorecard:
    submission_name: str           # Name of the evaluated submission
    pipeline_run_id: str           # Tekton PipelineRun ID
    eval_engine: str               # Primary evaluation engine used
    gates: list[GateResult]        # All gate results
    policy: GatePolicy             # Policy that was applied
    recommendation: Recommendation # pass, warn, or fail
    recommendation_reason: str     # Human-readable explanation
    gates_passed: int              # Count of passed gates
    gates_failed: int              # Count of failed gates
    blocking_gates_passed: int     # Count of passed blocking gates
    blocking_gates_failed: int     # Count of failed blocking gates
```

### Combination Modes

The scorecard supports three modes for combining gate results:

| Mode | Logic |
|------|-------|
| `all_pass` | All blocking gates must pass; failing warn gates produce warnings |
| `any_pass` | At least one blocking gate must pass |
| `weighted` | Weighted average of gate scores determines outcome |

### Output

The scorecard is written to `reports/{submission}/scorecard.json` and includes:
- All gate results with scores and findings
- Final recommendation with reasoning
- Provenance metadata (commit SHA, branch, pipeline run ID)

## Gate Policy Configuration

Gate policies are configured in `metadata.yaml` under the `gate_policy` key:

```yaml
# metadata.yaml
name: my-skill
eval_engine: harbor

gate_policy:
  default_mode: warn           # Default mode for all gates
  combination: all_pass        # How to combine gate results

  gates:
    # Security gate configuration
    security:
      mode: block              # Fail the scorecard on security issues
      threshold: 0.8           # Minimum score to pass

    # Quality gate configuration
    quality:
      mode: warn               # Advisory only
      threshold: 0.6           # Threshold for pass/fail

    # Engine gate configuration (uses eval_engine automatically)
    evaluation:
      mode: block
      threshold: 0.0           # Any positive uplift passes
```

### GatePolicyItem Options

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mode` | `disabled`/`warn`/`block` | `warn` | Enforcement mode |
| `threshold` | `float` | Gate-specific | Score threshold for pass/fail |
| `weight` | `float` | `1.0` | Weight for weighted combination mode |

## Compass Facts Integration

The pipeline can push gate results to Red Hat Compass as Soundcheck facts for visibility in the developer portal.

### Configuration

Enable fact pushing in `metadata.yaml`:

```yaml
gate_policy:
  push_facts:
    endpoint: https://compass.redhat.com/api/soundcheck/facts/
    entity_ref: component:default/my-component
```

### Fact Structure

Each gate result is pushed as a separate fact. The fact reference includes both the category and implementation:

```json
{
  "facts": [
    {
      "factRef": "catalog:default/abevalflow_evaluation_harbor",
      "entityRef": "component:default/my-component",
      "data": {
        "gate_name": "evaluation",
        "passed": true,
        "score": 0.85,
        "mode": "block",
        "message": "Harbor A/B: gap=0.15 >= threshold=0.0 -> PASS",
        "evaluated_at": "2026-06-21T10:35:53Z"
      }
    }
  ]
}
```

### Authentication

The Compass API token is stored in a Kubernetes secret:

```bash
oc create secret generic compass-facts-api --from-literal=token=<your-token>
```

## Persistence

### MinIO (Object Storage)

Reports and artifacts are uploaded to MinIO under a timestamped prefix:

```
s3://ab-eval-reports/YYYYMMDD_hhmmss_{submission}_{run-id}/
├── report.json              # Main evaluation report
├── report.md                # Human-readable report
├── scorecard.json           # Unified scorecard
├── security_scans/          # Security scan results
│   └── security-scan.json
├── generated/               # AI-generated artifacts
│   ├── instruction.md
│   └── test_outputs.py
├── scaffolded/              # Scaffolded configs and review
│   └── _ai_review.json
└── trials/                  # Per-trial artifacts (Harbor)
    ├── trial_001/
    │   ├── agent/
    │   └── verifier/
    └── ...
```

### PostgreSQL (Results Database)

Evaluation results are persisted for historical analysis and monitoring:

- **Script:** `scripts/store_results.py`
- **Data stored:**
  - Submission metadata
  - Per-trial results (Harbor/ASE)
  - Security scan findings
  - Aggregate statistics
  - Scorecard recommendation

## Extensibility

### Adding a New Engine

1. Create a new file in `abevalflow/engines/`:

```python
# abevalflow/engines/my_engine.py
from abevalflow.engines import register_engine
from abevalflow.engines.base import EvalEngine
from abevalflow.gates.base import GateResult, GateType

@register_engine("my-engine")
class MyEngine(EvalEngine):
    name = "my-engine"

    def read_result(self, reports_dir: Path) -> dict | None:
        """Read engine results from reports directory."""
        result_path = reports_dir / "my-engine-report.json"
        if not result_path.exists():
            return None
        return json.loads(result_path.read_text())

    def to_gate_result(self, raw_result: dict, policy: GatePolicy) -> GateResult:
        """Convert engine result to standardized GateResult."""
        score = raw_result.get("score", 0.0)
        threshold = policy.get_gate_policy(self.name).threshold or 0.0

        return GateResult(
            gate_type=GateType.ENGINE,
            gate_name="evaluation",
            policy_key=self.name,
            passed=score >= threshold,
            score=score,
            mode=policy.get_gate_policy(self.name).mode,
            message=f"MyEngine: score={score:.2f}",
        )
```

2. Import in `abevalflow/engines/__init__.py`:

```python
from abevalflow.engines.my_engine import MyEngine
```

### Adding a New Security Gate

1. Create a new file in `abevalflow/gates/security/`:

```python
# abevalflow/gates/security/snyk.py
from abevalflow.gates.security import register_security_gate
from abevalflow.gates.security.base import SecurityGate
from abevalflow.gates.base import GateResult, GateType

@register_security_gate("snyk")
class SnykGate(SecurityGate):
    name = "snyk"

    def evaluate(self, reports_dir: Path, policy: GatePolicy) -> GateResult:
        """Evaluate Snyk security scan results."""
        # Read snyk-report.json and produce GateResult
        ...
```

2. Import in `abevalflow/gates/security/__init__.py`:

```python
from abevalflow.gates.security.snyk import SnykGate
```

### Adding a New Quality Gate

1. Create a new file in `abevalflow/gates/quality/`:

```python
# abevalflow/gates/quality/custom_review.py
from abevalflow.gates.quality import register_quality_gate
from abevalflow.gates.quality.base import QualityGate
from abevalflow.gates.base import GateResult, GateType

@register_quality_gate("custom-review")
class CustomReviewGate(QualityGate):
    name = "custom-review"

    def evaluate(self, workspace_root: Path, policy: GatePolicy) -> GateResult:
        """Evaluate custom quality review results."""
        # Read review artifacts and produce GateResult
        ...
```

2. Import in `abevalflow/gates/quality/__init__.py`:

```python
from abevalflow.gates.quality.custom_review import CustomReviewGate
```

### Adding a New Gate Category

To add an entirely new gate category (e.g., "compliance", "performance"):

1. **Add the GateType enum** in `abevalflow/gates/base.py`:

```python
class GateType(str, Enum):
    ENGINE = "engine"
    SECURITY = "security"
    QUALITY = "quality"
    COMPLIANCE = "compliance"  # New category
```

2. **Create the gate directory** at `abevalflow/gates/compliance/`:

```
abevalflow/gates/compliance/
├── __init__.py      # Registry and exports
├── base.py          # ComplianceGate base class
└── my_checker.py    # First implementation
```

3. **Create the base class** in `abevalflow/gates/compliance/base.py`:

```python
from abc import abstractmethod
from abevalflow.gates.base import GateResult, GateType

class ComplianceGate:
    name: str

    @abstractmethod
    def evaluate(self, reports_dir: Path, policy: GatePolicy) -> GateResult:
        """Evaluate compliance and return standardized GateResult."""
        pass
```

4. **Update the scorecard aggregation** in `scripts/aggregate_scorecard.py`:

```python
from abevalflow.gates.compliance import get_all_compliance_gates

# In aggregate_scorecard():
for compliance_gate in get_all_compliance_gates():
    if not policy.is_enabled(compliance_gate.name):
        continue
    gate_result = compliance_gate.evaluate(reports_dir, policy)
    gates.append(gate_result)
```

5. **Add the category to policy schema** in `abevalflow/schemas.py` (documentation only, the schema is flexible)

## Submission Formats

### Skill Submission (Harbor)

For full agent evaluation with container isolation and A/B comparison:

```
my-skill/
├── instruction.md       # Task description (required, or generated from SKILL.md)
├── skills/
│   └── SKILL.md         # Skill definition (required)
├── tests/
│   ├── test_outputs.py  # Verification tests (required, or generated)
│   └── llm_judge.py     # LLM-based judge (optional)
├── docs/                # Reference documentation (optional)
├── supportive/          # Mock MCPs, data files (optional, <50MB)
└── metadata.yaml        # eval_engine: harbor (required)
```

### Skill Submission (ASE)

For lightweight LLM-as-judge evaluation without containers:

```
my-skill/
├── skills/
│   └── SKILL.md         # Skill definition (required)
├── evals/
│   ├── evals.json       # Evaluation prompts and assertions (optional, generated if missing)
│   └── files/           # Test data files (optional)
└── metadata.yaml        # eval_engine: ase (required)
```

### MCP Server Submission

For validating MCP server implementations:

```
my-mcp-server-eval/
├── metadata.yaml        # eval_engine: mcpchecker (required)
├── mcp-config.yaml      # MCP server connection settings (required)
│                        #   - url: MCP server endpoint
│                        #   - auth: authentication config (if needed)
└── tasks/
    ├── task-1.yaml      # Task definition with expected tool calls
    └── task-2.yaml      # Each task tests specific MCP functionality
```

MCPChecker validates that the MCP server correctly handles tool invocations and returns expected results.

### Agent Submission (A2A Protocol)

For evaluating agents that implement the A2A (Agent-to-Agent) protocol:

```
my-a2a-agent-eval/
├── metadata.yaml        # eval_engine: a2a (required)
├── agent-config.yaml    # Agent endpoint and auth config (required)
│                        #   - endpoint: http://agent-service:8000
│                        #   - auth: bearer token or API key
└── tasks/
    ├── instruction.md   # Task description
    ├── tests/
    │   └── test_outputs.py
    └── task.toml        # Task configuration
```

A2A evaluation connects to a deployed agent via the A2A protocol and runs evaluation tasks against it.

### Agent Submission (Harbor)

For evaluating general agents (non-A2A) with full container isolation:

```
my-agent-eval/
├── metadata.yaml        # eval_engine: harbor, persona: agent (required)
├── instruction.md       # Task description (required)
├── tests/
│   ├── test_outputs.py  # Verification tests (required)
│   └── llm_judge.py     # LLM-based judge (optional)
└── supportive/          # Environment files, data (optional)
```

Harbor creates treatment/control container variants and runs A/B comparison.

See [Manual Trigger Guide](Docs/manual_trigger_guide.md) for detailed submission and trigger instructions.

## LLM Access

The pipeline is LLM-agnostic. Three modes are supported:

| Mode | Proxy Required? |
|---|---|
| Direct API key (Anthropic, OpenAI, etc.) | No |
| opencode + self-hosted model (vLLM, Ollama) | No |
| Google Vertex AI + LiteLLM proxy | Yes |

## Prerequisites

- OpenShift cluster with Pipelines operator (Tekton)
- Container registry (Quay.io) with push credentials
- Harbor fork with OpenShift backend
- LLM access (one of the three modes above)
- Python 3.11+

## Documentation

- [Trigger Guide](Docs/trigger_guide.md) — How to submit skills, configure gate policies, and interpret scorecard results
- [ADR: Skill Evaluation Pipeline](Docs/ADR_Skill_Evaluation_Pipeline_and_Harbor_Execution_Strategy.txt)

## License

Apache License 2.0
