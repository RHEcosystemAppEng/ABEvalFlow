# Certification, Scorecards, Checks, and Facts

This document describes ABEvalFlow's certification system, which evaluates AI artifacts (skills, MCP servers, agents) against a comprehensive set of checks organized into three certification levels.

## Overview

```
Submission → Gates → Checks → Certification Levels → Facts (Compass)
                         ↓
                    Scorecard
```

**Key Concepts:**

| Concept | Description | Location |
|---------|-------------|----------|
| **Gate** | A category of evaluation (engine, security, quality) | `abevalflow/gates/` |
| **Check** | A specific validation criterion | `abevalflow/certification.py` |
| **Certification Level** | Foundational, Trusted, or Certified | `abevalflow/certification.py` |
| **Scorecard** | Unified result combining all gates and certification | `abevalflow/scorecard.py` |
| **Fact** | JSON payload pushed to Compass | `abevalflow/compass_facts.py` |

## Certification Levels

Levels are **hierarchical**: Certified requires Trusted, Trusted requires Foundational.

```
┌─────────────────────────────────────────────────────────────┐
│                        CERTIFIED                            │
│  Enterprise-grade validation for production deployment      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                      TRUSTED                         │   │
│  │  Production-ready with advanced security & quality   │   │
│  │  ┌─────────────────────────────────────────────┐    │   │
│  │  │                 FOUNDATIONAL                 │    │   │
│  │  │  Basic validation: structure, security,      │    │   │
│  │  │  execution, quality, metadata                │    │   │
│  │  └─────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Artifact Applicability Matrix

Different artifact types benefit from different checks:

| Artifact Type | Focus Area | Key Checks |
|---------------|------------|------------|
| **Skills** (Prompt + Code) | A/B gap testing, execution isolation | Functional Validation, Execution Validation |
| **MCP Servers** (Tool APIs) | API contracts, schema accuracy | Resilience/Chaos Testing, Operational Policy |
| **Plugins** (REST APIs) | OpenAPI validation, auth flows | Security Validation, Metadata Compliance |
| **Agent Evals** (Autonomous) | Reasoning trace, multi-turn | Advanced Agent Validation, Safety Guardrails |

---

## Certification Profiles

Profiles provide artifact-type-specific check defaults. They are defined in `config/certification_profiles.yaml` and selected at **pipeline deployment level**, not per-submission.

### Available Profiles

| Profile | Description | Certified Checks |
|---------|-------------|------------------|
| `skill` | Skills with prompts and code | advanced_agent_validation |
| `agent` | Autonomous agents | advanced_agent_validation, safety_guardrails |
| `mcp_server` | MCP tool servers | resilience_chaos_testing, enterprise_security_review |
| `plugin` | REST API plugins | enterprise_security_review |
| `full` | All checks enabled | comprehensive |

### How Profiles Work

```
Pipeline deployment: --certification-profile=skill
         ↓
    Profile defaults loaded from config/certification_profiles.yaml
         ↓
    Submission's metadata.yaml can override (optional)
         ↓
    Final policy used for certification
```

### Configuration

**Pipeline parameter** (set once per deployment):
```yaml
params:
  - name: certification-profile
    value: "skill"  # or: agent, mcp_server, plugin, full
```

**Profile definition** (`config/certification_profiles.yaml`):
```yaml
profiles:
  skill:
    description: "Skills with prompts and code"
    foundational:
      checks:
        - valid_skill_structure
        - basic_execution_validation
        - metadata_compliance
    trusted:
      checks:
        - functional_validation
    certified:
      checks:
        - advanced_agent_validation
```

**Submission override** (`metadata.yaml` - optional):
```yaml
certification_policy:
  trusted:
    thresholds:
      functional_validation: 0.9  # Override threshold only
```

### Priority Order

1. **Submission's `metadata.yaml`** — if `certification_policy` is present, use it
2. **Pipeline's `--certification-profile`** — if set, load profile from config
3. **Hardcoded defaults** — fallback to `FOUNDATIONAL_CHECKS`, `TRUSTED_CHECKS`, `CERTIFIED_CHECKS` in Python

### Code Location

- Profiles YAML: `config/certification_profiles.yaml`
- Profile loader: `abevalflow/certification.py` → `load_profile()`
- Aggregate script: `scripts/aggregate_scorecard.py` → `--certification-profile`
- Pipeline task: `pipeline/tasks/post/analyze-and-check-degradation.yaml`

---

## Foundational Level (5 checks)

All 5 checks are **fully implemented**.

### 1. Valid Skill Structure

**Check ID:** `valid_skill_structure`

**What it does:** Validates the submission directory has required files and structure.

| Sub-check | Code | What it verifies |
|-----------|------|------------------|
| skills/ directory | `_check_skills_dir()` | At least one SKILL.md exists |
| instruction.md | `_check_instruction_md()` | File exists and is non-empty |
| Python syntax | `_check_py_compiles()` | test_outputs.py and llm_judge.py compile |
| Supportive files | `_check_supportive_size()` | supportive/ folder ≤ 50MB |

**Source:** `scripts/validate.py` → `validation.json`

**Implementation:** `abevalflow/certification.py` lines 303-309

---

### 2. Basic Security Validation

**Check ID:** `basic_security_validation`

**What it does:** Scans submission code for security vulnerabilities using Cisco skill scanner.

| What it checks | How |
|----------------|-----|
| Hardcoded secrets | Pattern matching for API keys, passwords |
| Dangerous imports | Flags `os.system`, `subprocess`, `eval` |
| SQL injection risks | Detects string concatenation in queries |
| Path traversal | Checks for `../` patterns |

**Source:** Cisco security gate → `GateResult`

**Implementation:** `abevalflow/gates/security/cisco.py`

---

### 3. Basic Execution Validation

**Check ID:** `basic_execution_validation`

**What it does:** Runs the skill through an evaluation engine to verify execution.

| Engine | What it tests |
|--------|---------------|
| Harbor | Runs skill in container, executes test_outputs.py, scores with llm_judge.py |
| ASE | Runs skill against evals.json test cases, LLM-as-judge scoring |
| A2A | Sends tasks to A2A agent endpoint, verifies responses |
| MCPChecker | Calls MCP server tools, validates responses |

**Source:** Engine gate (harbor/ase/a2a/mcpchecker) → `GateResult`

**Implementation:** `abevalflow/engines/`

---

### 4. Content Quality Review

**Check ID:** `content_quality_review`

**What it does:** Uses LLM to review quality of skill content.

| What it reviews | Criteria |
|-----------------|----------|
| Instruction clarity | Is the task clear and unambiguous? |
| Test coverage | Do tests cover the instruction requirements? |
| Code quality | Is the code well-structured and maintainable? |

**Source:** LLM quality gate → `GateResult`

**Implementation:** `abevalflow/gates/quality/llm_review.py`

---

### 5. Metadata Compliance

**Check ID:** `metadata_compliance`

**What it does:** Validates metadata.yaml against Pydantic schema.

**Source:** `scripts/validate.py` → `validation.json`

**Implementation:** `abevalflow/schemas.py` (SubmissionMetadata model)

---

## Trusted Level (8 checks)

4 checks implemented, 4 not yet implemented.

### 6. Evaluation Assets ✅

**Check ID:** `evaluation_assets`

**What it does:** Validates that evaluation test files exist.

**Source:** File existence check (`evals/evals.json` or `tests/`)

**Implementation:** `scripts/aggregate_scorecard.py` lines 222-225

---

### 7. Advanced Security Validation ⚠️ Partial

**Check ID:** `advanced_security_validation`

**What we have:** Same as Basic Security with higher threshold (score ≥ 0.9).

**What's missing:**
- Dependency vulnerability scanning (CVEs via Snyk/Trivy)
- Container image scanning
- Runtime security analysis
- Supply chain validation

**Source:** Cisco security gate with threshold check

**Implementation:** `abevalflow/certification.py` lines 237-246

---

### 8. Functional Validation ✅

**Check ID:** `functional_validation`

**What it does:** Verifies skill produces correct outputs via A/B comparison and statistical significance.

**Source:** Engine gate (same as Basic Execution)

**Implementation:** `abevalflow/certification.py` lines 192-199

---

### 9. Instruction Quality ⚠️ Partial

**Check ID:** `instruction_quality`

**What we have:** Overall content review with threshold (score ≥ 0.7).

**What's missing:**
- Specific instruction.md metrics
- Readability scores (Flesch-Kincaid, etc.)
- Ambiguity detection
- Completeness scoring

**Source:** Quality gate with threshold check

**Implementation:** `abevalflow/certification.py` lines 270-278

---

### 10. Registry Governance ❌ Not Implemented

**Check ID:** `registry_governance`

**What it would do:**
- Validate skill registration in Compass registry
- Check ownership and maintainer info
- Verify version uniqueness
- Detect duplicates or conflicts

**Expected data source:** Compass API query

**In enum:** Yes | **In defaults:** No

---

### 11. Operational Policy Compliance ❌ Not Implemented

**Check ID:** `operational_policy_compliance`

**What it would do:**
- Verify resource limits (CPU, memory, timeout)
- Check logging requirements met
- Validate error handling standards
- Ensure graceful degradation

**Expected data source:** Policy rules engine + runtime metrics

**In enum:** Yes | **In defaults:** No

---

### 12. Efficiency & Cost Profiling ❌ Not Implemented

**Check ID:** `efficiency_cost_profiling`

**What it would do:**
- Measure token consumption (prompt + completion)
- Estimate cost per execution
- Flag context-limit risks
- Detect inefficient prompt patterns

**Expected data source:** LLM response metadata (token counts)

**In enum:** Yes | **In defaults:** No

---

### 13. Data Privacy & PII Handling ❌ Not Implemented

**Check ID:** `data_privacy_pii_handling`

**What it would do:**
- Detect PII in prompts/responses (names, emails, SSNs)
- Verify PII is not logged or leaked
- Check data handling policies
- Validate anonymization where required

**Expected data source:** PII scanner (Presidio, Scrubadub)

**In enum:** Yes | **In defaults:** No

---

## Certified Level (7 checks)

3 checks partially implemented, 4 not yet implemented.

### 1. Enterprise Structure Validation ⚠️ Partial

**Check ID:** `enterprise_structure_validation`

**What we have:** Derived from basic structure validation.

**What's missing:**
- Multi-team ownership validation
- Cross-skill dependency mapping
- Enterprise naming conventions
- Compliance tagging (SOC2, FedRAMP labels)

**Source:** Validation result (passthrough)

**Implementation:** `abevalflow/certification.py` lines 332-340

---

### 2. Enterprise Security Review ⚠️ Partial

**Check ID:** `enterprise_security_review`

**What we have:** Zero findings requirement from Cisco gate.

**What's missing:**
- Compliance framework mapping (SOC2, FedRAMP, HIPAA)
- Audit trail requirements
- Data classification validation
- Penetration testing results

**Source:** Cisco gate with zero-findings check

**Implementation:** `abevalflow/certification.py` lines 248-256

---

### 3. Enterprise Behavioral Testing ❌ Not Implemented

**Check ID:** `enterprise_behavioral_testing`

**What it would do:**
- Edge case testing (unusual inputs, boundary values)
- Concurrency testing (parallel execution)
- Failure mode analysis
- Long-running stability tests
- Memory leak detection

**Expected data source:** Chaos testing framework

**In enum:** Yes | **In defaults:** No

---

### 4. Advanced Agent Validation ⚠️ Partial

**Check ID:** `advanced_agent_validation`

**What we have:** High score threshold (≥ 0.8) from engine gate.

**What's missing:**
- Multi-turn conversation testing
- Context retention validation
- Tool use verification (correct tool selection)
- Hallucination detection
- Reasoning trace analysis

**Source:** Engine gate with threshold check

**Implementation:** `abevalflow/certification.py` lines 202-222

---

### 5. Continuous Optimization ❌ Not Implemented

**Check ID:** `continuous_optimization`

**What it would do:**
- Track historical performance across runs
- Detect regressions automatically
- Update baselines when improvements verified
- Generate optimization recommendations

**Expected data source:** Historical metrics database

**In enum:** Yes | **In defaults:** No

---

### 6. Safety, Toxicity & Bias Guardrails ❌ Not Implemented

**Check ID:** `safety_toxicity_bias_guardrails`

**What it would do:**
- Inject adversarial prompts (jailbreak attempts)
- Test for toxic/harmful output generation
- Detect demographic bias in responses
- Verify refusal mechanisms work

**Expected data source:** Safety testing framework (adversarial prompts)

**In enum:** Yes | **In defaults:** No

---

### 7. Resilience & Chaos Testing ❌ Not Implemented

**Check ID:** `resilience_chaos_testing`

**What it would do:**
- Simulate API timeouts
- Inject malformed responses
- Drop connections mid-stream
- Test rate limiting handling
- Verify graceful degradation

**Expected data source:** Chaos proxy / fault injection framework

**In enum:** Yes | **In defaults:** No

---

## Scorecard Structure

The scorecard (`abevalflow/scorecard.py`) combines all results:

```python
class Scorecard(BaseModel):
    submission_name: str
    pipeline_run_id: str
    eval_engine: str
    
    gates: list[GateResult]           # All gate results
    policy: GatePolicy                # Applied policy
    
    recommendation: Recommendation    # pass/warn/fail
    recommendation_reason: str
    
    certification: CertificationResult  # All 3 levels
    highest_certification: CertificationLevel  # Computed
    
    fact_push_results: list[FactPushResult]
    certification_fact_push_results: list[CertificationFactPushResult]
```

## Facts Pushed to Compass

When `push_facts` is configured, these facts are pushed:

| Fact Reference | Content |
|----------------|---------|
| `abevalflow_evaluation` | Engine gate result |
| `abevalflow_security` | Security gate result |
| `abevalflow_quality` | Quality gate result |
| `abevalflow_foundational` | Foundational level with all checks |
| `abevalflow_trusted` | Trusted level with all checks |
| `abevalflow_certified` | Certified level with all checks |
| `abevalflow_certification` | Summary (highest level achieved) |

## Configuration

### Default Checks per Level

```python
FOUNDATIONAL_CHECKS = [
    "valid_skill_structure",
    "basic_security_validation",
    "basic_execution_validation",
    "content_quality_review",
    "metadata_compliance",
]

TRUSTED_CHECKS = [
    "evaluation_assets",
    "advanced_security_validation",
    "functional_validation",
    "instruction_quality",
    # registry_governance - not yet implemented
    # operational_policy_compliance - not yet implemented
    # efficiency_cost_profiling - not yet implemented
    # data_privacy_pii_handling - not yet implemented
]

CERTIFIED_CHECKS = [
    "enterprise_structure_validation",
    "enterprise_security_review",
    "advanced_agent_validation",
    # enterprise_behavioral_testing - not yet implemented
    # continuous_optimization - not yet implemented
    # safety_toxicity_bias_guardrails - not yet implemented
    # resilience_chaos_testing - not yet implemented
]
```

### YAML Override (metadata.yaml)

```yaml
certification_policy:
  foundational:
    checks:
      - valid_skill_structure
      - basic_security_validation
      - metadata_compliance
      # Remove some checks for easier pass
    thresholds:
      basic_execution_validation: 0.5  # Lower threshold
  
  trusted:
    checks:
      - evaluation_assets
      - functional_validation
      # Only require 2 checks
  
  certified:
    checks:
      - advanced_agent_validation
    thresholds:
      advanced_agent_validation: 0.7  # Lower from 0.8
```

## Implementation Roadmap

### Priority 1: Quick Wins
| Check | Effort | Blocker |
|-------|--------|---------|
| Efficiency & Cost Profiling | Medium | Need token counts from LLM responses |
| Data Privacy & PII Handling | Medium | Integrate PII scanner library |

### Priority 2: Requires New Gates
| Check | Effort | Blocker |
|-------|--------|---------|
| Safety & Bias Guardrails | Medium | Need adversarial prompt set |
| Registry Governance | Medium | Compass API integration |

### Priority 3: Requires Infrastructure
| Check | Effort | Blocker |
|-------|--------|---------|
| Resilience & Chaos Testing | High | Fault injection framework |
| Enterprise Behavioral Testing | High | Chaos testing infrastructure |
| Continuous Optimization | High | Historical metrics database |

### Priority 4: Policy Definition
| Check | Effort | Blocker |
|-------|--------|---------|
| Operational Policy Compliance | Medium | Define policy rules |

## Extending the System

### Adding a New Security Scanner

Example: Adding an "Agent Security" scanner (different from Cisco).

**Step 1: Implement the gate** (`abevalflow/gates/security/agent_security.py`)

```python
from abevalflow.gates.base import GateResult, GateType, GateMode

def run_agent_security_scan(submission_dir: Path) -> GateResult:
    # Your scanner logic here
    findings = scan_for_agent_vulnerabilities(submission_dir)
    
    return GateResult(
        gate_name="agent_security",
        gate_type=GateType.SECURITY,
        passed=len(critical_findings) == 0,
        score=calculate_score(findings),
        mode=GateMode.BLOCK,
        findings=findings,
    )
```

**Step 2: Add the check ID** (`abevalflow/certification.py`)

```python
class CheckId(StrEnum):
    # ... existing checks ...
    AGENT_SECURITY_VALIDATION = "agent_security_validation"
```

**Step 3: Map gate to check** (`abevalflow/certification.py` → `_map_gate_to_checks()`)

```python
# In _map_gate_to_checks function
if gate.gate_type == GateType.SECURITY and gate.gate_name == "agent_security":
    checks.append(CheckResult(
        check_id=CheckId.AGENT_SECURITY_VALIDATION,
        name="Agent Security Validation",
        passed=gate.passed,
        score=gate.score,
        message="Agent-specific security scan",
        source_gate=gate.gate_name,
        details={"source_implementation": gate.policy_key or gate.gate_name},
    ))
```

**Step 4: Add to profile** (`config/certification_profiles.yaml`)

```yaml
profiles:
  agent:
    trusted:
      checks:
        - functional_validation
        - agent_security_validation  # ← Add here (agent profile only)
```

### Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                    SEPARATION OF CONCERNS                             │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  gates/security/*.py          HOW to run scans                       │
│       ↓                                                               │
│  certification.py             WHAT checks exist, gate→check mapping  │
│       ↓                                                               │
│  profiles.yaml                WHICH checks apply to which artifacts  │
│       ↓                                                               │
│  compass_facts.py             HOW to push to Compass (auto)          │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

### What Changes for What

| Change | Files to Modify |
|--------|-----------------|
| Add new scanner | `gates/security/new_scanner.py` |
| Add new check type | `certification.py` (CheckId + mapping) |
| Add check to profile | `config/certification_profiles.yaml` |
| Change check threshold | `profiles.yaml` or `metadata.yaml` |
| Change fact payload structure | `compass_facts.py` |

**No changes needed in:**
- `compass_facts.py` — automatically picks up new checks
- `scorecard.py` — automatically includes new checks  
- Pipeline YAML — just select the profile

---

## File Locations

| Component | Path |
|-----------|------|
| Check IDs and levels | `abevalflow/certification.py` |
| Certification computation | `abevalflow/certification.py:compute_certification()` |
| Profile loading | `abevalflow/certification.py:load_profile()` |
| Default thresholds | `abevalflow/certification.py:DEFAULT_THRESHOLDS` |
| Certification profiles | `config/certification_profiles.yaml` |
| Scorecard model | `abevalflow/scorecard.py` |
| Fact payloads | `abevalflow/compass_facts.py` |
| Schema (CertificationPolicy) | `abevalflow/schemas.py` |
| Aggregation script | `scripts/aggregate_scorecard.py` |
| Validation script | `scripts/validate.py` |
| Tests | `tests/test_certification.py` |
