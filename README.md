# ABEvalFlow

Automated Tekton-orchestrated pipeline on OpenShift for evaluating AI skill submissions. Measures skill efficacy by comparing agent performance with and without skills (the "gap"), producing statistical reports with pass rates, uplift metrics, and significance tests.

## How It Works

1. **Submit** — Push a skill directory to the submissions repo; a Tekton EventListener triggers the pipeline.
2. **Validate** — Checks structure, compiles test files, validates `metadata.yaml` schema.
3. **Quality Review** — AI-powered review of skill/test coherence (advisory, non-blocking).
4. **Security Scan** — Optional Cisco AI Defense scan for prompt injection, data exfiltration risks.
5. **Scaffold** — Generates two container variants via Jinja2 templates and an experiment strategy:
   - **Treatment** — includes the experimental material (e.g., skills and reference docs).
   - **Control** — baseline without the experimental material.
6. **Build & Push** — Builds both images and pushes to the OpenShift internal registry.
7. **Evaluate** — Two evaluation engines supported:
   - **Harbor** — Full agent evaluation with container isolation (N=20 attempts per variant).
   - **ASE** — Lightweight LLM-as-judge evaluation using `evals.json` assertions.
8. **Analyze** — Computes pass rates, uplift (gap), statistical significance (p-value).
9. **Publish** — Stores reports to MinIO, records results to PostgreSQL, promotes passing images.

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
| [skill-submissions](https://github.com/RHEcosystemAppEng/skill-submissions) | Submission intake — users push skills here to trigger evaluation |
| [skills_eval_corrections](https://github.com/RHEcosystemAppEng/skills_eval_corrections) | Harbor fork with OpenShift backend |

## Submission Formats

### Harbor Format (full agent evaluation)

```
my-skill-name/
├── instruction.md       # Task description (required)
├── skills/
│   └── SKILL.md         # Skill definition (required)
├── tests/
│   ├── test_outputs.py  # Verification tests (required)
│   └── llm_judge.py     # LLM-based judge (optional)
├── docs/                # Reference documentation (optional)
├── supportive/          # Mock MCPs, data files (optional, <50MB)
└── metadata.yaml        # Name, persona, etc. (required)
```

### ASE Format (lightweight LLM-as-judge)

```
my-skill-name/
├── skills/
│   └── SKILL.md         # Skill definition (required)
├── evals/
│   ├── evals.json       # Evaluation prompts and assertions (optional, generated if missing)
│   └── files/           # Test data files (optional)
└── metadata.yaml        # Name, etc. (required)
```

Trigger with `eval-engine=ase` parameter. See [Trigger Guide](Docs/trigger_guide.md) for details.

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

- [Trigger Guide](Docs/trigger_guide.md) — How to submit skills for evaluation
- [Implementation Plan](Docs/implementation_plan.md) — Phased development plan
- [ADR: Skill Evaluation Pipeline](Docs/ADR_Skill_Evaluation_Pipeline_and_Harbor_Execution_Strategy.txt)

## License

Apache License 2.0
