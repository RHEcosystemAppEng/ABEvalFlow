# ABEvalFlow

Automated Tekton-orchestrated pipeline on OpenShift for evaluating AI skill submissions. Measures skill efficacy by comparing agent performance with and without skills (the "gap"), producing statistical reports with pass rates, uplift metrics, and significance tests.

## How It Works

1. **Submit** — Push a skill directory to the submissions repo; a Tekton EventListener triggers the pipeline.
2. **Validate** — Checks structure, compiles test files, validates `metadata.yaml` schema.
3. **Generate** — AI-assisted generation of missing test artifacts (optional):
   - Harbor: generates `instruction.md` and `test_outputs.py` from `SKILL.md`
   - ASE: generates `evals.json` from `SKILL.md` if not provided
4. **Quality Review** — AI-powered review of skill/test coherence (advisory, non-blocking).
5. **Security Scan** — Optional Cisco AI Defense scan for prompt injection, data exfiltration risks.
6. **Evaluate** — Two evaluation engines supported:
   - **Harbor** — Full agent evaluation with container isolation:
     - Scaffold treatment/control container variants
     - Build & push images to OpenShift internal registry
     - Run N=20 attempts per variant
   - **ASE** — Lightweight LLM-as-judge evaluation using `evals.json` assertions (no containers).
7. **Analyze** — Computes pass rates, uplift (gap), statistical significance (p-value).
8. **Publish** — Stores reports to MinIO, records results to PostgreSQL.

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
| [cisco-ai-defense/skill-scanner](https://github.com/cisco-ai-defense/skill-scanner) | Security scanner for prompt injection and data exfiltration detection |

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
- [ADR: Skill Evaluation Pipeline](Docs/ADR_Skill_Evaluation_Pipeline_and_Harbor_Execution_Strategy.txt)

## License

Apache License 2.0
