# ABEvalFlow

Automated Tekton-orchestrated pipeline on OpenShift for evaluating AI skill submissions. Measures skill efficacy by comparing agent performance with and without skills (the "gap"), producing statistical reports with pass rates, uplift metrics, and significance tests.

## How It Works

1. **Submit** — Push a skill directory to the submissions repo; a Tekton EventListener triggers the pipeline.
2. **Validate** — Checks structure, compiles test files, validates `metadata.yaml` schema.
3. **Scaffold** — Generates two container variants via Jinja2 templates and an experiment strategy:
   - **Treatment** — includes the experimental material (e.g., skills and reference docs for a skill experiment).
   - **Control** — baseline without the experimental material.
4. **Build & Push** — Builds both images and pushes to the OpenShift internal registry.
5. **Evaluate** — Harbor runs N attempts per variant (default N=20, 40 total) using a custom OpenShift backend.
6. **Analyze** — Computes pass rates, uplift (gap), statistical significance (p-value), and generates heatmaps.
7. **Publish** — Stores reports, promotes passing images to Quay.io, and records results.

## Repository Structure

```
ABEvalFlow/
├── Docs/                    # ADR, implementation plan
├── pipeline/                # Tekton pipeline and task definitions
│   ├── pipeline.yaml
│   ├── triggers/            # EventListener, TriggerTemplate, TriggerBinding, Interceptor
│   └── tasks/               # validate, scaffold, build-push, harbor-eval, analyze-report, publish-store
├── templates/               # Jinja2 templates (Dockerfiles, test.sh, task.toml)
├── scripts/                 # Python scripts invoked by pipeline tasks
├── config/                  # K8s manifests (RBAC, LiteLLM, pipeline config)
└── tests/                   # Unit and integration tests
```

## Related Repositories

| Repository | Purpose |
|---|---|
| [skill-submissions](https://github.com/RHEcosystemAppEng/skill-submissions) | Submission intake — users push skills here to trigger evaluation |
| [skills_eval_corrections](https://github.com/RHEcosystemAppEng/skills_eval_corrections) | Harbor fork with OpenShift backend |

## Submission Contract

A skill submission directory follows this structure:

```
my-skill-name/
├── instruction.md       # Task description (required)
├── skills/              # Must contain SKILL.md (required, canonical name)
├── docs/                # Reference documentation (optional)
├── tests/
│   ├── test_outputs.py  # Verification tests (required)
│   └── llm_judge.py     # LLM-based judge (optional)
├── supportive/          # Mock MCPs, data files (optional, <50MB)
└── metadata.yaml        # Name, persona, generation_mode, etc. (required)
```

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

- [ADR: Skill Evaluation Pipeline and Harbor Execution Strategy](Docs/ADR_Skill_Evaluation_Pipeline_and_Harbor_Execution_Strategy.txt)
- [Implementation Plan](Docs/implementation_plan.md)

## License

Apache License 2.0
