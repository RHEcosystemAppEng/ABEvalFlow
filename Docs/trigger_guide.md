# How to Submit a Skill for A/B Evaluation

This guide explains how to submit a skill to the ABEvalFlow pipeline for
automated evaluation. By the end, you'll know what files to prepare, how to
submit them, and what happens next.

---

## What is this pipeline?

ABEvalFlow automatically tests whether an AI agent performs better **with**
your skill than **without** it. It does this by running the same task many
times in two configurations:

```
                  ┌──────────────────────┐
                  │   You push a skill   │
                  │   submission folder  │
                  └──────────┬───────────┘
                             │
                  ┌──────────▼───────────┐
                  │  Pipeline validates  │
                  │  your files          │
                  └──────────┬───────────┘
                             │
               ┌─────────────┴─────────────┐
               │                           │
    ┌──────────▼──────────┐     ┌──────────▼──────────┐
    │   Treatment (WITH   │     │   Control (WITHOUT   │
    │   your skill)       │     │   your skill)        │
    │   × 20 trials       │     │   × 20 trials        │
    └──────────┬──────────┘     └──────────┬──────────┘
               │                           │
               └─────────────┬─────────────┘
                             │
                  ┌──────────▼───────────┐
                  │  Compare results:    │
                  │  pass rates, uplift, │
                  │  statistical tests   │
                  └──────────┬───────────┘
                             │
                  ┌──────────▼───────────┐
                  │  Report: PASS / FAIL │
                  │  (stored for history)│
                  └──────────────────────┘
```

**Treatment** = the agent has your skill loaded.
**Control** = the agent runs without it (baseline).

If the treatment performs significantly better, the skill **passes**.

---

## What you need to prepare

There are two submission modes:

### Mode 1: Manual (you provide everything)

You write the skill, a task description, and verification tests yourself.
This is the default and currently supported mode.

### Mode 2: AI-assisted (you provide only the skill)

You write just the skill file. The pipeline generates the task description
and tests automatically using an AI assistant. To use this mode, set
`generation_mode: ai` in your `metadata.yaml`.

> **Note:** AI-assisted mode is planned but not yet implemented. For now,
> use manual mode.

---

## Step 1: Create your submission folder

Create a folder with your skill name. The folder must contain these files:

```
my-skill/
├── metadata.yaml          ← describes your submission (required)
├── instruction.md         ← the task the agent must solve (required in manual mode)
├── skills/
│   └── SKILL.md           ← your skill file (required)
├── tests/
│   └── test_outputs.py    ← pytest tests that verify the solution (required in manual mode)
└── docs/                  ← reference docs for the agent (optional)
```

### metadata.yaml (required)

At minimum, you only need a name:

```yaml
name: my-skill
```

A more complete example:

```yaml
name: my-skill
description: Teaches the agent to generate Kubernetes manifests
persona: rh-developer
version: "0.1.0"
author: Jane Doe
tags:
  - kubernetes
  - openshift
```

For AI-assisted mode (not yet available):

```yaml
name: my-skill
generation_mode: ai
```

**Name rules:** lowercase letters, numbers, hyphens, dots, and underscores
only. Must start with a letter or number. Examples: `my-skill`,
`k8s-manifest-gen`, `ocp.admin.tool`.

### instruction.md (required in manual mode)

A clear description of the task the agent must complete. Write it as if
you're explaining the task to a developer. Example:

```markdown
# Create a Greeting Module

Create a `greeting.py` module with a `greet(name: str) -> str` function
that returns a personalized greeting.

## Requirements

- Accept a single `name` argument
- Return format: "Hello, {name}! Welcome aboard."
- Handle empty string by returning "Hello, stranger! Welcome aboard."
```

### skills/SKILL.md (required)

The skill file that will be loaded into the agent during the **treatment**
runs. This is what you're evaluating — the guidance that should make the
agent perform better. Example:

```markdown
# Greeting Module Skill

When asked to create a greeting module:

- Use a single function `greet(name: str) -> str`
- Default to "stranger" when the name is empty
- Keep the output friendly and professional
- Use f-strings for formatting
```

### tests/test_outputs.py (required in manual mode)

Standard pytest tests that verify the agent's output. These run
automatically after each trial. Example:

```python
import importlib
import sys
from pathlib import Path


def _load_module():
    sys.path.insert(0, str(Path("/workspace")))
    return importlib.import_module("greeting")


def test_greet_with_name():
    mod = _load_module()
    assert mod.greet("Alice") == "Hello, Alice! Welcome aboard."


def test_greet_empty_string():
    mod = _load_module()
    assert mod.greet("") == "Hello, stranger! Welcome aboard."
```

### docs/ (optional)

Reference documentation copied into both treatment and control containers
for the agent to consult during trials. Place any relevant `.md`, `.txt`,
or `.pdf` files here.

### supportive/ (optional)

Mock MCP servers, sample data files, or other supporting resources.
Recommended to keep under 50 MB total.

---

## Step 2: Submit your skill

Push your folder to the submissions repository under the `submissions/`
directory:

```bash
# Clone the submissions repo (first time only)
git clone https://github.com/RHEcosystemAppEng/skill-submissions.git
cd skill-submissions

# Add your skill folder
cp -r ~/my-skill submissions/my-skill

# Push
git add submissions/my-skill/
git commit -m "Submit my-skill for evaluation"
git push
```

That's it. The push triggers the pipeline automatically.

---

## Step 3: Wait for results

After you push, the pipeline runs automatically:

1. **Validates** your files (structure, naming, tests compile)
2. **Builds** two container images (one with your skill, one without)
3. **Runs** 20 trials per variant (40 total) against an LLM agent
4. **Analyzes** pass rates, computes uplift and statistical significance
5. **Stores** results in the database for historical tracking
6. **Reports** a PASS or FAIL recommendation

Typical runtime: **10-30 minutes** depending on task complexity.

### Where to find results

- **Pipeline status:** visible in the OpenShift console under
  Pipelines > PipelineRuns in the `ab-eval-flow` namespace
- **Report:** a detailed Markdown/JSON report with pass rates, uplift,
  p-values, and a PASS/FAIL recommendation
- **Historical results:** queryable via `scripts/query_results.py`

---

## Manual trigger (for testing / operators)

You can bypass the webhook and trigger the pipeline directly:

### Option 1: `tkn` CLI

```bash
tkn pipeline start skills-eval-pipeline \
  -p repo-url=https://github.com/RHEcosystemAppEng/skill-submissions.git \
  -p revision=main \
  -p skill-dir=my-skill \
  -w name=shared-workspace,volumeClaimTemplateFile=pipeline/triggers/pvc-template.yaml \
  -n ab-eval-flow
```

### Option 2: PipelineRun YAML

```yaml
apiVersion: tekton.dev/v1
kind: PipelineRun
metadata:
  generateName: skills-eval-manual-
  namespace: ab-eval-flow
spec:
  pipelineRef:
    name: skills-eval-pipeline
  params:
    - name: repo-url
      value: https://github.com/RHEcosystemAppEng/skill-submissions.git
    - name: revision
      value: main
    - name: skill-dir
      value: my-skill
  workspaces:
    - name: shared-workspace
      volumeClaimTemplate:
        spec:
          accessModes:
            - ReadWriteOnce
          resources:
            requests:
              storage: 1Gi
```

### Webhook configuration

Configure a GitHub webhook on the submissions repository:

| Setting      | Value                                                    |
|--------------|----------------------------------------------------------|
| Payload URL  | `https://<eventlistener-route>/`                         |
| Content type | `application/json`                                       |
| Events       | **Just the push event**                                  |
| Secret       | Shared secret (configure in EventListener if needed)     |

To find the EventListener route:

```bash
oc get route -n ab-eval-flow -l eventlistener=submission-listener
```

---

## Quick checklist

Before submitting, verify:

- [ ] `metadata.yaml` exists and has a valid `name`
- [ ] `skills/SKILL.md` exists and is non-empty
- [ ] `instruction.md` exists and clearly describes the task
- [ ] `tests/test_outputs.py` exists and runs with `pytest` locally
- [ ] No secrets, passwords, or API keys in any file
- [ ] `supportive/` folder (if present) is under 50 MB (recommended)
- [ ] Folder name matches the `name` in `metadata.yaml`

---

## Example: complete sample submission

A working example is available in the repository:

```
examples/sample_skill/
├── metadata.yaml
├── instruction.md
├── skills/
│   └── SKILL.md
└── tests/
    └── test_outputs.py
```

You can copy this as a starting point:

```bash
cp -r examples/sample_skill submissions/my-new-skill
# Edit the files for your use case
```

---

## Frequently asked questions

**Q: What happens if my submission fails validation?**
The pipeline stops immediately and reports which checks failed (e.g.,
missing files, invalid metadata, tests that don't compile). Fix the
issues and push again.

**Q: Can I re-run an evaluation?**
Yes. Make any change to your submission folder and push again. Each push
triggers a new evaluation run.

**Q: How many trials are run?**
20 per variant by default (40 total). You can change this in
`metadata.yaml`:

```yaml
experiment:
  n_trials: 10
```

**Q: What counts as a "pass"?**
Each trial runs your tests against the agent's output. If the tests pass,
the trial passes. The overall evaluation compares treatment vs. control
pass rates and uses statistical tests (Fisher's exact test, t-test) to
determine if the improvement is significant.

**Q: Can I evaluate something other than a skill?**
Yes. The pipeline supports different experiment types (model comparison,
prompt comparison, custom). Set `experiment.type` in `metadata.yaml`.
See `Docs/trigger_models_and_experiment_types.md` for details.

**Q: Who do I contact for help?**
Reach out to the ABEvalFlow team or open an issue in the
[ABEvalFlow repository](https://github.com/RHEcosystemAppEng/ABEvalFlow).

---

## Appendix: Operator Reference

This section is for platform operators who deploy and maintain the pipeline
infrastructure. Submitters can skip this.

### Webhook Configuration

Configure a GitHub webhook on the submissions repository:

| Setting      | Value                                                |
|--------------|------------------------------------------------------|
| Payload URL  | `https://<eventlistener-route>/`                     |
| Content type | `application/json`                                   |
| Events       | **Just the push event**                              |
| Secret       | Shared secret (configure in EventListener if needed) |

To find the EventListener route:

```bash
oc get route -n ab-eval-flow -l eventlistener=skills-submission-listener
```

### Manual Trigger (for Testing)

#### Option 1: `tkn` CLI

```bash
tkn pipeline start abevalflow-pipeline \
  -p repo-url=https://github.com/RHEcosystemAppEng/agentic-collections.git \
  -p revision=main \
  -p submission-dir=my-skill \
  -w name=shared-workspace,volumeClaimTemplateFile=pipeline/triggers/pvc-template.yaml \
  -n ab-eval-flow
```

#### Option 2: PipelineRun YAML

```yaml
apiVersion: tekton.dev/v1
kind: PipelineRun
metadata:
  generateName: abevalflow-manual-
  namespace: ab-eval-flow
spec:
  pipelineRef:
    name: abevalflow-pipeline
  params:
    - name: repo-url
      value: https://github.com/RHEcosystemAppEng/agentic-collections.git
    - name: revision
      value: main
    - name: submission-dir
      value: my-skill
  workspaces:
    - name: shared-workspace
      volumeClaimTemplate:
        spec:
          accessModes:
            - ReadWriteOnce
          resources:
            requests:
              storage: 1Gi
```

Apply with:

```bash
oc create -f pipelinerun.yaml -n ab-eval-flow
```

### Tekton Components

| Component | File | Purpose |
|-----------|------|---------|
| Pipeline | `pipeline/pipeline.yaml` | End-to-end pipeline wiring all tasks |
| EventListener | `pipeline/triggers/event-listener.yaml` | Receives webhooks, filters, extracts submission dir |
| TriggerBinding | `pipeline/triggers/trigger-binding.yaml` | Maps webhook payload to pipeline params |
| TriggerTemplate | `pipeline/triggers/trigger-template.yaml` | Creates PipelineRun from params |
