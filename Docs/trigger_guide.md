# Trigger Guide — SkillsEvalFlow Pipeline

## How the Pipeline is Triggered

The pipeline is triggered automatically when a skill submission is pushed to a
repository configured with a GitHub webhook pointing at the EventListener.

```
git push (submissions/my-skill/)
    → GitHub webhook (POST)
    → EventListener (skills-submission-listener)
    → CEL interceptor filters for submissions/ path changes
    → TriggerBinding extracts repo URL, revision, skill directory
    → TriggerTemplate creates a PipelineRun
    → Pipeline executes: validate → scaffold → build → evaluate → report
```

## Submission Contract

A valid skill submission must follow this structure:

```
submissions/<skill-name>/
├── metadata.yaml              # Required — name is the only mandatory field
├── instruction.md             # Required (manual mode) — task description
├── skills/
│   └── SKILL.md               # Required — canonical skill file
├── tests/
│   ├── test_outputs.py        # Required (manual mode) — pytest verification
│   └── llm_judge.py           # Optional — LLM-based evaluation
├── docs/                      # Optional — reference documentation
├── scripts/                   # Optional — helper scripts
└── supportive/                # Optional — mock MCPs, data files (<50MB)
```

See `examples/sample_skill/` for a minimal working example.

## Webhook Configuration

Configure a GitHub webhook on the submissions repository:

| Setting      | Value                                                    |
|--------------|----------------------------------------------------------|
| Payload URL  | `https://<eventlistener-route>/`                         |
| Content type | `application/json`                                       |
| Events       | **Just the push event**                                  |
| Secret       | Shared secret (configure in EventListener if needed)     |

The EventListener route is created automatically when the EventListener is
deployed. To find it:

```bash
oc get route -n skills-eval-flow -l eventlistener=skills-submission-listener
```

## Manual Trigger (for Testing)

You can bypass the webhook and trigger the pipeline directly.

### Option 1: `tkn` CLI

```bash
tkn pipeline start skills-eval-pipeline \
  -p repo-url=https://github.com/RHEcosystemAppEng/agentic-collections.git \
  -p revision=main \
  -p skill-dir=my-skill \
  -w name=shared-workspace,volumeClaimTemplateFile=pipeline/triggers/pvc-template.yaml \
  -n skills-eval-flow
```

### Option 2: PipelineRun YAML

```yaml
apiVersion: tekton.dev/v1
kind: PipelineRun
metadata:
  generateName: skills-eval-manual-
  namespace: skills-eval-flow
spec:
  pipelineRef:
    name: skills-eval-pipeline
  params:
    - name: repo-url
      value: https://github.com/RHEcosystemAppEng/agentic-collections.git
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

Apply with:

```bash
oc create -f pipelinerun.yaml -n skills-eval-flow
```

## How the CEL Interceptor Works

The EventListener uses an inline CEL interceptor to:

1. **Filter** — only fires when at least one commit touches a file under
   `submissions/`:

   ```cel
   body.commits.exists(c,
     c.added.exists(f, f.startsWith('submissions/')) ||
     c.modified.exists(f, f.startsWith('submissions/'))
   )
   ```

2. **Extract** — pulls the skill directory name from the first matching file
   path (e.g., `submissions/my-skill/SKILL.md` → `my-skill`):

   ```cel
   body.commits.map(c, c.added + c.modified)
     .flatten()
     .filter(f, f.startsWith('submissions/'))
     [0].split('/')[1]
   ```

> **Note:** Single-skill-per-push is assumed. If multiple skills are pushed in
> one commit, only the first one detected is evaluated.

## Tekton Components

| Component | File | Purpose |
|-----------|------|---------|
| EventListener | `pipeline/triggers/event-listener.yaml` | Receives webhooks, filters, extracts skill dir |
| TriggerBinding | `pipeline/triggers/trigger-binding.yaml` | Maps webhook payload to pipeline params |
| TriggerTemplate | `pipeline/triggers/trigger-template.yaml` | Creates PipelineRun from params |
| Validate Task | `pipeline/tasks/validate.yaml` | Validates submission structure and schema |
