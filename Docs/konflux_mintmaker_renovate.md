# Konflux Mintmaker and Dependency Updates

This document describes how Konflux **Mintmaker** manages dependency updates for this
repository, why `renovate.json` exists, and how the team should review Mintmaker pull
requests.

Mintmaker is Konflux's [Renovate](https://docs.renovatebot.com/)-based automation. It
opens pull requests when it detects outdated dependencies in onboarded repositories.
PRs are authored by `red-hat-konflux-kflux-prd-rh02[bot]`.

## Background

Before `renovate.json` was added, Mintmaker opened many routine upgrade PRs, including:

- Lock-file-only refreshes (`uv.lock` without a corresponding `pyproject.toml` change)
- UBI and Go base image bumps in Containerfiles
- Tekton catalog task version and digest updates in `.tekton/` pipeline definitions
- Routine Python package major, minor, and patch bumps

That volume made it hard to distinguish security work from noise. The repository now
opts into **security/CVE-driven updates only**.

## Policy

| Goal | Mechanism |
|------|-----------|
| Open PRs for known vulnerabilities | `osvVulnerabilityAlerts: true` |
| Skip lock-only maintenance PRs | `lockFileMaintenance.enabled: false` |
| Skip routine version bumps | Disable `major`, `minor`, `patch`, `pin`, and `digest` for all managers |

Configuration lives at the repository root:

```json
{
  "$schema": "https://docs.renovatebot.com/renovate-schema.json",
  "osvVulnerabilityAlerts": true,
  "lockFileMaintenance": { "enabled": false },
  "packageRules": [
    {
      "matchUpdateTypes": ["major", "minor", "patch", "pin", "digest"],
      "enabled": false
    }
  ]
}
```

Mintmaker reads `renovate.json` from the **default branch** (`main`). Changes take
effect after merge; open PRs already created under the old policy are not retroactively
changed.

## What Mintmaker may still update

With the current policy, expect PRs primarily when OSV reports a known CVE affecting:

- Python dependencies tracked in `pyproject.toml` / `uv.lock`
- Base images and packages referenced in Containerfiles under `containers/` and
  `pipeline/images/`
- Dependencies referenced in Konflux pipeline definitions (for example `.tekton/`
  Tekton catalog task references)

Routine "latest version available" bumps should **not** appear unless tied to a
security advisory.

## What is suppressed

The following update categories are disabled globally:

| Update type | Examples in this repo |
|-------------|----------------------|
| `major` / `minor` / `patch` | Python packages in `pyproject.toml` |
| `pin` | Pinned dependency version changes |
| `digest` | Tekton task image digest bumps in `.tekton/*.yaml` |
| Lock file maintenance | Standalone `uv.lock` refresh PRs |

If the team later wants controlled non-security upgrades (for example periodic Tekton
catalog refreshes), add explicit `packageRules` rather than re-enabling all update
types globally. See [Renovate package rules](https://docs.renovatebot.com/configuration-options/#packagerules).

## Review process for Mintmaker PRs

**Do not auto-merge** Mintmaker PRs. Treat each one as a normal change that must pass
CI and Konflux PipelineRuns.

When a **Python security** PR is opened:

1. Confirm **`pyproject.toml` and `uv.lock` are both updated**. If only `uv.lock`
   changed, align `pyproject.toml` manually before merge.
2. Wait for GitHub Actions and the Konflux PipelineRun on the PR to pass.
3. Merge only when green.

For Containerfile or Tekton updates, verify the change addresses a reported CVE and
that downstream builds still succeed.

## Related files

| Path | Role |
|------|------|
| `renovate.json` | Mintmaker / Renovate policy (this document) |
| `pyproject.toml`, `uv.lock` | Python dependencies (uv) |
| `containers/*/Containerfile` | Application and harness container bases |
| `pipeline/images/*/Containerfile` | Pipeline-side images (for example PyRIT) |
| `.tekton/agent-eval-harness-*.yaml` | Konflux build pipelines and Tekton catalog refs |

## Changing this policy

1. Propose changes to `renovate.json` in a normal pull request with rationale.
2. Get reviewer agreement — dependency policy affects CI noise and security posture.
3. Merge to `main`; Mintmaker picks up the new config on subsequent scans.

Introduced in [PR #86](https://github.com/RHEcosystemAppEng/agentic_eval_flow/pull/86).

## See also

- [Konflux Integration Guide](konflux-integration-guide.md) — evaluation tasks and
  IntegrationTestScenario setup
- [Infrastructure & Operations Guide](infrastructure_ops.md) — Quay images, Tekton
  bundles, and cluster operations
