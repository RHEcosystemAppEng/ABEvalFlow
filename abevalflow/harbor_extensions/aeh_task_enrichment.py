"""Post-process AEH-generated Harbor task packages for OpenShift runs.

Upstream ``agent_eval.harbor.tasks.generate_tasks`` (v1.0.3) omits:

1. ``annotations.yaml`` in ``environment/`` and blanks ``dataset.path``, so
   Harbor verifiers cannot load judge annotations (``case_dir`` is the
   workdir basename, typically ``workspace``).
2. Submission ``plugin_dirs`` / ``skills/`` trees, so Claude Code gets
   ``Unknown command: /<skill>`` when the instruction uses a slash-skill.

This module patches generated task packages in-place before ``harbor run``.
It does not modify upstream agent-eval-harness source.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import yaml

_ANNOTATIONS_STAGE = """
# ABEvalFlow: stage annotations for Harbor verifier judges.
# reward.py uses case_dir basename (e.g. "workspace") under dataset.path.
if [ -f "@@WORKDIR@@/annotations.yaml" ]; then
  _aeh_case="$(basename "@@WORKDIR@@")"
  mkdir -p "/tests/cases/${_aeh_case}"
  cp "@@WORKDIR@@/annotations.yaml" "/tests/cases/${_aeh_case}/annotations.yaml"
fi
""".lstrip()


def enrich_harbor_tasks(tasks_dir: Path, *, config_path: Path) -> int:
    """Enrich every Harbor task package under *tasks_dir*.

    Returns the number of task packages updated.
    """
    config_path = Path(config_path)
    tasks_dir = Path(tasks_dir)
    if not tasks_dir.is_dir():
        return 0

    raw = yaml.safe_load(config_path.read_text()) or {}
    submission_root = config_path.parent
    dataset_rel = ""
    dataset = raw.get("dataset") or {}
    if isinstance(dataset, dict):
        dataset_rel = str(dataset.get("path") or "").strip()

    plugin_dirs = []
    runner = raw.get("runner") or {}
    if isinstance(runner, dict):
        plugin_dirs = list(runner.get("plugin_dirs") or [])

    updated = 0
    for task_dir in sorted(p for p in tasks_dir.iterdir() if p.is_dir()):
        if not (task_dir / "task.toml").is_file():
            continue
        _enrich_one_task(
            task_dir,
            case_id=task_dir.name,
            submission_root=submission_root,
            dataset_rel=dataset_rel,
            plugin_dirs=plugin_dirs,
        )
        updated += 1
    return updated


def _enrich_one_task(
    task_dir: Path,
    *,
    case_id: str,
    submission_root: Path,
    dataset_rel: str,
    plugin_dirs: list[str],
) -> None:
    env_dir = task_dir / "environment"
    env_dir.mkdir(parents=True, exist_ok=True)
    tests_dir = task_dir / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)

    # 1) annotations.yaml → environment/ (uploaded to workdir)
    annotations_src = _find_annotations(submission_root, dataset_rel, case_id)
    if annotations_src is not None:
        shutil.copy2(annotations_src, env_dir / "annotations.yaml")

    # 2) Restore dataset.path so load_case_record can resolve annotations
    bundled = tests_dir / "eval.yaml"
    if bundled.is_file():
        cfg = yaml.safe_load(bundled.read_text()) or {}
        if isinstance(cfg.get("dataset"), dict):
            cfg["dataset"]["path"] = "cases"
        elif "dataset" not in cfg:
            cfg["dataset"] = {"path": "cases"}
        bundled.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))

    # 3) Stage annotations in test.sh before reward runs
    test_sh = tests_dir / "test.sh"
    if test_sh.is_file():
        _inject_annotations_stage(test_sh)

    # 4) Copy plugin skill trees + point Harbor at them
    copied_skills = False
    for rel in plugin_dirs:
        src = (submission_root / rel).resolve()
        if not src.is_dir():
            continue
        if not _dir_has_skill_md(src):
            continue
        dest = env_dir / "skills"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
        copied_skills = True
        break  # Harbor expects a single skills_dir root

    if copied_skills:
        _ensure_skills_dir_in_task_toml(task_dir / "task.toml", "/workspace/skills")


def _find_annotations(submission_root: Path, dataset_rel: str, case_id: str) -> Path | None:
    candidates: list[Path] = []
    if dataset_rel:
        candidates.append(submission_root / dataset_rel / case_id / "annotations.yaml")
    candidates.append(submission_root / "cases" / case_id / "annotations.yaml")
    for path in candidates:
        if path.is_file():
            return path
    return None


def _dir_has_skill_md(skills_root: Path) -> bool:
    if (skills_root / "SKILL.md").is_file():
        return True
    for child in skills_root.iterdir():
        if child.is_dir() and (child / "SKILL.md").is_file():
            return True
    return False


def _inject_annotations_stage(test_sh: Path) -> None:
    text = test_sh.read_text()
    if "ABEvalFlow: stage annotations" in text:
        return

    workdir = "/workspace"
    m = re.search(r'--case-dir\s+"([^"]+)"', text)
    if m:
        workdir = m.group(1)
    stage = _ANNOTATIONS_STAGE.replace("@@WORKDIR@@", workdir)

    marker = "mkdir -p /logs/verifier"
    if marker in text:
        text = text.replace(marker, marker + "\n\n" + stage, 1)
    else:
        text = stage + "\n" + text
    test_sh.write_text(text)


def _ensure_skills_dir_in_task_toml(task_toml: Path, skills_dir: str) -> None:
    text = task_toml.read_text()
    if re.search(r"^\s*skills_dir\s*=", text, flags=re.MULTILINE):
        return
    if re.search(r'^\s*workdir\s*=\s*"[^"]*"', text, flags=re.MULTILINE):
        text = re.sub(
            r'^(\s*workdir\s*=\s*"[^"]*")\s*$',
            rf'\1\nskills_dir = "{skills_dir}"',
            text,
            count=1,
            flags=re.MULTILINE,
        )
    else:
        text = text.rstrip() + f'\n\n[environment]\nskills_dir = "{skills_dir}"\n'
    task_toml.write_text(text)
