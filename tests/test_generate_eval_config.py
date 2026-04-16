"""Tests for scripts/generate_eval_config.py — per-variant Harbor eval config generation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.generate_eval_config import (
    build_variant_config,
    generate_eval_configs,
    load_metadata,
    main,
)


@pytest.fixture()
def minimal_submission(tmp_path: Path) -> Path:
    """Submission with only a name — all defaults."""
    sub = tmp_path / "my-submission"
    sub.mkdir()
    (sub / "metadata.yaml").write_text(yaml.dump({"name": "my-submission"}))
    return sub


@pytest.fixture()
def custom_submission(tmp_path: Path) -> Path:
    """Submission with custom experiment and resource config."""
    sub = tmp_path / "custom-eval"
    sub.mkdir()
    meta = {
        "name": "custom-eval",
        "description": "A custom evaluation",
        "experiment": {"n_trials": 10, "type": "model"},
        "agent_timeout_sec": 1200.0,
        "verifier_timeout_sec": 240.0,
        "agent_setup_timeout_sec": 300.0,
        "build_timeout_sec": 900.0,
        "cpus": 2,
        "memory_mb": 4096,
        "storage_mb": 20480,
    }
    (sub / "metadata.yaml").write_text(yaml.dump(meta))
    return sub


TREATMENT_DIR = "/workspace/tasks-treatment/my-submission"
CONTROL_DIR = "/workspace/tasks-control/my-submission"
TREATMENT_REF = "registry.example.com/ns/my-submission@sha256:aaa111"
CONTROL_REF = "registry.example.com/ns/my-submission@sha256:bbb222"


class TestLoadMetadata:
    def test_loads_minimal(self, minimal_submission: Path):
        meta = load_metadata(minimal_submission)
        assert meta.name == "my-submission"
        assert meta.experiment.n_trials == 20

    def test_loads_custom(self, custom_submission: Path):
        meta = load_metadata(custom_submission)
        assert meta.name == "custom-eval"
        assert meta.experiment.n_trials == 10
        assert meta.cpus == 2

    def test_missing_metadata_raises(self, tmp_path: Path):
        sub = tmp_path / "empty"
        sub.mkdir()
        with pytest.raises(FileNotFoundError):
            load_metadata(sub)


class TestBuildVariantConfigPrebuilt:
    def test_basic_structure(self, minimal_submission: Path):
        meta = load_metadata(minimal_submission)
        config = build_variant_config(
            meta, "treatment", TREATMENT_DIR, "prebuilt",
            jobs_dir="results/treatment", image_ref=TREATMENT_REF,
        )
        assert config["job_name"] == "my-submission-treatment"
        assert config["n_attempts"] == 20
        assert config["environment"]["type"] == "openshift"
        assert config["environment"]["delete"] is True
        assert len(config["tasks"]) == 1
        assert config["tasks"][0]["path"] == TREATMENT_DIR

    def test_image_ref_in_env_kwargs(self, minimal_submission: Path):
        meta = load_metadata(minimal_submission)
        config = build_variant_config(
            meta, "treatment", TREATMENT_DIR, "prebuilt",
            jobs_dir="results/treatment", image_ref=TREATMENT_REF,
        )
        assert config["environment"]["kwargs"]["image_ref"] == TREATMENT_REF

    def test_control_variant_naming(self, minimal_submission: Path):
        meta = load_metadata(minimal_submission)
        config = build_variant_config(
            meta, "control", CONTROL_DIR, "prebuilt",
            jobs_dir="results/control", image_ref=CONTROL_REF,
        )
        assert config["job_name"] == "my-submission-control"
        assert config["tasks"][0]["path"] == CONTROL_DIR

    def test_no_force_build(self, minimal_submission: Path):
        meta = load_metadata(minimal_submission)
        config = build_variant_config(
            meta, "treatment", TREATMENT_DIR, "prebuilt",
            jobs_dir="results/treatment", image_ref=TREATMENT_REF,
        )
        assert "force_build" not in config["environment"]

    def test_missing_image_ref_raises(self, minimal_submission: Path):
        meta = load_metadata(minimal_submission)
        with pytest.raises(ValueError, match="image_ref is required"):
            build_variant_config(
                meta, "treatment", TREATMENT_DIR, "prebuilt",
                jobs_dir="results/treatment", image_ref="",
            )


class TestBuildVariantConfigLocalBuild:
    def test_no_env_kwargs(self, minimal_submission: Path):
        meta = load_metadata(minimal_submission)
        config = build_variant_config(
            meta, "treatment", TREATMENT_DIR, "local-build",
            jobs_dir="results/treatment",
        )
        assert "kwargs" not in config["environment"]

    def test_force_build_enabled(self, minimal_submission: Path):
        meta = load_metadata(minimal_submission)
        config = build_variant_config(
            meta, "treatment", TREATMENT_DIR, "local-build",
            jobs_dir="results/treatment",
        )
        assert config["environment"]["force_build"] is True


class TestCustomMetadataFields:
    def test_n_trials_from_metadata(self, custom_submission: Path):
        meta = load_metadata(custom_submission)
        config = build_variant_config(
            meta, "treatment", TREATMENT_DIR, "local-build",
            jobs_dir="results/treatment",
        )
        assert config["n_attempts"] == 10

    def test_resource_overrides(self, custom_submission: Path):
        meta = load_metadata(custom_submission)
        config = build_variant_config(
            meta, "treatment", TREATMENT_DIR, "local-build",
            jobs_dir="results/treatment",
        )
        assert config["environment"]["override_cpus"] == 2
        assert config["environment"]["override_memory_mb"] == 4096
        assert config["environment"]["override_storage_mb"] == 20480

    def test_timeout_multipliers(self, custom_submission: Path):
        meta = load_metadata(custom_submission)
        config = build_variant_config(
            meta, "treatment", TREATMENT_DIR, "local-build",
            jobs_dir="results/treatment",
        )
        assert config["agent_timeout_multiplier"] == pytest.approx(2.0)
        assert config["verifier_timeout_multiplier"] == pytest.approx(2.0)
        assert config["agent_setup_timeout_multiplier"] == pytest.approx(0.5)
        assert config["environment_build_timeout_multiplier"] == pytest.approx(1.5)

    def test_default_timeouts_produce_1x_multiplier(self, minimal_submission: Path):
        meta = load_metadata(minimal_submission)
        config = build_variant_config(
            meta, "treatment", TREATMENT_DIR, "local-build",
            jobs_dir="results/treatment",
        )
        assert config["agent_timeout_multiplier"] == pytest.approx(1.0)
        assert config["verifier_timeout_multiplier"] == pytest.approx(1.0)
        assert config["agent_setup_timeout_multiplier"] == pytest.approx(1.0)
        assert config["environment_build_timeout_multiplier"] == pytest.approx(1.0)

    def test_custom_jobs_dir(self, minimal_submission: Path):
        meta = load_metadata(minimal_submission)
        config = build_variant_config(
            meta, "treatment", TREATMENT_DIR, "local-build",
            jobs_dir="/workspace/results/treatment",
        )
        assert config["jobs_dir"] == "/workspace/results/treatment"


class TestGenerateEvalConfigs:
    def test_writes_two_yaml_files(self, minimal_submission: Path, tmp_path: Path):
        out_dir = tmp_path / "configs"
        configs = generate_eval_configs(
            submission_dir=minimal_submission,
            treatment_task_dir=TREATMENT_DIR,
            control_task_dir=CONTROL_DIR,
            output_dir=out_dir,
            eval_mode="prebuilt",
            results_base_dir="eval-results",
            treatment_image_ref=TREATMENT_REF,
            control_image_ref=CONTROL_REF,
        )
        assert (out_dir / "treatment-config.yaml").is_file()
        assert (out_dir / "control-config.yaml").is_file()
        assert "treatment" in configs
        assert "control" in configs

    def test_variant_jobs_dirs_are_separate(
        self, minimal_submission: Path, tmp_path: Path,
    ):
        out_dir = tmp_path / "configs"
        configs = generate_eval_configs(
            submission_dir=minimal_submission,
            treatment_task_dir=TREATMENT_DIR,
            control_task_dir=CONTROL_DIR,
            output_dir=out_dir,
            eval_mode="prebuilt",
            results_base_dir="eval-results",
            treatment_image_ref=TREATMENT_REF,
            control_image_ref=CONTROL_REF,
        )
        assert configs["treatment"]["jobs_dir"] == "eval-results/treatment"
        assert configs["control"]["jobs_dir"] == "eval-results/control"

    def test_each_config_has_single_task(
        self, minimal_submission: Path, tmp_path: Path,
    ):
        out_dir = tmp_path / "configs"
        configs = generate_eval_configs(
            submission_dir=minimal_submission,
            treatment_task_dir=TREATMENT_DIR,
            control_task_dir=CONTROL_DIR,
            output_dir=out_dir,
            eval_mode="local-build",
            results_base_dir="eval-results",
        )
        assert len(configs["treatment"]["tasks"]) == 1
        assert len(configs["control"]["tasks"]) == 1
        assert configs["treatment"]["tasks"][0]["path"] == TREATMENT_DIR
        assert configs["control"]["tasks"][0]["path"] == CONTROL_DIR

    def test_yaml_roundtrips(self, minimal_submission: Path, tmp_path: Path):
        out_dir = tmp_path / "configs"
        configs = generate_eval_configs(
            submission_dir=minimal_submission,
            treatment_task_dir=TREATMENT_DIR,
            control_task_dir=CONTROL_DIR,
            output_dir=out_dir,
            eval_mode="prebuilt",
            results_base_dir="eval-results",
            treatment_image_ref=TREATMENT_REF,
            control_image_ref=CONTROL_REF,
        )
        for variant in ("treatment", "control"):
            loaded = yaml.safe_load(
                (out_dir / f"{variant}-config.yaml").read_text()
            )
            assert loaded["job_name"] == configs[variant]["job_name"]
            assert loaded["n_attempts"] == configs[variant]["n_attempts"]

    def test_creates_output_dir(self, minimal_submission: Path, tmp_path: Path):
        out_dir = tmp_path / "nested" / "dir" / "configs"
        generate_eval_configs(
            submission_dir=minimal_submission,
            treatment_task_dir=TREATMENT_DIR,
            control_task_dir=CONTROL_DIR,
            output_dir=out_dir,
            eval_mode="local-build",
            results_base_dir="eval-results",
        )
        assert (out_dir / "treatment-config.yaml").is_file()


class TestMainCLI:
    def test_prebuilt_mode(self, minimal_submission: Path, tmp_path: Path):
        out_dir = tmp_path / "out"
        rc = main([
            "--submission-dir", str(minimal_submission),
            "--treatment-task-dir", TREATMENT_DIR,
            "--control-task-dir", CONTROL_DIR,
            "--output-dir", str(out_dir),
            "--eval-mode", "prebuilt",
            "--treatment-image-ref", TREATMENT_REF,
            "--control-image-ref", CONTROL_REF,
        ])
        assert rc == 0
        t_config = yaml.safe_load((out_dir / "treatment-config.yaml").read_text())
        assert t_config["environment"]["kwargs"]["image_ref"] == TREATMENT_REF

    def test_local_build_mode(self, minimal_submission: Path, tmp_path: Path):
        out_dir = tmp_path / "out"
        rc = main([
            "--submission-dir", str(minimal_submission),
            "--treatment-task-dir", TREATMENT_DIR,
            "--control-task-dir", CONTROL_DIR,
            "--output-dir", str(out_dir),
            "--eval-mode", "local-build",
        ])
        assert rc == 0
        t_config = yaml.safe_load((out_dir / "treatment-config.yaml").read_text())
        assert "kwargs" not in t_config["environment"]

    def test_prebuilt_missing_refs_exits_error(
        self, minimal_submission: Path, tmp_path: Path,
    ):
        out_dir = tmp_path / "out"
        with pytest.raises(SystemExit) as exc_info:
            main([
                "--submission-dir", str(minimal_submission),
                "--treatment-task-dir", TREATMENT_DIR,
                "--control-task-dir", CONTROL_DIR,
                "--output-dir", str(out_dir),
                "--eval-mode", "prebuilt",
            ])
        assert exc_info.value.code == 2

    def test_nonexistent_submission_dir(self, tmp_path: Path):
        out_dir = tmp_path / "out"
        rc = main([
            "--submission-dir", str(tmp_path / "no-such-dir"),
            "--treatment-task-dir", TREATMENT_DIR,
            "--control-task-dir", CONTROL_DIR,
            "--output-dir", str(out_dir),
            "--eval-mode", "local-build",
        ])
        assert rc == 1
