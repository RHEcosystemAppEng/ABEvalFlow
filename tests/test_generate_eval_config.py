"""Tests for scripts/generate_eval_config.py — per-variant Harbor eval config generation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from abevalflow.harbor_extensions import OPENSHIFT_ENVIRONMENT_IMPORT_PATH
from scripts.generate_eval_config import (
    build_variant_config,
    generate_eval_configs,
    load_metadata,
    main,
    set_task_docker_image,
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


@pytest.fixture()
def task_dirs(tmp_path: Path) -> tuple[Path, Path]:
    """Scaffolded treatment/control task dirs with task.toml."""
    treatment = tmp_path / "tasks-treatment" / "my-submission"
    control = tmp_path / "tasks-control" / "my-submission"
    for d in (treatment, control):
        d.mkdir(parents=True)
        (d / "task.toml").write_text('version = "1.0"\n\n[environment]\ncpus = 1\nmemory_mb = 2048\n')
    return treatment, control


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


class TestSetTaskDockerImage:
    def test_sets_docker_image(self, tmp_path: Path):
        task = tmp_path / "task"
        task.mkdir()
        (task / "task.toml").write_text('version = "1.0"\n\n[environment]\ncpus = 1\n')
        set_task_docker_image(task, TREATMENT_REF)
        text = (task / "task.toml").read_text()
        assert f'docker_image = "{TREATMENT_REF}"' in text
        assert "cpus = 1" in text

    def test_overwrites_existing_docker_image(self, tmp_path: Path):
        task = tmp_path / "task"
        task.mkdir()
        (task / "task.toml").write_text('version = "1.0"\n\n[environment]\ndocker_image = "old@sha256:x"\n')
        set_task_docker_image(task, TREATMENT_REF)
        text = (task / "task.toml").read_text()
        assert text.count("docker_image") == 1
        assert TREATMENT_REF in text
        assert "old@sha256:x" not in text

    def test_overwrites_docker_image_with_trailing_comment(self, tmp_path: Path):
        task = tmp_path / "task"
        task.mkdir()
        (task / "task.toml").write_text(
            'version = "1.0"\n\n[environment]\ndocker_image = "old@sha256:x" # pinned digest\n'
        )
        set_task_docker_image(task, TREATMENT_REF)
        text = (task / "task.toml").read_text()
        assert text.count("docker_image") == 1
        assert f'docker_image = "{TREATMENT_REF}"' in text
        assert "old@sha256:x" not in text
        assert "# pinned digest" in text  # tomlkit preserves inline comments

    def test_handles_environment_header_with_trailing_comment(self, tmp_path: Path):
        task = tmp_path / "task"
        task.mkdir()
        (task / "task.toml").write_text('version = "1.0"\n\n[environment] # config\ncpus = 1\n')
        set_task_docker_image(task, TREATMENT_REF)
        text = (task / "task.toml").read_text()
        assert "[environment] # config" in text
        assert text.count("[environment]") == 1
        assert f'docker_image = "{TREATMENT_REF}"' in text
        assert "cpus = 1" in text


class TestBuildVariantConfigPrebuilt:
    def test_basic_structure(self, minimal_submission: Path):
        meta = load_metadata(minimal_submission)
        config = build_variant_config(
            meta,
            "treatment",
            TREATMENT_DIR,
            "prebuilt",
            jobs_dir="results/treatment",
            image_ref=TREATMENT_REF,
        )
        assert config["job_name"] == "my-submission-treatment"
        assert config["n_attempts"] == 20
        assert config["environment"]["import_path"] == OPENSHIFT_ENVIRONMENT_IMPORT_PATH
        assert "type" not in config["environment"]
        assert "image_ref" not in config["environment"].get("kwargs", {})
        assert config["environment"]["delete"] is True
        assert len(config["tasks"]) == 1
        assert config["tasks"][0]["path"] == TREATMENT_DIR

    def test_no_image_ref_in_env_kwargs(self, minimal_submission: Path):
        meta = load_metadata(minimal_submission)
        config = build_variant_config(
            meta,
            "treatment",
            TREATMENT_DIR,
            "prebuilt",
            jobs_dir="results/treatment",
            image_ref=TREATMENT_REF,
        )
        assert "image_ref" not in config["environment"].get("kwargs", {})
        assert config["environment"]["kwargs"]["cpu_request"] == "100m"

    def test_control_variant_naming(self, minimal_submission: Path):
        meta = load_metadata(minimal_submission)
        config = build_variant_config(
            meta,
            "control",
            CONTROL_DIR,
            "prebuilt",
            jobs_dir="results/control",
            image_ref=CONTROL_REF,
        )
        assert config["job_name"] == "my-submission-control"
        assert config["tasks"][0]["path"] == CONTROL_DIR

    def test_no_force_build(self, minimal_submission: Path):
        meta = load_metadata(minimal_submission)
        config = build_variant_config(
            meta,
            "treatment",
            TREATMENT_DIR,
            "prebuilt",
            jobs_dir="results/treatment",
            image_ref=TREATMENT_REF,
        )
        assert "force_build" not in config["environment"]

    def test_missing_image_ref_raises(self, minimal_submission: Path):
        meta = load_metadata(minimal_submission)
        with pytest.raises(ValueError, match="image_ref is required"):
            build_variant_config(
                meta,
                "treatment",
                TREATMENT_DIR,
                "prebuilt",
                jobs_dir="results/treatment",
                image_ref="",
            )


class TestBuildVariantConfigLocalBuild:
    def test_docker_type_without_import_path(self, minimal_submission: Path):
        meta = load_metadata(minimal_submission)
        config = build_variant_config(
            meta,
            "treatment",
            TREATMENT_DIR,
            "local-build",
            jobs_dir="results/treatment",
        )
        assert config["environment"]["type"] == "docker"
        assert "import_path" not in config["environment"]
        assert "kwargs" not in config["environment"]

    def test_force_build_enabled(self, minimal_submission: Path):
        meta = load_metadata(minimal_submission)
        config = build_variant_config(
            meta,
            "treatment",
            TREATMENT_DIR,
            "local-build",
            jobs_dir="results/treatment",
        )
        assert config["environment"]["force_build"] is True


class TestCustomMetadataFields:
    def test_n_trials_from_metadata(self, custom_submission: Path):
        meta = load_metadata(custom_submission)
        config = build_variant_config(
            meta,
            "treatment",
            TREATMENT_DIR,
            "local-build",
            jobs_dir="results/treatment",
        )
        assert config["n_attempts"] == 10

    def test_resource_overrides(self, custom_submission: Path):
        meta = load_metadata(custom_submission)
        config = build_variant_config(
            meta,
            "treatment",
            TREATMENT_DIR,
            "local-build",
            jobs_dir="results/treatment",
        )
        assert config["environment"]["override_memory_mb"] == 4096
        assert config["environment"]["override_storage_mb"] == 20480

    def test_timeout_multipliers(self, custom_submission: Path):
        meta = load_metadata(custom_submission)
        config = build_variant_config(
            meta,
            "treatment",
            TREATMENT_DIR,
            "local-build",
            jobs_dir="results/treatment",
        )
        assert config["agent_timeout_multiplier"] == pytest.approx(2.0)
        assert config["verifier_timeout_multiplier"] == pytest.approx(2.0)
        assert config["agent_setup_timeout_multiplier"] == pytest.approx(0.5)
        assert config["environment_build_timeout_multiplier"] == pytest.approx(1.5)

    def test_default_timeouts_produce_1x_multiplier(self, minimal_submission: Path):
        meta = load_metadata(minimal_submission)
        config = build_variant_config(
            meta,
            "treatment",
            TREATMENT_DIR,
            "local-build",
            jobs_dir="results/treatment",
        )
        assert config["agent_timeout_multiplier"] == pytest.approx(1.0)
        assert config["verifier_timeout_multiplier"] == pytest.approx(1.0)
        assert config["agent_setup_timeout_multiplier"] == pytest.approx(1.0)
        assert config["environment_build_timeout_multiplier"] == pytest.approx(1.0)

    def test_custom_jobs_dir(self, minimal_submission: Path):
        meta = load_metadata(minimal_submission)
        config = build_variant_config(
            meta,
            "treatment",
            TREATMENT_DIR,
            "local-build",
            jobs_dir="/workspace/results/treatment",
        )
        assert config["jobs_dir"] == "/workspace/results/treatment"


class TestAgentConfig:
    def test_default_oracle_when_no_llm(self, minimal_submission: Path):
        """No LLM params at all -> oracle agent (empty dict)."""
        meta = load_metadata(minimal_submission)
        config = build_variant_config(
            meta,
            "treatment",
            TREATMENT_DIR,
            "prebuilt",
            jobs_dir="results/treatment",
            image_ref=TREATMENT_REF,
        )
        assert config["agents"] == [{}]

    def test_claude_code_default(self, minimal_submission: Path):
        """Model without wrapper -> claude-code agent with Anthropic env."""
        meta = load_metadata(minimal_submission)
        config = build_variant_config(
            meta,
            "treatment",
            TREATMENT_DIR,
            "prebuilt",
            jobs_dir="results/treatment",
            image_ref=TREATMENT_REF,
            llm_model="claude-sonnet",
            llm_api_base="http://litellm:4000",
            llm_api_key="mock",
        )
        agent = config["agents"][0]
        assert agent["name"] == "claude-code"
        assert agent["model_name"] == "claude-sonnet"
        assert agent["env"]["ANTHROPIC_BASE_URL"] == "http://litellm:4000"
        assert agent["env"]["ANTHROPIC_API_KEY"] == "mock"

    def test_claude_code_model_only(self, minimal_submission: Path):
        """Model without api_base or key -> claude-code, model_name, no env."""
        meta = load_metadata(minimal_submission)
        config = build_variant_config(
            meta,
            "treatment",
            TREATMENT_DIR,
            "prebuilt",
            jobs_dir="results/treatment",
            image_ref=TREATMENT_REF,
            llm_model="claude-sonnet",
        )
        agent = config["agents"][0]
        assert agent["name"] == "claude-code"
        assert agent["model_name"] == "claude-sonnet"
        assert "env" not in agent

    def test_wrapper_agent_opencode(self, minimal_submission: Path):
        """Wrapper agent -> opencode with OPENAI_BASE_URL."""
        meta = load_metadata(minimal_submission)
        config = build_variant_config(
            meta,
            "treatment",
            TREATMENT_DIR,
            "prebuilt",
            jobs_dir="results/treatment",
            image_ref=TREATMENT_REF,
            llm_agent_wrapper="opencode",
            llm_model="openai/llama3",
            llm_api_base="http://litellm:4000",
        )
        agent = config["agents"][0]
        assert agent["name"] == "opencode"
        assert agent["model_name"] == "openai/llama3"
        assert agent["env"]["OPENAI_BASE_URL"] == "http://litellm:4000"

    def test_wrapper_without_api_base(self, minimal_submission: Path):
        """Wrapper agent with model but no api_base -> no env block."""
        meta = load_metadata(minimal_submission)
        config = build_variant_config(
            meta,
            "treatment",
            TREATMENT_DIR,
            "prebuilt",
            jobs_dir="results/treatment",
            image_ref=TREATMENT_REF,
            llm_agent_wrapper="opencode",
            llm_model="openai/gpt-4o",
        )
        agent = config["agents"][0]
        assert agent["name"] == "opencode"
        assert agent["model_name"] == "openai/gpt-4o"
        assert "env" not in agent

    def test_llm_config_in_generated_yaml(
        self,
        minimal_submission: Path,
        tmp_path: Path,
        task_dirs: tuple[Path, Path],
    ):
        treatment, control = task_dirs
        out_dir = tmp_path / "configs"
        configs = generate_eval_configs(
            submission_dir=minimal_submission,
            treatment_task_dir=str(treatment),
            control_task_dir=str(control),
            output_dir=out_dir,
            eval_mode="prebuilt",
            results_base_dir="eval-results",
            treatment_image_ref=TREATMENT_REF,
            control_image_ref=CONTROL_REF,
            llm_model="claude-sonnet",
            llm_api_base="http://litellm:4000",
            llm_api_key="mock",
        )
        for variant in ("treatment", "control"):
            agent = configs[variant]["agents"][0]
            assert agent["name"] == "claude-code"
            assert agent["model_name"] == "claude-sonnet"
            loaded = yaml.safe_load((out_dir / f"{variant}-config.yaml").read_text())
            assert loaded["agents"][0]["name"] == "claude-code"
            assert loaded["agents"][0]["model_name"] == "claude-sonnet"


class TestMetadataLlmOverrides:
    """Test that metadata.yaml llm: block overrides pipeline defaults."""

    @pytest.fixture()
    def submission_with_wrapper_override(self, tmp_path: Path) -> Path:
        """Submission overrides to use opencode wrapper with a different model."""
        sub = tmp_path / "wrapper-override"
        sub.mkdir()
        meta = {
            "name": "wrapper-override",
            "llm": {
                "agent_wrapper": "opencode",
                "model": "openai/llama3",
            },
        }
        (sub / "metadata.yaml").write_text(yaml.dump(meta))
        return sub

    @pytest.fixture()
    def submission_model_override(self, tmp_path: Path) -> Path:
        """Submission overrides only the model name."""
        sub = tmp_path / "model-override"
        sub.mkdir()
        meta = {
            "name": "model-override",
            "llm": {"model": "claude-haiku"},
        }
        (sub / "metadata.yaml").write_text(yaml.dump(meta))
        return sub

    def test_wrapper_override(self, submission_with_wrapper_override: Path):
        """Metadata switches from claude-code default to opencode wrapper."""
        meta = load_metadata(submission_with_wrapper_override)
        config = build_variant_config(
            meta,
            "treatment",
            TREATMENT_DIR,
            "prebuilt",
            jobs_dir="results/treatment",
            image_ref=TREATMENT_REF,
            llm_model="claude-sonnet",
            llm_api_base="http://litellm:4000",
            llm_api_key="mock",
        )
        agent = config["agents"][0]
        assert agent["name"] == "opencode"
        assert agent["model_name"] == "openai/llama3"
        assert agent["env"]["OPENAI_BASE_URL"] == "http://litellm:4000"

    def test_model_override_keeps_claude_code(self, submission_model_override: Path):
        """Overriding only model keeps claude-code as the agent."""
        meta = load_metadata(submission_model_override)
        config = build_variant_config(
            meta,
            "treatment",
            TREATMENT_DIR,
            "prebuilt",
            jobs_dir="results/treatment",
            image_ref=TREATMENT_REF,
            llm_model="claude-sonnet",
            llm_api_base="http://litellm:4000",
            llm_api_key="mock",
        )
        agent = config["agents"][0]
        assert agent["name"] == "claude-code"
        assert agent["model_name"] == "claude-haiku"
        assert agent["env"]["ANTHROPIC_BASE_URL"] == "http://litellm:4000"
        assert agent["env"]["ANTHROPIC_API_KEY"] == "mock"

    def test_no_llm_block_uses_pipeline_defaults(self, minimal_submission: Path):
        """No llm: in metadata -> pipeline defaults (claude-code) used."""
        meta = load_metadata(minimal_submission)
        config = build_variant_config(
            meta,
            "treatment",
            TREATMENT_DIR,
            "prebuilt",
            jobs_dir="results/treatment",
            image_ref=TREATMENT_REF,
            llm_model="claude-sonnet",
            llm_api_base="http://litellm:4000",
            llm_api_key="mock",
        )
        agent = config["agents"][0]
        assert agent["name"] == "claude-code"
        assert agent["model_name"] == "claude-sonnet"

    def test_oracle_override(self, tmp_path: Path):
        """Submission can force oracle mode by clearing model."""
        sub = tmp_path / "force-oracle"
        sub.mkdir()
        meta = {
            "name": "force-oracle",
            "llm": {"model": "", "api_base": "", "api_key": "", "agent_wrapper": ""},
        }
        (sub / "metadata.yaml").write_text(yaml.dump(meta))
        meta = load_metadata(sub)
        config = build_variant_config(
            meta,
            "treatment",
            TREATMENT_DIR,
            "prebuilt",
            jobs_dir="results/treatment",
            image_ref=TREATMENT_REF,
            llm_model="claude-sonnet",
            llm_api_base="http://litellm:4000",
            llm_api_key="mock",
        )
        agent = config["agents"][0]
        assert agent == {}


class TestGenerateEvalConfigs:
    def test_writes_two_yaml_files(
        self,
        minimal_submission: Path,
        tmp_path: Path,
        task_dirs: tuple[Path, Path],
    ):
        treatment, control = task_dirs
        out_dir = tmp_path / "configs"
        configs = generate_eval_configs(
            submission_dir=minimal_submission,
            treatment_task_dir=str(treatment),
            control_task_dir=str(control),
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

    def test_writes_docker_image_to_task_toml(
        self,
        minimal_submission: Path,
        tmp_path: Path,
        task_dirs: tuple[Path, Path],
    ):
        treatment, control = task_dirs
        out_dir = tmp_path / "configs"
        generate_eval_configs(
            submission_dir=minimal_submission,
            treatment_task_dir=str(treatment),
            control_task_dir=str(control),
            output_dir=out_dir,
            eval_mode="prebuilt",
            results_base_dir="eval-results",
            treatment_image_ref=TREATMENT_REF,
            control_image_ref=CONTROL_REF,
        )
        assert f'docker_image = "{TREATMENT_REF}"' in (treatment / "task.toml").read_text()
        assert f'docker_image = "{CONTROL_REF}"' in (control / "task.toml").read_text()

    def test_variant_jobs_dirs_are_separate(
        self,
        minimal_submission: Path,
        tmp_path: Path,
        task_dirs: tuple[Path, Path],
    ):
        treatment, control = task_dirs
        out_dir = tmp_path / "configs"
        configs = generate_eval_configs(
            submission_dir=minimal_submission,
            treatment_task_dir=str(treatment),
            control_task_dir=str(control),
            output_dir=out_dir,
            eval_mode="prebuilt",
            results_base_dir="eval-results",
            treatment_image_ref=TREATMENT_REF,
            control_image_ref=CONTROL_REF,
        )
        assert configs["treatment"]["jobs_dir"] == "eval-results/treatment"
        assert configs["control"]["jobs_dir"] == "eval-results/control"

    def test_each_config_has_single_task(
        self,
        minimal_submission: Path,
        tmp_path: Path,
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

    def test_yaml_roundtrips(
        self,
        minimal_submission: Path,
        tmp_path: Path,
        task_dirs: tuple[Path, Path],
    ):
        treatment, control = task_dirs
        out_dir = tmp_path / "configs"
        configs = generate_eval_configs(
            submission_dir=minimal_submission,
            treatment_task_dir=str(treatment),
            control_task_dir=str(control),
            output_dir=out_dir,
            eval_mode="prebuilt",
            results_base_dir="eval-results",
            treatment_image_ref=TREATMENT_REF,
            control_image_ref=CONTROL_REF,
        )
        for variant in ("treatment", "control"):
            loaded = yaml.safe_load((out_dir / f"{variant}-config.yaml").read_text())
            assert loaded["job_name"] == configs[variant]["job_name"]
            assert loaded["n_attempts"] == configs[variant]["n_attempts"]
            assert loaded["environment"]["import_path"] == OPENSHIFT_ENVIRONMENT_IMPORT_PATH

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
    def test_prebuilt_mode(
        self,
        minimal_submission: Path,
        tmp_path: Path,
        task_dirs: tuple[Path, Path],
    ):
        treatment, control = task_dirs
        out_dir = tmp_path / "out"
        rc = main(
            [
                "--submission-dir",
                str(minimal_submission),
                "--treatment-task-dir",
                str(treatment),
                "--control-task-dir",
                str(control),
                "--output-dir",
                str(out_dir),
                "--eval-mode",
                "prebuilt",
                "--treatment-image-ref",
                TREATMENT_REF,
                "--control-image-ref",
                CONTROL_REF,
            ]
        )
        assert rc == 0
        t_config = yaml.safe_load((out_dir / "treatment-config.yaml").read_text())
        assert t_config["environment"]["import_path"] == OPENSHIFT_ENVIRONMENT_IMPORT_PATH
        assert "image_ref" not in t_config["environment"].get("kwargs", {})
        assert f'docker_image = "{TREATMENT_REF}"' in (treatment / "task.toml").read_text()

    def test_local_build_mode(self, minimal_submission: Path, tmp_path: Path):
        out_dir = tmp_path / "out"
        rc = main(
            [
                "--submission-dir",
                str(minimal_submission),
                "--treatment-task-dir",
                TREATMENT_DIR,
                "--control-task-dir",
                CONTROL_DIR,
                "--output-dir",
                str(out_dir),
                "--eval-mode",
                "local-build",
            ]
        )
        assert rc == 0
        t_config = yaml.safe_load((out_dir / "treatment-config.yaml").read_text())
        assert t_config["environment"]["type"] == "docker"
        assert "import_path" not in t_config["environment"]
        assert t_config["environment"]["force_build"] is True

    def test_prebuilt_missing_refs_exits_error(
        self,
        minimal_submission: Path,
        tmp_path: Path,
    ):
        out_dir = tmp_path / "out"
        with pytest.raises(SystemExit) as exc_info:
            main(
                [
                    "--submission-dir",
                    str(minimal_submission),
                    "--treatment-task-dir",
                    TREATMENT_DIR,
                    "--control-task-dir",
                    CONTROL_DIR,
                    "--output-dir",
                    str(out_dir),
                    "--eval-mode",
                    "prebuilt",
                ]
            )
        assert exc_info.value.code == 2

    def test_nonexistent_submission_dir(self, tmp_path: Path):
        out_dir = tmp_path / "out"
        rc = main(
            [
                "--submission-dir",
                str(tmp_path / "no-such-dir"),
                "--treatment-task-dir",
                TREATMENT_DIR,
                "--control-task-dir",
                CONTROL_DIR,
                "--output-dir",
                str(out_dir),
                "--eval-mode",
                "local-build",
            ]
        )
        assert rc == 1
