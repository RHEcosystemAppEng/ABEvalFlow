"""Pydantic models for submission metadata validation.

The schema defines the structure of metadata.yaml files that accompany
submissions. The schema_version field tracks the format version so the
pipeline can handle older submissions gracefully when the schema evolves
(e.g., new fields added, defaults changed).

Current schema version: 1.0
"""

import re
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CURRENT_SCHEMA_VERSION = "1.0"

_SCHEMA_VERSION_RE = re.compile(r"\d+\.\d+")
_OCI_NAME_RE = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")


class GenerationMode(StrEnum):
    MANUAL = "manual"
    AI = "ai"


class LlmConfig(BaseModel):
    """Optional per-submission LLM overrides.

    When set in metadata.yaml, these override the pipeline-level defaults
    from the pipeline-defaults ConfigMap.  ``None`` means "use pipeline
    default" -- only explicitly set fields override.
    """

    model: str | None = Field(
        default=None,
        description="LLM model name (e.g. claude-sonnet, or openai/llama3 with a wrapper)",
    )
    api_base: str | None = Field(
        default=None,
        description="Base URL of the LLM API proxy",
    )
    api_key: str | None = Field(
        default=None,
        description="API key for the LLM provider (mock for LiteLLM proxy)",
    )
    agent_wrapper: str | None = Field(
        default=None,
        description=(
            "Agent wrapper for non-Claude models (e.g. opencode, qwen-coder). "
            "Empty or None uses claude-code agent directly."
        ),
    )


class ExperimentType(StrEnum):
    SKILL = "skill"
    MODEL = "model"
    PROMPT = "prompt"
    CUSTOM = "custom"


class CopySpec(BaseModel):
    """A source directory and its destination path inside the container."""

    src: str = Field(description="Directory name in submission (e.g. 'skills')")
    dest: str = Field(description="Absolute path in container (e.g. '/skills')")

    @field_validator("src")
    @classmethod
    def _validate_src(cls, v: str) -> str:
        v = v.rstrip("/")
        if ".." in v or v.startswith("/"):
            raise ValueError("src must be a relative top-level directory name")
        if not v:
            raise ValueError("src must not be empty")
        return v

    @field_validator("dest")
    @classmethod
    def _validate_dest(cls, v: str) -> str:
        v = v.rstrip("/")
        if not v:
            raise ValueError("dest must not be empty")
        if ".." in v:
            raise ValueError("dest must not contain '..'")
        if not v.startswith("/"):
            raise ValueError("dest must be an absolute path (start with '/')")
        return v


class VariantSpec(BaseModel):
    """Describes what goes into a single variant (treatment or control)."""

    model_config = ConfigDict(populate_by_name=True)

    copy_dirs: list[CopySpec] = Field(default_factory=list, alias="copy")
    env_from_secrets: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Env vars to inject at runtime via OpenShift Secrets. "
            "Keys are env var names, values are secret references "
            "(e.g. 'secret-name/key'). Raw values are NOT allowed."
        ),
    )

    @model_validator(mode="after")
    def _no_duplicate_src(self) -> "VariantSpec":
        srcs = [c.src for c in self.copy_dirs]
        if len(srcs) != len(set(srcs)):
            raise ValueError("Duplicate src directories in copy spec")
        return self


class ExperimentConfig(BaseModel):
    """A/B experiment configuration embedded in metadata.yaml."""

    type: ExperimentType = Field(
        default=ExperimentType.SKILL,
        description="Experiment type: skill, model, prompt, custom",
    )
    n_trials: int = Field(
        default=20, gt=0, le=100, description="Number of trials per variant"
    )
    treatment: VariantSpec = Field(
        default_factory=lambda: VariantSpec(
            copy=[
                CopySpec(src="skills", dest="/skills"),
                CopySpec(src="docs", dest="/workspace/docs"),
            ]
        ),
    )
    control: VariantSpec = Field(default_factory=VariantSpec)


class SubmissionMetadata(BaseModel):
    """Schema for metadata.yaml in a submission directory.

    Only 'name' is required. All other fields have sensible defaults so
    that a minimal metadata.yaml can be as simple as:

        name: my-submission
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(
        default=CURRENT_SCHEMA_VERSION,
        description=(
            "Format version of this metadata file. Defaults to the current "
            "version. The pipeline uses this to detect older submissions and "
            "apply any necessary migration or compatibility logic."
        ),
    )

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, v: str) -> str:
        if not _SCHEMA_VERSION_RE.fullmatch(v):
            raise ValueError("schema_version must be in 'MAJOR.MINOR' format (e.g. '1.0')")
        return v

    name: str = Field(min_length=1, description="Submission name (OCI-safe: lowercase, alphanumeric, hyphens, dots, underscores)")

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        if not _OCI_NAME_RE.fullmatch(v):
            raise ValueError(
                "name must be OCI-safe: lowercase alphanumeric, hyphens, dots, "
                "underscores only, starting with alphanumeric"
            )
        return v

    description: str | None = Field(default=None, description="Brief description of the submission")
    persona: str | None = Field(
        default=None,
        description="Target persona (e.g. rh-sre, rh-developer). Used as category in Harbor.",
    )
    version: str = Field(default="0.1.0", min_length=1, description="Version string")
    author: str | None = Field(default=None, description="Author or team name")
    tags: list[str] | None = Field(default=None, description="Optional classification tags")
    generation_mode: GenerationMode = Field(
        default=GenerationMode.MANUAL,
        description=(
            "'manual' requires submitter-provided instruction.md and tests. "
            "'ai' generates those files from skills/SKILL.md using an LLM."
        ),
    )

    experiment: ExperimentConfig = Field(
        default_factory=ExperimentConfig,
        description=(
            "A/B experiment configuration. Omit entirely to use the default "
            "skill experiment with N=20 trials."
        ),
    )

    # Harbor timeout and resource configuration (all optional with defaults)
    agent_timeout_sec: float = Field(default=600.0, gt=0, description="Agent solving timeout")
    agent_setup_timeout_sec: float = Field(default=600.0, gt=0, description="Agent install timeout")
    verifier_timeout_sec: float = Field(default=120.0, gt=0, description="Test runner timeout")
    build_timeout_sec: float = Field(default=600.0, gt=0, description="Image build timeout")
    cpus: int = Field(default=1, gt=0, description="CPU cores for trial container")
    memory_mb: int = Field(default=2048, gt=0, description="Memory in MB for trial container")
    storage_mb: int = Field(default=10240, gt=0, description="Storage in MB for trial container")

    llm: LlmConfig | None = Field(
        default=None,
        description=(
            "Optional LLM config overrides for Harbor evaluation agents. "
            "Overrides pipeline-level defaults from the pipeline-defaults ConfigMap."
        ),
    )
