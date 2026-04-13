"""Pydantic models for skill submission metadata validation.

The schema defines the structure of metadata.yaml files that accompany
skill submissions. The schema_version field tracks the format version
so the pipeline can handle older submissions gracefully when the schema
evolves (e.g., new fields added, defaults changed).

Current schema version: 1.0
"""

import re
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

CURRENT_SCHEMA_VERSION = "1.0"

_SCHEMA_VERSION_RE = re.compile(r"\d+\.\d+")


class GenerationMode(StrEnum):
    MANUAL = "manual"
    AI = "ai"


class SubmissionMetadata(BaseModel):
    """Schema for metadata.yaml in a skill submission directory.

    Only 'name' is required. All other fields have sensible defaults so
    that a minimal metadata.yaml can be as simple as:

        name: my-skill
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

    name: str = Field(min_length=1, description="Skill name, must be non-empty")
    description: str | None = Field(default=None, description="Brief description of the skill")
    persona: str | None = Field(
        default=None,
        description="Target persona (e.g. rh-sre, rh-developer). Used as category in Harbor.",
    )
    version: str = Field(default="0.1.0", min_length=1, description="Skill version string")
    author: str | None = Field(default=None, description="Author or team name")
    tags: list[str] | None = Field(default=None, description="Optional classification tags")
    generation_mode: GenerationMode = Field(
        default=GenerationMode.MANUAL,
        description=(
            "Whether the submission includes hand-written tests (manual) or "
            "expects the pipeline to generate instruction/tests from the skill (ai). "
            "AI mode is not yet implemented; defaults to manual."
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
