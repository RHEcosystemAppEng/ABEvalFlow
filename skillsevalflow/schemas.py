"""Pydantic models for skill submission metadata validation."""

import re
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

_SCHEMA_VERSION_RE = re.compile(r"\d+\.\d+")


class GenerationMode(StrEnum):
    MANUAL = "manual"
    AI = "ai"


class SubmissionMetadata(BaseModel):
    """Schema for metadata.yaml in a skill submission directory.

    The schema_version field enables forward-compatible evolution
    of the metadata format without breaking existing submissions.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(
        description="Schema version for forward compatibility (e.g. '1.0')",
    )

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, v: str) -> str:
        if not _SCHEMA_VERSION_RE.fullmatch(v):
            raise ValueError("schema_version must be in 'MAJOR.MINOR' format (e.g. '1.0')")
        return v

    name: str = Field(min_length=1, description="Skill name, must be non-empty")
    description: str = Field(min_length=1, description="Brief description of the skill")
    persona: str = Field(min_length=1, description="Target persona (e.g. rh-sre, rh-developer)")
    version: str = Field(min_length=1, description="Skill version string")
    author: str = Field(min_length=1, description="Author or team name")
    tags: list[str] | None = Field(default=None, description="Optional classification tags")
    generation_mode: GenerationMode = Field(
        description="Whether the skill was created manually or AI-generated",
    )
