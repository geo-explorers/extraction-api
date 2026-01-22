"""Request schemas for API endpoints."""

from pydantic import BaseModel, Field, field_validator


class BatchExtractionRequest(BaseModel):
    """Request body for batch episode extraction."""

    podcast_ids: list[int] = Field(
        ...,
        description="List of podcast IDs to process",
        min_length=1
    )
    target: int | None = Field(
        None,
        description="Maintain claims for latest N episodes per podcast. If None, process all episodes.",
        ge=1
    )
    force: bool = Field(
        default=False,
        description="Force reprocessing of episodes that already have claims"
    )
    continue_on_error: bool = Field(
        default=False,
        description="Continue processing remaining episodes if one fails"
    )
    should_validate: bool = Field(
        default=False,
        description="If true, validate each claim for context independence after extraction"
    )

    @field_validator("podcast_ids")
    @classmethod
    def validate_podcast_ids(cls, v: list[int]) -> list[int]:
        """Validate podcast IDs are positive integers."""
        for podcast_id in v:
            if podcast_id <= 0:
                raise ValueError(f"Invalid podcast_id: {podcast_id}. Must be positive integer.")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "podcast_ids": [854, 907, 994],
                "target": 10,
                "force": False,
                "continue_on_error": True,
                "should_validate": False
            }
        }
