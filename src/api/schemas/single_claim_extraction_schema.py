"""Request schema for single claim extraction endpoint."""

from pydantic import BaseModel, Field


class SingleClaimExtractionRequest(BaseModel):
    """Request body for single episode claim extraction."""

    episode_id: int = Field(
        ..., description="ID of the episode to extract claims from", ge=1
    )
    force: bool = Field(
        default=False, description="Force reprocessing even if claims already exist"
    )
    should_validate: bool = Field(
        default=False, description="Whether to validate extracted claims"
    )

    class Config:
        json_schema_extra = {
            "example": {"episode_id": 123, "force": False, "should_validate": True}
        }
