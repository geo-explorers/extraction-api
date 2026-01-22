"""Response schemas for API endpoints."""

from pydantic import BaseModel, Field


class SimplifiedExtractionResponse(BaseModel):
    """Simplified response for single episode extraction (statistics only)."""

    episode_id: int = Field(..., description="Episode ID that was processed")
    processing_time_seconds: float = Field(
        ..., description="Total processing time in seconds"
    )
    claims_count: int = Field(..., description="Number of claims extracted and saved")
    quotes_count: int = Field(..., description="Number of quotes extracted and saved")
    validated_count: int | None = Field(
        None, description="Number of claims that passed context-independence validation (only if should_validate=true)"
    )
    invalid_count: int | None = Field(
        None, description="Number of claims that failed context-independence validation (only if should_validate=true)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "episode_id": 123,
                "processing_time_seconds": 342.5,
                "claims_count": 28,
                "quotes_count": 85,
                "validated_count": 25,
                "invalid_count": 3
            }
        }


class SimplifiedBatchExtractionResponse(BaseModel):
    """Simplified response for batch episode extraction."""

    results: list[SimplifiedExtractionResponse] = Field(
        default_factory=list,
        description="Results for each episode"
    )
    summary: "BatchExtractionSummary" = Field(
        ..., description="Summary statistics"
    )
    errors: dict[int, str] = Field(
        default_factory=dict,
        description="Map of episode_id to error message for failed episodes"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "results": [
                    {
                        "episode_id": 123,
                        "processing_time_seconds": 342.5,
                        "claims_count": 28,
                        "quotes_count": 85
                    },
                    {
                        "episode_id": 124,
                        "processing_time_seconds": 351.2,
                        "claims_count": 32,
                        "quotes_count": 96
                    }
                ],
                "summary": {
                    "total_episodes": 2,
                    "successful_episodes": 2,
                    "failed_episodes": 0,
                    "total_claims_extracted": 60,
                    "total_processing_time_seconds": 693.7
                },
                "errors": {}
            }
        }


class BatchExtractionSummary(BaseModel):
    """Summary statistics for batch extraction."""

    total_episodes: int = Field(..., description="Total episodes processed")
    successful_episodes: int = Field(..., description="Successfully processed episodes")
    failed_episodes: int = Field(..., description="Failed episodes")
    total_claims_extracted: int = Field(..., description="Total claims across all episodes")
    total_processing_time_seconds: float = Field(
        ..., description="Total processing time for all episodes"
    )
