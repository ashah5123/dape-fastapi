"""
Pydantic models for FastAPI request/response schemas.
"""

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """Request schema for text generation endpoint."""
    prompt: str = Field(..., description="Input prompt/question", min_length=1, max_length=1000)
    max_new_tokens: int = Field(default=256, description="Maximum number of tokens to generate", ge=1, le=1024)


class GenerateResponse(BaseModel):
    """Response schema for text generation endpoint."""
    output: str = Field(..., description="Generated text response")
    duration_s: float = Field(..., description="Generation time in seconds", ge=0.0)
