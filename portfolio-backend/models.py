# Pydantic models for request validation
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


class RagQueryRequest(BaseModel):
    query: str = Field(..., description="The question to ask about the resume", min_length=1, max_length=1000)
    include_metadata: bool = Field(default=False, description="Whether to include metadata in response")


class RagQueryResponse(BaseModel):
    success: bool
    response: Optional[str] = None
    error: Optional[str] = None
    processing_time: float
    query: str
    metadata: Optional[Dict[str, Any]] = None


class ResumeUploadRequest(BaseModel):
    resume_path: str = Field(..., description="Path to the resume file")
    working_dir: Optional[str] = Field(default=".", description="Working directory for RAG service")


class ServiceInfoResponse(BaseModel):
    service_name: str
    initialized: bool
    working_dir: Optional[str] = None
    resume_path: Optional[str] = None
    total_queries: int
    successful_queries: int
    success_rate: float
    initialization_time: Optional[float] = None


class HealthCheckResponse(BaseModel):
    status: str
    error: Optional[str] = None
    initialized: Optional[bool] = None
    total_queries: Optional[int] = None
    successful_queries: Optional[int] = None