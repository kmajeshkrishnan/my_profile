# Pydantic models for request validation
from pydantic import BaseModel


class RagQueryRequest(BaseModel):
    query: str