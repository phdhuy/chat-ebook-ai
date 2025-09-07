from typing import List, Optional

from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str
    history: Optional[str] = ""
    conversation_id: str  # UUID as string


class CitedExcerpt(BaseModel):
    id: int
    text: str
    page: int
    score: float


class QueryResponse(BaseModel):
    answer: str
    cited_excerpts: List[CitedExcerpt]


class UploadRequest(BaseModel):
    conversation_id: str  # UUID as string


class UploadResponse(BaseModel):
    message: str
    conversation_id: str
    chunks: int
