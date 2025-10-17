from pydantic import BaseModel
from typing import List, Optional

class IngestRequest(BaseModel):
    url: str
    use_js: bool = False

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    doc_count: int

class StatusResponse(BaseModel):
    status: str
    model: str
    documents: int
    ready: bool
    message: Optional[str] = None
