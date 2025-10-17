"""Improved API schemas with better validation and additional fields."""

from pydantic import BaseModel, Field, validator, HttpUrl
from typing import List, Optional, Dict, Any
import re

class IngestRequest(BaseModel):
    """Request schema for ingesting a website."""
    url: str = Field(..., description="Website URL to ingest")
    use_js: bool = Field(default=False, description="Use JavaScript rendering for dynamic sites")
    
    @validator('url')
    def validate_url(cls, v):
        """Validate and normalize URL."""
        if not v or not v.strip():
            raise ValueError("URL cannot be empty")
        
        url = v.strip()
        
        # Add protocol if missing
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        # Basic URL validation
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+'  # domain...
            r'(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # host...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        
        if not url_pattern.match(url):
            raise ValueError("Invalid URL format")
        
        return url
    
    class Config:
        schema_extra = {
            "example": {
                "url": "https://example.com",
                "use_js": False
            }
        }

class ChatRequest(BaseModel):
    """Request schema for asking questions."""
    question: str = Field(
        ..., 
        min_length=1, 
        max_length=1000,
        description="Question to ask about the ingested content"
    )
    
    @validator('question')
    def validate_question(cls, v):
        """Validate question content."""
        if not v or not v.strip():
            raise ValueError("Question cannot be empty")
        return v.strip()
    
    class Config:
        schema_extra = {
            "example": {
                "question": "What is this website about?"
            }
        }

class ChatResponse(BaseModel):
    """Response schema for chat answers."""
    answer: str = Field(..., description="Generated answer")
    sources: List[str] = Field(..., description="Source URLs used for the answer")
    doc_count: int = Field(..., description="Total documents in the knowledge base")
    confidence: Optional[str] = Field(default="medium", description="Confidence level of the answer")
    retrieved_chunks: Optional[int] = Field(default=0, description="Number of text chunks retrieved")
    
    class Config:
        schema_extra = {
            "example": {
                "answer": "This website is about...",
                "sources": ["https://example.com"],
                "doc_count": 5,
                "confidence": "high",
                "retrieved_chunks": 3
            }
        }

class StatusResponse(BaseModel):
    """Response schema for system status."""
    status: str = Field(..., description="System status (running, error, etc.)")
    model: str = Field(..., description="AI model being used")
    documents: int = Field(..., description="Number of documents in the knowledge base")
    ready: bool = Field(..., description="Whether the system is ready to answer questions")
    message: Optional[str] = Field(default=None, description="Status message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional status details")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "running",
                "model": "gemini-1.5-flash",
                "documents": 10,
                "ready": True,
                "message": "Ready for questions!",
                "details": {
                    "embeddings": "GoogleGenerativeAIEmbeddings",
                    "chunk_size": 1000,
                    "retrieval_k": 5
                }
            }
        }

class ErrorResponse(BaseModel):
    """Standard error response schema."""
    detail: str = Field(..., description="Error message")
    error_type: Optional[str] = Field(default=None, description="Type of error")
    timestamp: Optional[str] = Field(default=None, description="Error timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "detail": "Invalid URL format",
                "error_type": "ValidationError",
                "timestamp": "2025-10-17T18:30:00Z"
            }
        }
