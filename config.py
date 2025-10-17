"""Configuration management for Chat with Website application."""

import os
from typing import Optional
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings(BaseModel):
    """Application settings with validation."""
    
    # API Configuration
    google_api_key: str = Field(..., description="Google AI API key")
    
    # Model Configuration
    model_name: str = Field(default="gemini-1.5-flash", description="Gemini model name")
    temperature: float = Field(default=0.3, ge=0.0, le=1.0, description="Model temperature")
    max_tokens: int = Field(default=1000, gt=0, description="Maximum tokens for generation")
    
    # Text Processing
    chunk_size: int = Field(default=1000, gt=0, description="Text chunk size")
    chunk_overlap: int = Field(default=100, ge=0, description="Text chunk overlap")
    
    # Retrieval Configuration
    retrieval_k: int = Field(default=5, gt=0, description="Number of documents to retrieve")
    
    # API Server Configuration
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, gt=0, lt=65536, description="API port")
    
    # Gradio Configuration
    gradio_host: str = Field(default="0.0.0.0", description="Gradio host")
    gradio_port: int = Field(default=7860, gt=0, lt=65536, description="Gradio port")
    
    # Security
    allowed_origins: list = Field(
        default=["http://localhost:7860", "http://127.0.0.1:7860"],
        description="Allowed CORS origins"
    )
    
    # Timeouts
    request_timeout: int = Field(default=60, gt=0, description="Request timeout in seconds")
    playwright_timeout: int = Field(default=30000, gt=0, description="Playwright timeout in ms")
    
    @validator('google_api_key')
    def validate_api_key(cls, v):
        if not v or v.strip() == "":
            raise ValueError("Google API key is required. Get one at: https://makersuite.google.com/app/apikey")
        return v.strip()
    
    @validator('chunk_overlap')
    def validate_chunk_overlap(cls, v, values):
        if 'chunk_size' in values and v >= values['chunk_size']:
            raise ValueError("Chunk overlap must be less than chunk size")
        return v
    
    class Config:
        env_file = ".env"
        env_prefix = "CHAT_"

def get_settings() -> Settings:
    """Get validated application settings."""
    return Settings(
        google_api_key=os.getenv("GOOGLE_API_KEY", ""),
        model_name=os.getenv("CHAT_MODEL_NAME", "gemini-1.5-flash"),
        temperature=float(os.getenv("CHAT_TEMPERATURE", "0.3")),
        max_tokens=int(os.getenv("CHAT_MAX_TOKENS", "1000")),
        chunk_size=int(os.getenv("CHAT_CHUNK_SIZE", "1000")),
        chunk_overlap=int(os.getenv("CHAT_CHUNK_OVERLAP", "100")),
        retrieval_k=int(os.getenv("CHAT_RETRIEVAL_K", "5")),
        api_host=os.getenv("CHAT_API_HOST", "0.0.0.0"),
        api_port=int(os.getenv("CHAT_API_PORT", "8000")),
        gradio_host=os.getenv("CHAT_GRADIO_HOST", "0.0.0.0"),
        gradio_port=int(os.getenv("CHAT_GRADIO_PORT", "7860")),
        request_timeout=int(os.getenv("CHAT_REQUEST_TIMEOUT", "60")),
        playwright_timeout=int(os.getenv("CHAT_PLAYWRIGHT_TIMEOUT", "30000")),
    )