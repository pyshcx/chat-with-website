"""Improved FastAPI server with better security and error handling."""

import sys
import os
from pathlib import Path

# Fix import paths - add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

try:
    from ingestion.loader import ingest_url, URLValidationError
    from retrieval.rag_chain import RAGChain, RAGChainError
    from api.schemas import IngestRequest, ChatRequest, ChatResponse, StatusResponse
    from config import get_settings
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"📁 Current working directory: {os.getcwd()}")
    print(f"📁 Project root: {project_root}")
    print(f"🐍 Python path: {sys.path}")
    raise

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables
rag_chain: RAGChain = None
settings = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global rag_chain, settings
    
    # Startup
    try:
        settings = get_settings()
        logger.info("✅ Settings loaded successfully")
        
        rag_chain = RAGChain()
        logger.info("✅ RAG Chain initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize application: {e}")
        rag_chain = None
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down application")

app = FastAPI(
    title="Chat with Website",
    description="Improved RAG application powered by Google Gemini",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware with proper security
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins if settings else ["http://localhost:7860"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Accept"],
)

# Error handlers
@app.exception_handler(RAGChainError)
async def rag_chain_error_handler(request: Request, exc: RAGChainError):
    """Handle RAG chain specific errors."""
    logger.error(f"RAG Chain Error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"RAG Chain Error: {str(exc)}"}
    )

@app.exception_handler(URLValidationError)
async def url_validation_error_handler(request: Request, exc: URLValidationError):
    """Handle URL validation errors."""
    logger.error(f"URL Validation Error: {exc}")
    return JSONResponse(
        status_code=400,
        content={"detail": f"Invalid URL: {str(exc)}"}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error occurred"}
    )

# API endpoints
@app.get("/")
async def root() -> Dict[str, Any]:
    """Root endpoint with API information."""
    return {
        "message": "Chat with Website API v2.0", 
        "status": "running",
        "model": settings.model_name if settings else "unknown",
        "endpoints": {
            "ingest": "POST /ingest - Ingest a website URL",
            "ask": "POST /ask - Ask a question about ingested content", 
            "status": "GET /status - Get system status",
            "health": "GET /health - Health check"
        },
        "docs": "/docs - API documentation"
    }

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    if rag_chain is None:
        raise HTTPException(
            status_code=503, 
            detail="Service unavailable - RAG chain not initialized"
        )
    return {"status": "healthy", "message": "Service is running normally"}

@app.post("/ingest")
async def ingest_website(request: IngestRequest) -> Dict[str, Any]:
    """Ingest a website URL into the vector store."""
    if rag_chain is None:
        raise HTTPException(
            status_code=503,
            detail="Service unavailable. RAG chain not initialized. Check your GOOGLE_API_KEY."
        )
    
    logger.info(f"🕷️ Ingesting: {request.url}")
    
    try:
        # Ingest URL with validation
        docs = ingest_url(
            request.url, 
            request.use_js,
            timeout=settings.request_timeout if settings else 60
        )
        
        if not docs:
            raise HTTPException(
                status_code=400,
                detail="No content could be extracted from the URL. Please check if the URL is accessible and contains readable content."
            )
        
        # Add documents to RAG chain
        result = rag_chain.add_documents(docs)
        
        return {
            "message": f"✅ Successfully processed {result['processed_docs']} documents",
            "url": request.url,
            "details": {
                "processed_documents": result['processed_docs'],
                "text_chunks_created": result['text_chunks'],
                "total_documents_in_store": result['total_docs_in_store']
            }
        }
        
    except URLValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RAGChainError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Unexpected ingestion error: {e}")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during ingestion. Please try again."
        )

@app.post("/ask", response_model=ChatResponse)
async def ask_question(request: ChatRequest) -> ChatResponse:
    """Ask a question about ingested content."""
    if rag_chain is None:
        raise HTTPException(
            status_code=503, 
            detail="Service unavailable - RAG chain not initialized"
        )
    
    doc_count = rag_chain.get_doc_count()
    if doc_count == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents ingested yet! Please ingest a website first using the /ingest endpoint."
        )
    
    logger.info(f"❓ Processing question: {request.question[:100]}...")
    
    try:
        result = rag_chain.query(request.question)
        
        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            doc_count=doc_count,
            confidence=result.get("confidence", "medium"),
            retrieved_chunks=result.get("retrieved_chunks", 0)
        )
        
    except RAGChainError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Unexpected query error: {e}")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while processing your question. Please try again."
        )

@app.get("/status", response_model=StatusResponse)
async def get_status() -> StatusResponse:
    """Get comprehensive system status."""
    if rag_chain is None:
        return StatusResponse(
            status="error",
            model="None",
            documents=0,
            ready=False,
            message="RAG chain not initialized. Check GOOGLE_API_KEY and restart the service.",
            details={"error": "Initialization failed"}
        )
    
    try:
        status_info = rag_chain.get_status()
        doc_count = status_info["documents"]
        
        return StatusResponse(
            status="running",
            model=status_info["model"],
            documents=doc_count,
            ready=status_info["ready"],
            message="Ready for questions!" if doc_count > 0 else "No documents ingested yet.",
            details={
                "embeddings": status_info["embeddings"],
                "chunk_size": status_info["chunk_size"],
                "retrieval_k": status_info["retrieval_k"]
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return StatusResponse(
            status="error",
            model="unknown",
            documents=0,
            ready=False,
            message=f"Error retrieving status: {str(e)}",
            details={"error": str(e)}
        )

@app.delete("/documents")
async def clear_documents() -> Dict[str, str]:
    """Clear all ingested documents."""
    if rag_chain is None:
        raise HTTPException(
            status_code=503,
            detail="Service unavailable - RAG chain not initialized"
        )
    
    try:
        rag_chain.clear_documents()
        return {"message": "All documents cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing documents: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to clear documents"
        )

if __name__ == "__main__":
    print("🚀 Starting improved FastAPI server...")
    print(f"📁 Working directory: {os.getcwd()}")
    print(f"📁 Project root: {project_root}")
    
    # Load settings for server configuration
    try:
        server_settings = get_settings()
        host = server_settings.api_host
        port = server_settings.api_port
        print(f"⚙️ Loaded settings: {host}:{port}")
    except Exception as e:
        logger.error(f"Failed to load settings: {e}")
        host = "0.0.0.0"
        port = 8000
        print(f"⚙️ Using default settings: {host}:{port}")
    
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
        access_log=True
    )