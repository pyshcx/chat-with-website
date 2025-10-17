import sys
import os
# Fix Python path FIRST, before any other imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from dotenv import load_dotenv

from ingestion.loader import ingest_url
from retrieval.rag_chain import RAGChain
from api.schemas import IngestRequest, ChatRequest, ChatResponse, StatusResponse

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Chat with Website",
    description="RAG application powered by Google Gemini",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG chain
try:
    rag_chain = RAGChain()
    print("✅ RAG Chain initialized with Google Gemini!")
except Exception as e:
    print(f"❌ Failed to initialize RAG chain: {e}")
    print("Make sure to set GOOGLE_API_KEY in your environment")
    rag_chain = None

@app.get("/")
async def root():
    return {
        "message": "Chat with Website API", 
        "status": "running", 
        "model": "Google Gemini 1.5 Flash",
        "endpoints": ["/ingest", "/ask", "/status"]
    }

@app.post("/ingest")
async def ingest_website(request: IngestRequest):
    """Ingest a website URL into the vector store"""
    if not rag_chain:
        raise HTTPException(
            status_code=500, 
            detail="RAG chain not initialized. Check your GOOGLE_API_KEY."
        )
    
    try:
        print(f"🕷️ Ingesting: {request.url}")
        docs = ingest_url(request.url, request.use_js)
        
        if not docs:
            raise HTTPException(
                status_code=400, 
                detail="Failed to extract content from URL. Please check if the URL is accessible."
            )
        
        rag_chain.add_documents(docs)
        doc_count = rag_chain.get_doc_count()
        
        return {
            "message": f"✅ Successfully ingested {len(docs)} documents",
            "url": request.url,
            "total_docs": doc_count,
            "extracted_docs": len(docs)
        }
    except Exception as e:
        print(f"❌ Ingestion error: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Ingestion failed: {str(e)}"
        )

@app.post("/ask", response_model=ChatResponse)
async def ask_question(request: ChatRequest):
    """Ask a question about ingested content"""
    if not rag_chain:
        raise HTTPException(status_code=500, detail="RAG chain not initialized")
    
    doc_count = rag_chain.get_doc_count()
    if doc_count == 0:
        raise HTTPException(
            status_code=400, 
            detail="No documents ingested yet! Please ingest a website first using /ingest endpoint."
        )
    
    try:
        print(f"❓ Question: {request.question}")
        result = rag_chain.query(request.question)
        
        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            doc_count=doc_count
        )
    except Exception as e:
        print(f"❌ Query error: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Query failed: {str(e)}"
        )

@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get system status"""
    if not rag_chain:
        return StatusResponse(
            status="error",
            model="None",
            documents=0,
            ready=False,
            message="RAG chain not initialized. Check GOOGLE_API_KEY."
        )
    
    doc_count = rag_chain.get_doc_count()
    return StatusResponse(
        status="running",
        model="Google Gemini 1.5 Flash",
        documents=doc_count,
        ready=doc_count > 0,
        message="Ready for questions!" if doc_count > 0 else "No documents ingested yet."
    )

if __name__ == "__main__":
    print("🚀 Starting FastAPI server...")
    import uvicorn
    uvicorn.run(
        "api.main:app",  # Import string format
        host="0.0.0.0", 
        port=8000, 
        reload=False,  # Disable reload to avoid warnings
        log_level="info"
    )

