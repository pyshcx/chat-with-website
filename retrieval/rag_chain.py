"""Improved RAG Chain implementation with better error handling and embeddings."""

import logging
import os
from typing import List, Dict, Any, Optional
from contextlib import contextmanager

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

from config import get_settings

# Set environment variables to prevent threading issues on macOS
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGChainError(Exception):
    """Custom exception for RAG Chain errors."""
    pass

class RAGChain:
    """Improved RAG Chain with better error handling and configuration."""
    
    def __init__(self):
        """Initialize RAG Chain with validated settings."""
        try:
            self.settings = get_settings()
            logger.info("✅ Settings loaded and validated")
        except Exception as e:
            logger.error(f"❌ Failed to load settings: {e}")
            raise RAGChainError(f"Configuration error: {e}")
        
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize AI components with proper error handling."""
        # Initialize embeddings based on configuration
        if self.settings.use_local_embeddings:
            logger.info("💻 Using local embeddings (configured preference)")
            self._initialize_local_embeddings()
        else:
            logger.info("🌐 Attempting to use Google embeddings...")
            if not self._initialize_google_embeddings():
                logger.info("🔄 Falling back to local embeddings")
                self._initialize_local_embeddings()
        
        # Initialize LLM
        try:
            self.llm = ChatGoogleGenerativeAI(
                model=self.settings.model_name,
                google_api_key=self.settings.google_api_key,
                temperature=self.settings.temperature,
                max_tokens=self.settings.max_tokens,
            )
            logger.info(f"✅ {self.settings.model_name} LLM initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize LLM: {e}")
            raise RAGChainError(f"LLM initialization failed: {e}")
        
        # Initialize other components
        self.vector_store: Optional[FAISS] = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.qa_chain = None
        
        self.prompt_template = """You are a helpful assistant that answers questions based on provided context.

Context: {context}

Question: {question}

Instructions:
1. Answer the question using ONLY the information provided in the context
2. If the answer is not in the context, say "I don't have enough information to answer this question based on the provided content"
3. Be concise but comprehensive
4. Include relevant details from the context
5. If you find conflicting information, mention it

Answer:"""

        self.prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )
    
    def _initialize_google_embeddings(self) -> bool:
        """Try to initialize Google embeddings, return True if successful."""
        try:
            # Test with a simple embedding call to check quota
            test_embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=self.settings.google_api_key
            )
            
            # Test with a simple text to verify quota
            test_embeddings.embed_query("test")
            
            self.embeddings = test_embeddings
            logger.info("✅ Google Generative AI embeddings initialized and tested")
            return True
            
        except Exception as e:
            error_msg = str(e).lower()
            if "quota" in error_msg or "429" in error_msg:
                logger.warning(f"📊 Quota exceeded for Google embeddings: {e}")
            elif "401" in error_msg or "unauthorized" in error_msg:
                logger.warning(f"🔑 Invalid API key for Google embeddings: {e}")
            else:
                logger.warning(f"⚠️ Failed to initialize Google embeddings: {e}")
            return False
    
    def _initialize_local_embeddings(self):
        """Initialize local sentence transformer embeddings using the maintained package."""
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            
            # Use sentence-transformers prefix for model name if not already present
            model_name = self.settings.local_embedding_model
            if not model_name.startswith("sentence-transformers/"):
                model_name = f"sentence-transformers/{model_name}"
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cpu'},  # Force CPU to avoid GPU issues
                encode_kwargs={'normalize_embeddings': True}  # Better similarity scores
            )
            logger.info(f"✅ HuggingFace embeddings initialized: {model_name}")
            
        except ImportError:
            logger.error("❌ langchain-huggingface not installed. Installing...")
            try:
                import subprocess
                import sys
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", 
                    "langchain-huggingface>=0.0.3", 
                    "sentence-transformers>=3.0.1"
                ])
                
                from langchain_huggingface import HuggingFaceEmbeddings
                
                model_name = self.settings.local_embedding_model
                if not model_name.startswith("sentence-transformers/"):
                    model_name = f"sentence-transformers/{model_name}"
                
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                logger.info(f"✅ Installed and initialized HuggingFace embeddings: {model_name}")
                
            except Exception as install_error:
                logger.error(f"❌ Failed to install langchain-huggingface: {install_error}")
                logger.info("🔄 Trying fallback to basic embeddings...")
                self._initialize_fallback_embeddings()
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize HuggingFace embeddings: {e}")
            logger.info("🔄 Trying fallback to basic embeddings...")
            self._initialize_fallback_embeddings()
    
    def _initialize_fallback_embeddings(self):
        """Initialize a simple TF-IDF based embedding as last resort."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            import numpy as np
            from langchain_core.embeddings import Embeddings
            
            class TFIDFEmbeddings(Embeddings):
                def __init__(self, max_features=384):
                    self.vectorizer = TfidfVectorizer(
                        max_features=max_features,
                        stop_words='english',
                        lowercase=True,
                        ngram_range=(1, 2)
                    )
                    self.fitted = False
                    self.dim = max_features
                
                def embed_documents(self, texts: List[str]) -> List[List[float]]:
                    if not self.fitted:
                        self.vectorizer.fit(texts)
                        self.fitted = True
                    
                    vectors = self.vectorizer.transform(texts).toarray()
                    # Pad or truncate to fixed dimension
                    result = []
                    for vector in vectors:
                        if len(vector) < self.dim:
                            vector = np.pad(vector, (0, self.dim - len(vector)))
                        elif len(vector) > self.dim:
                            vector = vector[:self.dim]
                        result.append(vector.tolist())
                    return result
                
                def embed_query(self, text: str) -> List[float]:
                    if not self.fitted:
                        return [0.0] * self.dim
                    
                    vector = self.vectorizer.transform([text]).toarray()[0]
                    if len(vector) < self.dim:
                        vector = np.pad(vector, (0, self.dim - len(vector)))
                    elif len(vector) > self.dim:
                        vector = vector[:self.dim]
                    return vector.tolist()
            
            self.embeddings = TFIDFEmbeddings()
            logger.info("✅ Fallback TF-IDF embeddings initialized")
            
        except Exception as e:
            logger.error(f"❌ All embedding methods failed: {e}")
            raise RAGChainError("Could not initialize any embedding model")
    
    @contextmanager
    def _error_context(self, operation: str):
        """Context manager for consistent error handling."""
        try:
            logger.info(f"🔄 Starting {operation}")
            yield
            logger.info(f"✅ Completed {operation}")
        except Exception as e:
            logger.error(f"❌ Error in {operation}: {e}")
            raise RAGChainError(f"{operation} failed: {str(e)}")
    
    def add_documents(self, docs: List[Document]) -> Dict[str, Any]:
        """Add documents to vector store with improved error handling."""
        if not docs:
            raise RAGChainError("No documents provided")
        
        with self._error_context(f"processing {len(docs)} documents"):
            # Split documents into chunks
            splits = self.text_splitter.split_documents(docs)
            logger.info(f"📄 Created {len(splits)} text chunks")
            
            if not splits:
                raise RAGChainError("No content could be extracted from documents")
            
            try:
                # Create or update vector store
                if self.vector_store is None:
                    self.vector_store = FAISS.from_documents(splits, self.embeddings)
                    logger.info("🆕 Created new FAISS vector store")
                else:
                    # More efficient: add documents to existing store
                    texts = [doc.page_content for doc in splits]
                    metadatas = [doc.metadata for doc in splits]
                    self.vector_store.add_texts(texts, metadatas)
                    logger.info("➕ Added documents to existing vector store")
                    
            except Exception as e:
                error_msg = str(e).lower()
                if "quota" in error_msg or "429" in error_msg:
                    logger.error(f"📊 Embedding quota exceeded during document processing")
                    if not self.settings.use_local_embeddings:
                        logger.info("🔄 Switching to local embeddings and retrying...")
                        self._initialize_local_embeddings()
                        
                        # Retry with local embeddings
                        if self.vector_store is None:
                            self.vector_store = FAISS.from_documents(splits, self.embeddings)
                            logger.info("🆕 Created new FAISS vector store with local embeddings")
                        else:
                            texts = [doc.page_content for doc in splits]
                            metadatas = [doc.metadata for doc in splits]
                            self.vector_store.add_texts(texts, metadatas)
                            logger.info("➕ Added documents to existing vector store with local embeddings")
                    else:
                        raise
                else:
                    raise
            
            # Create/update QA chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": self.settings.retrieval_k}
                ),
                return_source_documents=True,
                chain_type_kwargs={"prompt": self.prompt}
            )
            
            return {
                "processed_docs": len(docs),
                "text_chunks": len(splits),
                "total_docs_in_store": self.get_doc_count()
            }
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG chain with improved error handling."""
        if not question or not question.strip():
            raise RAGChainError("Question cannot be empty")
        
        if self.qa_chain is None:
            raise RAGChainError("No documents ingested yet! Please ingest a website first.")
        
        with self._error_context(f"processing question: {question[:50]}..."):
            result = self.qa_chain({"query": question.strip()})
            
            # Extract unique sources
            sources = []
            source_chunks = []
            for doc in result.get("source_documents", []):
                source = doc.metadata.get("source", "Unknown")
                if source not in sources:
                    sources.append(source)
                source_chunks.append({
                    "source": source,
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                })
            
            response = {
                "answer": result["result"].strip(),
                "sources": sources,
                "retrieved_chunks": len(result.get("source_documents", [])),
                "source_chunks": source_chunks[:3],  # Limit to first 3 for brevity
                "confidence": "high" if len(result.get("source_documents", [])) >= 3 else "medium"
            }
            
            return response
    
    def get_doc_count(self) -> int:
        """Get number of documents in vector store."""
        if self.vector_store is None:
            return 0
        try:
            return len(self.vector_store.docstore._dict)
        except Exception:
            # Fallback count method
            return len(self.vector_store.index_to_docstore_id) if hasattr(self.vector_store, 'index_to_docstore_id') else 0
    
    def get_status(self) -> Dict[str, Any]:
        """Get detailed status information."""
        embedding_type = type(self.embeddings).__name__
        is_local = "HuggingFace" in embedding_type or "TFIDF" in embedding_type
        
        return {
            "model": self.settings.model_name,
            "embeddings": embedding_type,
            "documents": self.get_doc_count(),
            "ready": self.qa_chain is not None and self.get_doc_count() > 0,
            "chunk_size": self.settings.chunk_size,
            "retrieval_k": self.settings.retrieval_k,
            "using_local_embeddings": is_local
        }
    
    def clear_documents(self):
        """Clear all documents from vector store."""
        with self._error_context("clearing documents"):
            self.vector_store = None
            self.qa_chain = None
            logger.info("🗑️ Cleared all documents from vector store")
    
    def save_vector_store(self, path: str):
        """Save vector store to disk."""
        if self.vector_store is None:
            raise RAGChainError("No vector store to save")
        
        with self._error_context(f"saving vector store to {path}"):
            self.vector_store.save_local(path)
    
    def load_vector_store(self, path: str):
        """Load vector store from disk."""
        with self._error_context(f"loading vector store from {path}"):
            self.vector_store = FAISS.load_local(
                path, 
                self.embeddings,
                allow_dangerous_deserialization=True  # Only if you trust the source
            )
            
            # Recreate QA chain
            if self.vector_store:
                self.qa_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=self.vector_store.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": self.settings.retrieval_k}
                    ),
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": self.prompt}
                )
