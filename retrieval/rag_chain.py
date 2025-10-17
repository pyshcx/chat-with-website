from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.embeddings import Embeddings
from typing import List, Dict, Any
from langchain_core.documents import Document
import os
import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleTFIDFEmbeddings(Embeddings):
    """Simple TFIDF embeddings that won't cause threading issues"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=384, 
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2)
        )
        self.is_fitted = False
        self.dim = 384
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        if not self.is_fitted:
            self.vectorizer.fit(texts)
            self.is_fitted = True
            
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
        """Embed a single query"""
        if not self.is_fitted:
            # If not fitted, return zero vector
            return [0.0] * self.dim
            
        vector = self.vectorizer.transform([text]).toarray()[0]
        
        if len(vector) < self.dim:
            vector = np.pad(vector, (0, self.dim - len(vector)))
        elif len(vector) > self.dim:
            vector = vector[:self.dim]
            
        return vector.tolist()

class RAGChain:
    def __init__(self):
        # Get API key from environment
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "Please set GOOGLE_API_KEY environment variable. "
                "Get your free key at: https://makersuite.google.com/app/apikey"
            )
        
        try:
            # Simple local embeddings (no threading issues!)
            self.embeddings = SimpleTFIDFEmbeddings()
            logger.info("✅ Simple TFIDF embeddings initialized")
            
            # Keep using Gemini for LLM
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-exp",
                google_api_key=api_key,
                temperature=0.3,
                max_tokens=1000,
            )
            logger.info("✅ Google Gemini LLM initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize AI components: {e}")
            raise
        
        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
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

Answer:"""

        self.prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )
        
    # ... keep all your existing methods (add_documents, query, etc.)
    def add_documents(self, docs: List[Document]):
        """Add documents to vector store"""
        if not docs:
            raise ValueError("No documents provided")
        
        logger.info(f"🔄 Processing {len(docs)} documents...")
        
        # Split documents into chunks
        splits = self.text_splitter.split_documents(docs)
        logger.info(f"📄 Created {len(splits)} text chunks")
        
        try:
            if self.vector_store is None:
                # Create new vector store
                self.vector_store = FAISS.from_documents(splits, self.embeddings)
                logger.info("🆕 Created new FAISS vector store")
            else:
                # Add to existing vector store
                self.vector_store.add_documents(splits)
                logger.info("➕ Added documents to existing vector store")
            
            # Create/update QA chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 3}
                ),
                return_source_documents=True,
                chain_type_kwargs={"prompt": self.prompt}
            )
            logger.info("🔗 QA chain updated successfully")
            
        except Exception as e:
            logger.error(f"❌ Error processing documents: {e}")
            raise
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG chain"""
        if not question.strip():
            raise ValueError("Question cannot be empty")
        
        if self.qa_chain is None:
            raise ValueError("No documents ingested yet! Please ingest a website first.")
        
        try:
            logger.info(f"🔍 Processing question: {question}")
            result = self.qa_chain({"query": question})
            
            # Extract unique sources
            sources = []
            for doc in result["source_documents"]:
                source = doc.metadata.get("source", "Unknown")
                if source not in sources:
                    sources.append(source)
            
            response = {
                "answer": result["result"].strip(),
                "sources": sources,
                "retrieved_chunks": len(result["source_documents"])
            }
            
            logger.info(f"✅ Generated answer with {len(sources)} sources")
            return response
            
        except Exception as e:
            logger.error(f"❌ Error processing query: {e}")
            return {
                "answer": f"I encountered an error while processing your question: {str(e)}",
                "sources": [],
                "retrieved_chunks": 0
            }
    
    def get_doc_count(self) -> int:
        """Get number of documents in vector store"""
        if self.vector_store is None:
            return 0
        return len(self.vector_store.docstore._dict)
    
    def clear_documents(self):
        """Clear all documents from vector store"""
        self.vector_store = None
        self.qa_chain = None
        logger.info("🗑️ Cleared all documents from vector store")
