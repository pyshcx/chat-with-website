# 🌐 Chat with Website v2.0

An improved RAG (Retrieval-Augmented Generation) application that allows you to chat with any website content using Google Gemini and LangChain.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.1+-purple.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## ✨ Features

- **Smart Content Extraction**: Handles both static and JavaScript-rendered websites
- **Advanced AI**: Powered by Google Gemini 1.5 Flash with intelligent embeddings
- **Robust Error Handling**: Comprehensive error management and graceful fallbacks
- **Security First**: Proper CORS configuration and input validation
- **Easy Configuration**: Environment-based settings with validation
- **Rich API**: RESTful API with automatic documentation
- **User-friendly UI**: Clean Gradio interface for easy interaction

## 🆕 What's New in v2.0

- ✅ **Fixed Model Issues**: Stable Gemini 1.5 Flash instead of experimental versions
- ✅ **Better Embeddings**: Google Generative AI embeddings with sentence-transformers fallback
- ✅ **Enhanced Security**: Proper CORS configuration and input validation
- ✅ **Robust Error Handling**: Comprehensive error management throughout the application
- ✅ **Resource Management**: Proper cleanup of browsers and HTTP sessions
- ✅ **Configuration System**: Centralized settings with validation
- ✅ **Improved Documentation**: Comprehensive setup and usage instructions

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.8 or higher
- Google AI API key ([Get one free](https://makersuite.google.com/app/apikey))

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/pyshcx/chat-with-website.git
cd chat-with-website

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Playwright browsers
playwright install chromium
```

### 3. Configuration

Create a `.env` file in the project root:

```env
# Required: Google AI API Key
GOOGLE_API_KEY=your_api_key_here

# Optional: Model Configuration
CHAT_MODEL_NAME=gemini-1.5-flash
CHAT_TEMPERATURE=0.3
CHAT_MAX_TOKENS=1000

# Optional: Text Processing
CHAT_CHUNK_SIZE=1000
CHAT_CHUNK_OVERLAP=100
CHAT_RETRIEVAL_K=5

# Optional: Server Configuration
CHAT_API_HOST=0.0.0.0
CHAT_API_PORT=8000
CHAT_GRADIO_HOST=0.0.0.0
CHAT_GRADIO_PORT=7860

# Optional: Timeouts
CHAT_REQUEST_TIMEOUT=60
CHAT_PLAYWRIGHT_TIMEOUT=30000
```

### 4. Run the Application

```bash
# Start the API server (Terminal 1)
python api/main.py

# Start the Gradio UI (Terminal 2)
python demo.py
```

### 5. Use the Application

1. Open http://localhost:7860 in your browser
2. Go to the "📥 Ingest Website" tab
3. Enter a website URL (e.g., `https://python.langchain.com/docs/tutorials/rag/`)
4. Click "🕷️ Ingest Website"
5. Go to the "💬 Chat" tab
6. Ask questions about the website content!

## 📖 API Documentation

Once the API is running, visit:
- **Interactive Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

### Key Endpoints

- `POST /ingest` - Ingest a website URL
- `POST /ask` - Ask questions about ingested content
- `GET /status` - Get system status
- `GET /health` - Health check
- `DELETE /documents` - Clear all ingested documents

## 🛠️ Advanced Usage

### Using the API Directly

```python
import requests

# Ingest a website
response = requests.post("http://localhost:8000/ingest", 
    json={"url": "https://example.com", "use_js": False})

# Ask a question
response = requests.post("http://localhost:8000/ask",
    json={"question": "What is this website about?"})
print(response.json()["answer"])
```

### Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_API_KEY` | **Required** | Google AI API key |
| `CHAT_MODEL_NAME` | `gemini-1.5-flash` | Gemini model name |
| `CHAT_TEMPERATURE` | `0.3` | Model temperature (0.0-1.0) |
| `CHAT_CHUNK_SIZE` | `1000` | Text chunk size for processing |
| `CHAT_RETRIEVAL_K` | `5` | Number of chunks to retrieve |
| `CHAT_API_PORT` | `8000` | API server port |
| `CHAT_GRADIO_PORT` | `7860` | Gradio UI port |

## 🔧 Troubleshooting

### Common Issues

**❌ "RAG chain not initialized"**
- Check your `GOOGLE_API_KEY` in the `.env` file
- Ensure the API key is valid and has proper permissions

**❌ "Failed to fetch content from URL"**
- Try enabling JavaScript rendering with `use_js: true`
- Check if the website is accessible and doesn't block bots
- Some websites require specific headers or authentication

**❌ "Playwright timeout"**
- Increase the timeout in your `.env` file: `CHAT_PLAYWRIGHT_TIMEOUT=60000`
- Some websites load slowly or have heavy JavaScript

**❌ "Connection error"**
- Ensure the API server is running on the correct port
- Check firewall settings if running on different machines

### Debug Mode

Set log level to DEBUG for more detailed information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Gradio UI     │◄──►│   FastAPI       │◄──►│   RAG Chain     │
│   (Frontend)    │    │   (Backend)     │    │   (AI Logic)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Input    │    │   URL Ingestion │    │   Vector Store  │
│   & Display     │    │   & Validation  │    │   (FAISS)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   Content       │    │   Google Gemini │
                       │   Extraction    │    │   LLM + Embed   │
                       └─────────────────┘    └─────────────────┘
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and test them
4. Commit your changes: `git commit -m 'Add amazing feature'`
5. Push to the branch: `git push origin feature/amazing-feature`
6. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) for the RAG framework
- [Google AI](https://ai.google/) for Gemini models and embeddings
- [Trafilatura](https://trafilatura.readthedocs.io/) for content extraction
- [FastAPI](https://fastapi.tiangolo.com/) for the API framework
- [Gradio](https://gradio.app/) for the user interface

## 📊 Performance Tips

1. **Use appropriate chunk sizes**: Smaller chunks (500-1000 chars) for detailed QA, larger chunks (1500-2000 chars) for summarization
2. **Enable JS rendering only when needed**: Static sites load faster without Playwright
3. **Monitor memory usage**: Clear documents periodically if processing many large websites
4. **Optimize retrieval**: Adjust `CHAT_RETRIEVAL_K` based on your use case (3-10 chunks typically work well)

---

**Made with ❤️ by [Pranay](https://github.com/pyshcx)**
