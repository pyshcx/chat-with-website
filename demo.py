import gradio as gr
import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
API_URL = "http://localhost:8000"

def check_api_status():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_URL}/status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return True, data
        else:
            return False, {"error": "API not responding"}
    except Exception as e:
        return False, {"error": str(e)}

def ingest_website(url, use_js, progress=gr.Progress()):
    """Ingest website via API"""
    if not url.strip():
        return "❌ Please enter a URL"
    
    # Add protocol if missing
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    progress(0.1, desc="Checking API status...")
    
    api_running, status_data = check_api_status()
    if not api_running:
        return f"❌ API Error: {status_data.get('error', 'API not running. Please start with: python api/main.py')}"
    
    progress(0.3, desc="Starting ingestion...")
    
    try:
        response = requests.post(
            f"{API_URL}/ingest", 
            json={"url": url, "use_js": use_js},
            timeout=60
        )
        
        progress(0.8, desc="Processing response...")
        
        if response.status_code == 200:
            data = response.json()
            progress(1.0, desc="Complete!")
            return f"✅ {data['message']}\n📊 Total documents: {data['total_docs']}\n🔗 URL: {data['url']}"
        else:
            error_data = response.json()
            return f"❌ Error: {error_data.get('detail', 'Unknown error')}"
    except requests.Timeout:
        return "❌ Timeout: The website took too long to process. Try a smaller page or enable JS rendering."
    except Exception as e:
        return f"❌ Connection error: {str(e)}\nMake sure the API is running: python api/main.py"

def ask_question(question, progress=gr.Progress()):
    """Ask question via API"""
    if not question.strip():
        return "❌ Please enter a question"
    
    progress(0.2, desc="Checking API status...")
    
    api_running, status_data = check_api_status()
    if not api_running:
        return f"❌ API Error: {status_data.get('error', 'API not running')}"
    
    if not status_data.get('ready', False):
        return "❌ No documents ingested yet! Please ingest a website first in the 'Ingest Website' tab."
    
    progress(0.5, desc="Processing question...")
    
    try:
        response = requests.post(
            f"{API_URL}/ask", 
            json={"question": question},
            timeout=30
        )
        
        progress(0.8, desc="Generating response...")
        
        if response.status_code == 200:
            result = response.json()
            sources_text = "\n".join([f"• {src}" for src in result["sources"]])
            progress(1.0, desc="Complete!")
            return f"{result['answer']}\n\n**📚 Sources ({len(result['sources'])}):**\n{sources_text}\n\n**📄 Documents in database:** {result['doc_count']}"
        else:
            error_data = response.json()
            return f"❌ Error: {error_data.get('detail', 'Unknown error')}"
    except requests.Timeout:
        return "❌ Timeout: The question took too long to process."
    except Exception as e:
        return f"❌ Connection error: {str(e)}\nMake sure the API is running: python api/main.py"

def get_system_status():
    """Get current system status"""
    api_running, status_data = check_api_status()
    if not api_running:
        return f"❌ API Status: Not Running\nError: {status_data.get('error')}"
    
    return f"""✅ API Status: Running
🤖 Model: {status_data.get('model', 'Unknown')}
📚 Documents: {status_data.get('documents', 0)}
🟢 Ready: {"Yes" if status_data.get('ready') else "No"}
💬 Message: {status_data.get('message', 'N/A')}"""

# Gradio Interface
with gr.Blocks(
    theme=gr.themes.Soft(),
    title="Chat with Website",
    css=".gradio-container {max-width: 1200px !important}"
) as demo:
    gr.Markdown("""
    # 🌐 Chat with Any Website
    
    **Powered by Google Gemini & LangChain**
    
    1. **Ingest**: Add any website to the knowledge base
    2. **Chat**: Ask questions about the content
    3. **Sources**: Get citations for all answers
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            with gr.Tab("📥 Ingest Website"):
                gr.Markdown("### Add a website to the knowledge base")
                
                url_input = gr.Textbox(
                    label="Website URL",
                    placeholder="https://example.com or just example.com",
                    lines=1
                )
                
                with gr.Row():
                    js_checkbox = gr.Checkbox(
                        label="Use JavaScript rendering (slower, for dynamic sites)",
                        value=False
                    )
                
                ingest_btn = gr.Button("🕷️ Ingest Website", variant="primary", size="lg")
                ingest_output = gr.Textbox(
                    label="Ingestion Status",
                    lines=4,
                    max_lines=10
                )
                
                gr.Examples(
                    examples=[
                        ["https://python.langchain.com/docs/tutorials/rag/", False],
                        ["https://docs.python.org/3/tutorial/", False],
                        ["https://fastapi.tiangolo.com/", False],
                    ],
                    inputs=[url_input, js_checkbox],
                    label="Example URLs"
                )
            
            with gr.Tab("💬 Chat"):
                gr.Markdown("### Ask questions about ingested content")
                
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="What is this website about?",
                    lines=2
                )
                
                ask_btn = gr.Button("🤔 Ask Question", variant="primary", size="lg")
                answer_output = gr.Textbox(
                    label="Answer with Sources",
                    lines=10,
                    max_lines=20
                )
                
                gr.Examples(
                    examples=[
                        ["What is this website about?"],
                        ["Can you summarize the main topics?"],
                        ["What are the key concepts mentioned?"],
                        ["How does this technology work?"],
                    ],
                    inputs=[question_input],
                    label="Example Questions"
                )
        
        with gr.Column(scale=1):
            gr.Markdown("### 📊 System Status")
            status_output = gr.Textbox(
                label="Current Status",
                lines=8,
                max_lines=10
            )
            status_btn = gr.Button("🔄 Refresh Status", size="sm")
            
            gr.Markdown("""
            ### 💡 Tips
            - Start the API first: `python api/main.py`
            - Use JS rendering for dynamic sites (React, etc.)
            - Ask specific questions for better answers
            - Check sources for verification
            
            ### 🔧 Setup
            1. Set `GOOGLE_API_KEY` in `.env`
            2. Install: `pip install -r requirements.txt`
            3. Run API: `python api/main.py`
            4. Run UI: `python demo.py`
            """)
    
    # Event handlers
    ingest_btn.click(
        ingest_website,
        inputs=[url_input, js_checkbox],
        outputs=[ingest_output]
    )
    
    ask_btn.click(
        ask_question,
        inputs=[question_input],
        outputs=[answer_output]
    )
    
    status_btn.click(
        get_system_status,
        outputs=[status_output]
    )
    
    # Auto-refresh status on load
    demo.load(
        get_system_status,
        outputs=[status_output]
    )

if __name__ == "__main__":
    host = os.getenv("GRADIO_HOST", "0.0.0.0")
    port = int(os.getenv("GRADIO_PORT", 7860))
    demo.launch(
        server_name=host,
        server_port=port,
        share=False,
        show_error=True
    )
