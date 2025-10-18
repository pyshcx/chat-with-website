#!/usr/bin/env python3
"""Simple script to start the API server from project root."""

import os
import sys
from pathlib import Path

# Ensure we're in the project root
project_root = Path(__file__).parent
os.chdir(project_root)

# Add project root to Python path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print(f"📁 Project root: {project_root}")
print(f"📁 Working directory: {os.getcwd()}")

try:
    from api.main import app
    import uvicorn
    from config import get_settings
    
    print("✅ All imports successful!")
    
    # Load settings
    try:
        settings = get_settings()
        host = settings.api_host
        port = settings.api_port
        print(f"⚙️ Using settings: {host}:{port}")
    except Exception as e:
        print(f"⚠️ Could not load settings: {e}")
        print("Using default values...")
        host = "0.0.0.0"
        port = 8000
    
    print("🚀 Starting FastAPI server...")
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )
    
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("\n🔧 Troubleshooting:")
    print("1. Make sure you're in the project root directory")
    print("2. Install dependencies: pip install -r requirements.txt")
    print("3. Set up your .env file with GOOGLE_API_KEY")
    print("4. Install Playwright browsers: playwright install chromium")
    sys.exit(1)
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("\n🔧 Make sure:")
    print("1. GOOGLE_API_KEY is set in your .env file")
    print("2. All dependencies are installed")
    sys.exit(1)
