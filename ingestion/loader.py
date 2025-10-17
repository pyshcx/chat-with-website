"""Improved website ingestion with better error handling and resource management."""

import logging
import time
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
from urllib.parse import urlparse, urljoin

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from trafilatura import fetch_url, extract, bare_extraction
from trafilatura.settings import use_config
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

from langchain_core.documents import Document
from config import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class URLValidationError(Exception):
    """Raised when URL validation fails."""
    pass

class ContentExtractionError(Exception):
    """Raised when content extraction fails."""
    pass

def validate_url(url: str) -> str:
    """Validate and normalize URL."""
    if not url or not url.strip():
        raise URLValidationError("URL cannot be empty")
    
    url = url.strip()
    
    # Add protocol if missing
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    # Parse and validate URL
    try:
        parsed = urlparse(url)
        if not parsed.netloc:
            raise URLValidationError("Invalid URL: missing domain")
        if parsed.scheme not in ['http', 'https']:
            raise URLValidationError("Invalid URL: only HTTP and HTTPS are supported")
    except Exception as e:
        raise URLValidationError(f"Invalid URL format: {e}")
    
    return url

@contextmanager
def create_session(timeout: int = 30):
    """Create a robust HTTP session with retry strategy."""
    session = requests.Session()
    
    # Configure retry strategy
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    # Set headers to mimic a real browser
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    })
    
    try:
        yield session
    finally:
        session.close()

def fetch_simple_html(url: str, timeout: int = 30) -> Optional[str]:
    """Fetch HTML content using requests with robust error handling."""
    try:
        with create_session(timeout) as session:
            logger.info(f"📡 Fetching {url} with requests (timeout: {timeout}s)")
            
            response = session.get(url, timeout=timeout, allow_redirects=True)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' not in content_type and 'application/xml' not in content_type:
                logger.warning(f"⚠️ Unexpected content type: {content_type}")
            
            logger.info(f"✅ Successfully fetched {len(response.text)} characters")
            return response.text
            
    except requests.exceptions.Timeout:
        logger.warning(f"⏰ Timeout fetching {url}")
        return None
    except requests.exceptions.RequestException as e:
        logger.warning(f"🌐 Request failed for {url}: {e}")
        return None
    except Exception as e:
        logger.warning(f"❌ Unexpected error fetching {url}: {e}")
        return None

def fetch_with_playwright(url: str, timeout: int = 30000) -> Optional[str]:
    """Fetch HTML content using Playwright with proper resource management."""
    try:
        logger.info(f"🎭 Fetching {url} with Playwright (timeout: {timeout}ms)")
        
        with sync_playwright() as playwright:
            # Launch browser with optimized settings
            browser = playwright.chromium.launch(
                headless=True,
                args=[
                    '--no-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-gpu',
                    '--no-first-run',
                    '--disable-extensions'
                ]
            )
            
            try:
                context = browser.new_context(
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                )
                
                page = context.new_page()
                
                # Block unnecessary resources to speed up loading
                page.route("**/*.{png,jpg,jpeg,gif,svg,css,woff,woff2}", lambda route: route.abort())
                
                # Navigate and wait for content
                page.goto(url, timeout=timeout, wait_until='domcontentloaded')
                
                # Wait a bit for dynamic content
                page.wait_for_timeout(2000)
                
                content = page.content()
                logger.info(f"✅ Successfully fetched {len(content)} characters with Playwright")
                
                return content
                
            finally:
                browser.close()
                
    except PlaywrightTimeout:
        logger.warning(f"⏰ Playwright timeout for {url}")
        return None
    except Exception as e:
        logger.warning(f"🎭 Playwright failed for {url}: {e}")
        return None

def extract_content(html: str, url: str) -> Optional[str]:
    """Extract readable content from HTML."""
    if not html:
        return None
    
    try:
        # Primary extraction with trafilatura
        text = extract(html, with_metadata=False, include_comments=False)
        
        if text and len(text.strip()) > 50:  # Ensure we got substantial content
            logger.info(f"✅ Extracted {len(text)} characters with trafilatura")
            return text
        
        # Fallback: bare extraction with metadata
        logger.info("🔄 Trying bare extraction fallback")
        bare_result = bare_extraction(html, with_metadata=True)
        
        if bare_result and 'text' in bare_result:
            fallback_text = bare_result['text']
            if fallback_text and len(fallback_text.strip()) > 50:
                logger.info(f"✅ Extracted {len(fallback_text)} characters with bare extraction")
                return fallback_text
        
        logger.warning(f"⚠️ No substantial content extracted from {url}")
        return None
        
    except Exception as e:
        logger.error(f"❌ Content extraction failed for {url}: {e}")
        return None

def ingest_url(
    url: str,
    use_js: bool = False,
    timeout: int = 60,
    config_path: Optional[str] = None
) -> List[Document]:
    """Ingest a single URL with comprehensive error handling."""
    
    # Validate URL first
    try:
        validated_url = validate_url(url)
        logger.info(f"🎯 Ingesting: {validated_url}")
    except URLValidationError as e:
        logger.error(f"❌ URL validation failed: {e}")
        raise
    
    # Convert timeout to milliseconds for Playwright
    playwright_timeout = timeout * 1000
    
    # Attempt to fetch HTML
    html = None
    fetch_method = "unknown"
    
    if use_js:
        # Try Playwright first for JavaScript-heavy sites
        html = fetch_with_playwright(validated_url, playwright_timeout)
        fetch_method = "playwright"
        
        if html is None:
            logger.info("🔄 Playwright failed, falling back to requests")
            html = fetch_simple_html(validated_url, timeout)
            fetch_method = "requests_fallback"
    else:
        # Try requests first for speed
        html = fetch_simple_html(validated_url, timeout)
        fetch_method = "requests"
        
        if html is None:
            logger.info("🔄 Requests failed, trying Playwright")
            html = fetch_with_playwright(validated_url, playwright_timeout)
            fetch_method = "playwright_fallback"
    
    # Last resort: try trafilatura's own fetching
    if html is None:
        logger.info("🔄 All methods failed, trying trafilatura fetch")
        try:
            html = fetch_url(validated_url)
            fetch_method = "trafilatura"
        except Exception as e:
            logger.warning(f"📡 Trafilatura fetch failed: {e}")
    
    if html is None:
        raise ContentExtractionError(f"Failed to fetch content from {validated_url} using any method")
    
    # Extract readable content
    text = extract_content(html, validated_url)
    
    if not text:
        raise ContentExtractionError(f"No readable content could be extracted from {validated_url}")
    
    # Create document with rich metadata
    metadata = {
        "source": validated_url,
        "fetch_method": fetch_method,
        "content_length": len(text),
        "extraction_timestamp": time.time(),
        "use_js": use_js
    }
    
    # Try to extract additional metadata
    try:
        bare_result = bare_extraction(html, with_metadata=True)
        if bare_result and 'metadata' in bare_result:
            extracted_meta = bare_result['metadata']
            metadata.update({
                "title": extracted_meta.get("title", "Unknown"),
                "author": extracted_meta.get("author"),
                "date": extracted_meta.get("date"),
                "description": extracted_meta.get("description")
            })
    except Exception as e:
        logger.warning(f"⚠️ Failed to extract metadata: {e}")
    
    document = Document(page_content=text, metadata=metadata)
    
    logger.info(f"✅ Successfully created document with {len(text)} characters")
    return [document]
