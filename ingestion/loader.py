# ingestion/loader.py

import logging
from typing import Optional, List, Dict, Any
import requests
from trafilatura import fetch_url, extract
from trafilatura.settings import use_config
from trafilatura import bare_extraction
from playwright.sync_api import sync_playwright

from langchain.schema import Document  # or your own Document class

logger = logging.getLogger(__name__)


def fetch_simple_html(url: str, timeout: int = 10) -> Optional[str]:
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        logger.warning(f"fetch_simple_html failed for {url}: {e}")
        return None


def fetch_with_playwright(url: str, timeout: int = 30_000) -> Optional[str]:
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=timeout)
            content = page.content()
            browser.close()
            return content
    except Exception as e:
        logger.warning(f"fetch_with_playwright failed for {url}: {e}")
        return None


def ingest_url(
    url: str,
    use_js: bool = False,
    config_path: Optional[str] = None
) -> List[Document]:
    """
    Ingest a single URL. Returns 0 or more Document objects.
    """
    html: Optional[str]
    if use_js:
        html = fetch_with_playwright(url)
    else:
        html = fetch_simple_html(url)
    if html is None:
        # fallback: try fetch via trafilatura's fetch_url
        fetched = fetch_url(url)
        if fetched is None:
            logger.error(f"Failed to fetch URL {url} by any method.")
            return []
        html = fetched

    # Use trafilatura to extract
    # Optionally load custom config
    if config_path:
        cfg = use_config(config_path)
        text = extract(html, config=cfg, with_metadata=False)
    else:
        text = extract(html, with_metadata=False)

    if not text:
        # fallback: try bare_extraction to get metadata + text
        try:
            bare = bare_extraction(html, with_metadata=True)
            fetched_text = bare.get("text", "")
            metadata = bare.get("metadata", {})
            if fetched_text:
                doc = Document(page_content=fetched_text, metadata={**metadata, "source": url})
                return [doc]
        except Exception as e:
            logger.warning(f"bare_extraction fallback failed for {url}: {e}")
        # still no text
        return []

    metadata: Dict[str, Any] = {"source": url}
    # (Could also extract metadata like title, date, etc., in future)
    doc = Document(page_content=text, metadata=metadata)
    return [doc]
