# scripts/test_ingest.py

from ingestion.loader import ingest_url

def main():
    url = "https://www.indiatoday.in/"
    docs = ingest_url(url, use_js=False)
    print(f"Ingested {len(docs)} documents from {url}")
    for i, doc in enumerate(docs):
        print(f"--- Doc #{i} ---")
        content = doc.page_content
        meta = doc.metadata
        snippet = content[:500].replace("\n", " ")
        print("Snippet:", snippet)
        print("Metadata:", meta)

if __name__ == "__main__":
    main()
