"""
Orchestrates Steps 1-3 of the methodology:
  1. Scrape -> clean -> chunk
  2. Embed -> store in Chroma
  3. Extract entities/relations -> store in Neo4j

Run from the project root:
    python scripts/run_pipeline.py
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import settings
from data_pipeline.scraper import scrape_urls, save_raw
from data_pipeline.cleaner import clean_documents
from data_pipeline.chunker import chunk_documents
from vectorstore.embed_store import index_chunks
from knowledge_graph.kg_builder import build_graph_from_chunks


def main():
    print("=== Step 1: Scrape -> Clean -> Chunk ===")
    if not settings.SEED_URLS:
        print("No SEED_URLS configured in .env — add comma-separated URLs and re-run.")
        return

    raw_docs = scrape_urls(settings.SEED_URLS)
    save_raw(raw_docs)
    cleaned_docs = clean_documents(raw_docs)
    chunks = chunk_documents(cleaned_docs)
    print(f"Produced {len(chunks)} chunks from {len(cleaned_docs)} documents.")

    print("\n=== Step 2: Embed -> Store in Chroma ===")
    n_indexed = index_chunks(chunks)
    print(f"Indexed {n_indexed} chunks into Chroma at {settings.CHROMA_PERSIST_DIR}.")

    print("\n=== Step 3: Extract entities/relations -> Store in Neo4j ===")
    n_triples = build_graph_from_chunks(chunks)
    print(f"Loaded {n_triples} triples into Neo4j.")

    print("\nPipeline complete. Start the API with:\n  uvicorn api.main:app --reload --port 8000")


if __name__ == "__main__":
    main()
