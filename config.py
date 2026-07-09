"""
Central configuration for the Memory-Augmented Chatbot project.
All other modules import settings from here instead of reading
environment variables directly.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # --- LLM ---
    # Set USE_LOCAL_MODELS=true to run entirely on free local models via Ollama
    # (no OpenAI key/billing needed). Requires Ollama installed and running.
    # "local" (Ollama, free, your machine) | "groq" (free hosted API) | "openai" (paid)
    LLM_BACKEND: str = os.getenv("LLM_BACKEND", "local")

    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    # --- Local model settings (used when USE_LOCAL_MODELS=true) ---
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    LOCAL_LLM_MODEL: str = os.getenv("LOCAL_LLM_MODEL", "llama3.1")
    LOCAL_EMBEDDING_MODEL: str = os.getenv(
        "LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    # --- Groq (used when LLM_BACKEND=groq) ---
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")
    # --- Neo4j ---
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "password")

    # --- Data pipeline ---
    SEED_URLS: list[str] = [
        u.strip() for u in os.getenv("SEED_URLS", "").split(",") if u.strip()
    ]
    RAW_DATA_DIR: str = os.getenv("RAW_DATA_DIR", "./data/raw")
    PROCESSED_DATA_DIR: str = os.getenv("PROCESSED_DATA_DIR", "./data/processed")

    # --- Chunking ---
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 800))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 120))

    # --- Vector store ---
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
    CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "static_knowledge")
    RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", 6))

    # --- Memory ---
    MEMORY_DB_PATH: str = os.getenv("MEMORY_DB_PATH", "./data/memory.db")
    MEMORY_HISTORY_WINDOW: int = int(os.getenv("MEMORY_HISTORY_WINDOW", 10))


settings = Settings()
