"""
Single place that decides which LLM / embedding backend to use.
Every other module calls get_llm() / get_embeddings() instead of
instantiating a provider directly, so switching backends is a
one-line config change.

Backends:
- "local" : Ollama (free, runs on your machine) - good for local dev
- "groq"  : Groq's free hosted API (fast, free tier, no card needed) - good for deployment
- "openai": OpenAI (paid) - fallback if you ever want it
"""
from config import settings


def get_llm(temperature: float = 0.2):
    backend = settings.LLM_BACKEND

    if backend == "local":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=settings.LOCAL_LLM_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=temperature,
        )
    elif backend == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=settings.GROQ_MODEL,
            api_key=settings.GROQ_API_KEY,
            temperature=temperature,
        )
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.LLM_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=temperature,
        )


def get_embeddings():
    # Embeddings always run locally via HuggingFace - free, no API needed,
    # works the same whether you're on your machine or deployed.
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=settings.LOCAL_EMBEDDING_MODEL)