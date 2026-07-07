"""
Single place that decides which LLM / embedding backend to use.
Every other module calls get_llm() / get_embeddings() instead of
instantiating a provider directly, so switching backends is a
one-line config change.
"""
from functools import lru_cache
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


@lru_cache(maxsize=1)
def get_embeddings():
    """
    Cached, ONNX-based embeddings (fastembed) instead of sentence-transformers.
    sentence-transformers pulls in PyTorch, which is heavy enough on its own
    to push memory usage past free-tier hosting limits (512MB on Render) even
    at idle. fastembed uses ONNX Runtime instead - no PyTorch, much lighter.
    """
    from langchain_community.embeddings import FastEmbedEmbeddings
    return FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")