"""
Single place that decides which LLM / embedding backend to use.
Every other module calls get_llm() / get_embeddings() instead of
instantiating ChatOpenAI/OpenAIEmbeddings directly, so switching between
local (free, Ollama + HuggingFace) and OpenAI is a one-line config change.
"""
from config import settings


def get_llm(temperature: float = 0.2):
    if settings.USE_LOCAL_MODELS:
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=settings.LOCAL_LLM_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
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
    if settings.USE_LOCAL_MODELS:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=settings.LOCAL_EMBEDDING_MODEL)
    else:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=settings.EMBEDDING_MODEL, api_key=settings.OPENAI_API_KEY)
