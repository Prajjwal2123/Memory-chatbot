"""
Step 2: Embedding generation & vector storage (Chroma).
"""
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from config import settings
from models import get_embeddings
from data_pipeline.chunker import Chunk
from functools import lru_cache


def get_embedding_model():
    return get_embeddings()


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    """Load (or lazily create) the persistent Chroma collection. Cached so
    we don't reconnect/reload on every single query or upload."""
    return Chroma(
        collection_name=settings.CHROMA_COLLECTION,
        embedding_function=get_embedding_model(),
        persist_directory=settings.CHROMA_PERSIST_DIR,
    )


def index_chunks(chunks: list[Chunk]) -> int:
    """Embed and upsert chunks into the vector store. Returns number indexed."""
    if not chunks:
        return 0

    docs = [
        Document(
            page_content=c.text,
            metadata={"source": c.source, "chunk_id": c.chunk_id},
        )
        for c in chunks
    ]
    ids = [c.chunk_id for c in chunks]

    vs = get_vectorstore()
    vs.add_documents(documents=docs, ids=ids)
    return len(docs)


def similarity_search(query: str, k: int = None) -> list[Document]:
    k = k or settings.RAG_TOP_K
    vs = get_vectorstore()
    return vs.similarity_search(query, k=k)
