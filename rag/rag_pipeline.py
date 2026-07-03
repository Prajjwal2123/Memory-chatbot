"""
Step 4: RAG pipeline - query embedding -> similarity search -> context
retrieval -> LLM answer generation.
"""
from config import settings
from models import get_llm
from vectorstore.embed_store import similarity_search

ANSWER_PROMPT = """You are a helpful assistant. Answer the user's question using
ONLY the context provided below. If the context does not contain the answer,
say you don't have enough information rather than guessing.

Context:
{context}

Question: {question}

Answer:"""


def retrieve_context(query: str, k: int = None) -> tuple[str, list[str]]:
    docs = similarity_search(query, k=k)
    context = "\n\n".join(d.page_content for d in docs)
    sources = list({d.metadata.get("source", "") for d in docs})
    return context, sources


def generate_rag_answer(query: str, k: int = None) -> dict:
    context, sources = retrieve_context(query, k=k)
    llm = get_llm(temperature=0.2)
    prompt = ANSWER_PROMPT.format(context=context or "No relevant context found.", question=query)
    response = llm.invoke(prompt)
    return {
        "answer": response.content,
        "context": context,
        "sources": sources,
    }
