"""
Step 4: RAG pipeline - query embedding -> similarity search -> context
retrieval -> LLM answer generation.
"""
from config import settings
from models import get_llm
from vectorstore.embed_store import similarity_search

ANSWER_PROMPT = """You are a precise, careful assistant. Answer using ONLY
the context provided below.

Strict rules:
1. Every factual claim in your answer must be directly traceable to a
   specific sentence in the context. If you can't point to where a fact
   came from, do not state it.
2. Do NOT assume two facts are related just because they appear near each
   other in the context. Only connect facts if the text explicitly states
   they are connected.
3. If the context is insufficient, partial, or ambiguous, say exactly what
   is missing rather than filling gaps with assumptions.
4. Do not add outside knowledge, even if you're confident it's true. Stay
   strictly inside what the context says.
5. Prefer being brief and exact over being complete. A shorter, fully
   accurate answer is better than a longer one with unsupported additions.

Context:
{context}

Question: {question}

Answer:"""


def retrieve_context(query: str, k: int = None, last_uploaded_file: str = None) -> tuple[str, list[str]]:
    docs = similarity_search(query, k=k)
    context_parts = [d.page_content for d in docs]
    sources = {d.metadata.get("source", "") for d in docs}

    # For enumeration-style questions about an uploaded document ("what
    # projects", "list", "everything in my resume"), pull ALL chunks from
    # that file directly instead of relying on top-k similarity, which can
    # silently miss sections that score lower but are still relevant.
    doc_keywords = ("resume", "document", "file", "uploaded", "cv", "pdf")
    if last_uploaded_file and any(kw in query.lower() for kw in doc_keywords):
        from vectorstore.embed_store import get_all_chunks_from_source
        all_chunks = get_all_chunks_from_source(last_uploaded_file)
        context_parts.extend(all_chunks)
        sources.add(last_uploaded_file)

    context = "\n\n".join(dict.fromkeys(context_parts))  # dedupe, preserve order
    return context, list(sources)


def generate_rag_answer(query: str, k: int = None) -> dict:
    context, sources = retrieve_context(query, k=k)
    llm = get_llm(temperature=0.0)
    prompt = ANSWER_PROMPT.format(context=context or "No relevant context found.", question=query)
    response = llm.invoke(prompt)
    return {
        "answer": response.content,
        "context": context,
        "sources": sources,
    }
