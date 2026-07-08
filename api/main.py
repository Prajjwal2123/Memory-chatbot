"""
FastAPI service exposing the memory-augmented chatbot.

Run with:
    uvicorn api.main:app --reload --port 8000
"""
from fastapi import UploadFile, File
from data_pipeline.document_loader import extract_text_from_upload
from data_pipeline.cleaner import clean_text
from data_pipeline.chunker import chunk_documents
from vectorstore.embed_store import index_chunks

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
from graph_workflow.langgraph_workflow import run_chat
from memory.memory_store import MemoryStore
from knowledge_graph.kg_builder import KnowledgeGraph

app = FastAPI(title="Memory-Augmented Chatbot API")
memory_store = MemoryStore()

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")


@app.get("/")
def serve_ui():
    """Serves the chat UI at the root URL, e.g. http://localhost:8000/"""
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


class ChatRequest(BaseModel):
    user_id: str
    message: str


class ChatResponse(BaseModel):
    answer: str
    route: str
    sources: list[str] = []


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    result = run_chat(req.user_id, req.message)
    return ChatResponse(
        answer=result.get("answer", ""),
        route=result.get("route", ""),
        sources=result.get("sources", []),
    )


@app.get("/memory/{user_id}")
def get_memory(user_id: str):
    """Inspect what the system currently remembers about a user."""
    return memory_store.get_user_context(user_id)


@app.get("/kg/{entity}")
def query_kg(entity: str, depth: int = 1):
    """Inspect the knowledge graph neighborhood of an entity."""
    with KnowledgeGraph() as kg:
        return {"entity": entity, "relations": kg.query_neighbors(entity, depth=depth)}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Accepts a PDF or text file, extracts its text, cleans and chunks it,
    and indexes it into the same Chroma vector store used for RAG answers.
    Uploaded content becomes queryable immediately - no pipeline restart needed.
    """
    file_bytes = await file.read()
    try:
        raw_text = extract_text_from_upload(file.filename, file_bytes)
    except ValueError as e:
        return {"error": str(e)}

    if not raw_text.strip():
        return {"error": "No extractable text found in the file."}

    cleaned = clean_text(raw_text)
    # Larger chunks for uploaded documents (resumes, short docs) so distinct
    # sections like "Projects" or "Experience" are less likely to get split
    # across multiple chunks and missed by retrieval.
    chunks = chunk_documents({file.filename: cleaned}, chunk_size=1500, chunk_overlap=200)

    # Remove any previously-indexed chunks from this same filename first,
    # so re-uploading the same file doesn't create duplicate, noisy copies
    from vectorstore.embed_store import get_vectorstore
    vs = get_vectorstore()
    try:
        vs.delete(where={"source": file.filename})
    except Exception:
        pass  # nothing to delete yet, that's fine

    n_indexed = index_chunks(chunks)

    return {
        "filename": file.filename,
        "chunks_indexed": n_indexed,
        "message": f"'{file.filename}' is now searchable - ask me about it!",
    }