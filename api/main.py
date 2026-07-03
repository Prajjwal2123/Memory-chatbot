"""
FastAPI service exposing the memory-augmented chatbot.

Run with:
    uvicorn api.main:app --reload --port 8000
"""
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
