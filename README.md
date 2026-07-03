# Memory-Augmented Chatbot with Knowledge Graph and Hybrid RAG

End-to-end implementation of the project brief:
- Static Knowledge Layer (RAG over scraped web data, Chroma vector store)
- Knowledge Graph Layer (entity/relationship extraction → Neo4j)
- Dynamic Intelligence Layer (LangGraph router: RAG node / Tool node / Memory node / Model node)
- Long-term memory (per-user preferences + history, SQLite by default, MongoDB-ready)
- Evaluation framework (context relevance, answer correctness, faithfulness via RAGAS)
- FastAPI server exposing the chatbot as an API

---

## 0. Prerequisites

- Python 3.10+
- VS Code with the **Python** extension installed
- A Neo4j instance (free options: [Neo4j Desktop](https://neo4j.com/download/) or [Neo4j AuraDB Free](https://neo4j.com/cloud/aura-free/))
- **[Ollama](https://ollama.com/download)** installed — this project runs entirely on free local models by default (no OpenAI key or billing needed). If you'd rather use OpenAI, set `USE_LOCAL_MODELS=false` in `.env` and supply an `OPENAI_API_KEY`.

### Setting up Ollama (one-time)

1. Install Ollama from https://ollama.com/download
2. Pull the model used by default:
   ```bash
   ollama pull llama3.1
   ```
   (This downloads a few GB — grab a coffee. If your machine is low on RAM, use a smaller model like `ollama pull llama3.2:3b` and set `LOCAL_LLM_MODEL=llama3.2:3b` in `.env` instead.)
3. Ollama runs a local server automatically after install. Verify it's up:
   ```bash
   ollama list
   ```
   You should see `llama3.1` listed. Leave Ollama running in the background — the app talks to it over `http://localhost:11434`.

Embeddings use `sentence-transformers/all-MiniLM-L6-v2` from HuggingFace, which downloads automatically the first time you run the pipeline (also free, runs on CPU).

---

## 1. Open the project in VS Code

```bash
cd memory_chatbot
code .
```

Select/create a Python interpreter: `Ctrl+Shift+P` → "Python: Select Interpreter".

## 2. Create a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Configure environment variables

Copy `.env.example` to `.env` and fill in your Neo4j credentials (models are already configured to run locally via Ollama, no key needed):

```bash
cp .env.example .env
```

```
USE_LOCAL_MODELS=true
NEO4J_URI=neo4j+s://xxxx.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password
SEED_URLS=https://en.wikipedia.org/wiki/Retrieval-augmented_generation,https://en.wikipedia.org/wiki/Knowledge_graph
```

## 5. Run the pipeline end-to-end (Steps 1–3 of the methodology)

This scrapes the seed URLs, cleans + chunks the text, builds embeddings into Chroma,
and extracts entities/relationships into Neo4j.

```bash
python scripts/run_pipeline.py
```

## 6. Start the chatbot API (Steps 4–6: RAG + LangGraph routing + tools)

```bash
uvicorn api.main:app --reload --port 8000
```

Test it:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_123", "message": "What is a knowledge graph?"}'
```

Interactive docs: http://localhost:8000/docs

## 7. Run evaluation (Step 7)

```bash
python evaluation/evaluate.py
```

This prints faithfulness, answer-correctness, and context-relevance scores for a
small held-out set of test questions defined in `evaluation/test_questions.json`.

---

## Project layout

```
memory_chatbot/
├── config.py                  # central settings (env vars, model names, paths)
├── data_pipeline/
│   ├── scraper.py             # BeautifulSoup web scraper
│   ├── cleaner.py             # text cleaning
│   └── chunker.py             # text chunking
├── vectorstore/
│   └── embed_store.py         # embeddings + Chroma vector store
├── knowledge_graph/
│   ├── entity_extractor.py    # LLM-based entity/relation extraction
│   └── kg_builder.py          # Neo4j ingestion + query helpers
├── memory/
│   └── memory_store.py        # long-term user memory (SQLite, Mongo-ready)
├── tools/
│   └── dynamic_tools.py       # live/dynamic data tools (web search, weather, etc.)
├── rag/
│   └── rag_pipeline.py        # retrieval + answer generation
├── graph_workflow/
│   └── langgraph_workflow.py  # LangGraph: model/memory/RAG/tool nodes + router
├── evaluation/
│   ├── evaluate.py            # RAGAS-based evaluation
│   └── test_questions.json    # held-out eval set
├── api/
│   └── main.py                # FastAPI app exposing /chat
└── scripts/
    └── run_pipeline.py        # orchestrates steps 1-3
```

## How each requirement maps to code

| Requirement (from problem statement) | Where it's implemented |
|---|---|
| Lack of long-term memory | `memory/memory_store.py` |
| Limited reasoning over structured data | `knowledge_graph/` (Neo4j) |
| Static + real-time data | `rag/` (static) + `tools/` (real-time) |
| RAG for static knowledge | `vectorstore/embed_store.py`, `rag/rag_pipeline.py` |
| Knowledge Graph for structured reasoning | `knowledge_graph/kg_builder.py` |
| Long-term memory for personalization | `memory/memory_store.py` |
| Tool-based dynamic retrieval (LangGraph) | `graph_workflow/langgraph_workflow.py`, `tools/dynamic_tools.py` |
| Web scraping → cleaning → chunking | `data_pipeline/` |
| Embedding generation + vector storage | `vectorstore/embed_store.py` |
| Entity/relationship extraction | `knowledge_graph/entity_extractor.py` |
| LangGraph nodes: Model/Memory/RAG/Tool + routing | `graph_workflow/langgraph_workflow.py` |
| Evaluation: relevance, correctness, faithfulness | `evaluation/evaluate.py` |
| FastAPI service | `api/main.py` |
