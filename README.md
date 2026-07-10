# Memory-Augmented Chatbot with Knowledge Graph Fusion

🔗 **Live demo:** https://memory-chatbot-xp26.onrender.com/
> Hosted on Render's free tier — the app sleeps after ~15 minutes of inactivity and can take up to ~50 seconds to wake on the first request.

A conversational AI system that routes each query through the right retrieval strategy — static knowledge (RAG), structured relationships (knowledge graph), or live web/news search — fuses the results, and grounds every answer with visible sources. Built to address the core limitations of plain RAG chatbots: no memory across sessions, no structured reasoning, and no access to real-time information.

---

## What it does

- **Hybrid retrieval, not just RAG.** For knowledge questions, a vector store (ChromaDB) and a Neo4j knowledge graph are queried **in parallel** and fused into one grounded answer — not two competing systems, one combined context.
- **Live information when it matters.** Questions about current events route to a news/web search tool instead of the static knowledge base, with real article URLs shown as sources.
- **Long-term memory.** Preferences and conversation history persist per user across sessions, and inform future answers.
- **Document upload.** Users can upload their own PDF/text files, which get chunked, embedded, and become queryable alongside the scraped knowledge base — including full-file retrieval for "list everything in my document" style questions that plain similarity search tends to miss.
- **A visible safety net.** A hallucination self-check step reviews each grounded answer against its retrieved sources and flags it if the model appears to have stated something unsupported.
- **Related-question suggestions.** After a knowledge or news answer, the system suggests relevant follow-up questions.
- **A dashboard-style UI**, not a generic chat widget — a live pipeline trace showing exactly which nodes fired for each message, plus dedicated panels for sources and suggestions.

---

## Architecture

```
                              START
                                │
                          memory_node          (loads user preferences + history)
                                │
                          router_node           (LLM classifies: rag / tool / direct)
                     ┌──────────┼──────────┐
                 rag_node    tool_node    (direct → skip both)
                 kg_node    (news → web
              (parallel,     fallback)
               fan-in)
                     └──────────┼──────────┘
                                │
                          model_node            (synthesizes grounded answer)
                                │
                        self_check_node          (flags unsupported claims)
                                │
                       suggestions_node          (related follow-up questions)
                                │
                               END
```

Built with **LangGraph**, where `rag_node` and `kg_node` run as true parallel branches and fan into `model_node` once both complete.

---

## Tech stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph, LangChain |
| LLM | Groq (`llama-3.1-8b-instant`) |
| Embeddings | FastEmbed (ONNX-based — chosen over `sentence-transformers`/PyTorch specifically to keep memory usage low enough for free-tier hosting) |
| Vector store | ChromaDB |
| Knowledge graph | Neo4j AuraDB |
| Web scraping | BeautifulSoup |
| Live search | NewsAPI (news queries), DuckDuckGo (general fallback) |
| Backend | FastAPI |
| Memory store | SQLite |
| Frontend | Vanilla HTML/CSS/JS (no framework) |
| Deployment | Render |

---

## Running it locally

```bash
git clone <this-repo-url>
cd memory_chatbot_local_final
python -m venv venv
venv\Scripts\Activate.ps1        # Windows
# source venv/bin/activate       # macOS/Linux

pip install -r requirements.txt
cp .env.example .env             # then fill in your API keys (see below)

python scripts/run_pipeline.py   # scrapes seed URLs, builds the vector store + knowledge graph
python -m uvicorn api.main:app --reload --port 8000
```

Open `http://localhost:8000/`.

### Required environment variables (`.env`)

```
LLM_BACKEND=groq                 # or "local" to use Ollama instead
GROQ_API_KEY=your-key
GROQ_MODEL=llama-3.1-8b-instant

NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password

NEWS_API_KEY=your-key            # free tier: newsapi.org (100 requests/day)

SEED_URLS=https://en.wikipedia.org/wiki/...   # comma-separated
```

---

## Honest evaluation notes

Ran against RAGAS on the included test set:

| Metric | Result |
|---|---|
| Faithfulness | 1.00 |
| Context precision | 1.00 |
| Answer correctness | Not obtained — Groq's free-tier token-per-minute limit was too low for this particular metric's larger prompt payloads |

**What this does and doesn't show:** faithfulness and context precision confirm the RAG pipeline retrieves relevant context and stays grounded in it. It does **not** constitute a measured accuracy percentage across a broad question set — that would require the `answer_correctness` run, which hit a real infrastructure limit rather than completing. In production, this would be resolved with a paid LLM tier or local inference.

### Known limitations
- **Free-tier hosting constraints.** Render's free tier sleeps on inactivity and caps memory at 512MB, which shaped some architecture decisions (e.g. the embedding backend).
- **Memory store resets on redeploy.** SQLite on Render's ephemeral filesystem doesn't persist across restarts — a production version would use a hosted Postgres/Mongo instance.
- **Entity extraction is LLM-based and imperfect.** Some knowledge graph triples are noisy or duplicated (e.g. casing inconsistencies).
- **News/search coverage gaps.** Free-tier news APIs don't index every story, especially regional ones; the system is designed to say "I couldn't find relevant information" in that case rather than force an answer from off-topic results — but this depends on the model correctly recognizing the mismatch.
- **Self-check is a mitigation, not a guarantee.** It catches many but not all unsupported claims, and is itself an LLM call subject to the same class of error it's checking for.

---

## Project structure

```
├── api/                    # FastAPI app, /chat and /upload endpoints
├── data_pipeline/          # scraping, cleaning, chunking, document loading
├── vectorstore/            # Chroma embedding + retrieval
├── knowledge_graph/        # Neo4j entity/relation extraction and querying
├── graph_workflow/         # LangGraph node definitions and graph assembly
├── memory/                 # long-term user memory (SQLite)
├── tools/                  # live news/web search
├── rag/                    # RAG answer generation
├── evaluation/             # RAGAS-based evaluation
├── frontend/               # dashboard UI (vanilla JS)
└── scripts/                # pipeline orchestration
```
