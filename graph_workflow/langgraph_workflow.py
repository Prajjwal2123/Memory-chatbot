r"""
Step 5: LangGraph workflow.

Defines the Dynamic Intelligence Layer described in the problem statement:
a graph with a Model node (routing/reasoning), a Memory node (long-term
personalization), a RAG node (static knowledge), and a Tool node (dynamic
real-time data), wired together with conditional routing.

Graph shape:

        START
          |
      memory_node            (always runs: loads user context)
          |
      router_node             (LLM decides: rag | tool | direct)
       /    |    \
   rag_node tool_node  (direct -> skip both)
       \    |    /
      model_node             (final answer synthesis, writes memory)
          |
         END
"""
from __future__ import annotations
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from config import settings
from models import get_llm
from memory.memory_store import MemoryStore, extract_and_store_preferences
from rag.rag_pipeline import retrieve_context
from tools.dynamic_tools import web_search_tool, format_search_results
from knowledge_graph.kg_builder import KnowledgeGraph

memory_store = MemoryStore()


class ChatState(TypedDict, total=False):
    user_id: str
    message: str
    preferences: dict
    history: list[dict]
    route: Literal["rag", "tool", "direct"]
    context: str
    sources: list[str]
    tool_results: str
    graph_facts: str
    answer: str


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def memory_node(state: ChatState) -> ChatState:
    """Loads long-term preferences + recent history for this user."""
    ctx = memory_store.get_user_context(state["user_id"])
    return {**state, "preferences": ctx["preferences"], "history": ctx["history"]}


def router_node(state: ChatState) -> ChatState:
    """
    Decides whether the query needs static knowledge (RAG), live/dynamic
    data (tool), or can be answered directly from memory/LLM knowledge.
    """
    llm = get_llm(temperature=0)
    prompt = f"""Classify the user's message into exactly one category:
- "rag": needs static/background knowledge that would live in a knowledge base
- "tool": needs current/real-time/live information (news, prices, weather, "latest", "today")
- "direct": small talk, or answerable purely from conversation/user memory, no lookup needed

Message: "{state['message']}"
User preferences known: {state.get('preferences')}

Respond with ONLY one word: rag, tool, or direct."""
    result = llm.invoke(prompt).content.strip().lower()
    route = result if result in ("rag", "tool", "direct") else "rag"
    return {**state, "route": route}


def rag_node(state: ChatState) -> ChatState:
    context, sources = retrieve_context(state["message"])
    return {"context": context, "sources": sources}


def extract_candidate_entities(message: str) -> list[str]:
    """
    Naive entity-name guesser: pulls out capitalized word groups as
    candidates to look up in the knowledge graph. e.g.
    "How does Knowledge Graph relate to RAG?" -> ["Knowledge Graph", "RAG"]
    Good enough as a first pass - swap for an LLM call later for better recall.
    """
    words = message.replace("?", "").replace(",", "").split()
    candidates = []
    current = []
    for w in words:
        if w[:1].isupper():
            current.append(w)
        else:
            if current:
                candidates.append(" ".join(current))
                current = []
    if current:
        candidates.append(" ".join(current))
    # de-duplicate, keep order
    seen = set()
    unique = []
    for c in candidates:
        if c.lower() not in seen:
            seen.add(c.lower())
            unique.append(c)
    return unique[:3]  # cap it - avoid hammering Neo4j on long messages


def kg_node(state: ChatState) -> ChatState:
    """
    Pulls structured (subject, relation, object) facts from Neo4j for any
    entities mentioned in the query, to complement the unstructured RAG
    context with explicit relationships.
    """
    entities = extract_candidate_entities(state["message"])
    if not entities:
        return {"graph_facts": ""}

    facts_lines = []
    with KnowledgeGraph() as kg:
        for entity in entities:
            relations = kg.query_neighbors(entity, depth=1)
            for r in relations[:5]:  # cap per entity to keep prompt short
                facts_lines.append(f"{r['subject']} --{r['relation']}--> {r['object']}")

    graph_facts = "\n".join(facts_lines) if facts_lines else ""
    return {"graph_facts": graph_facts}


def tool_node(state: ChatState) -> ChatState:
    results = web_search_tool(state["message"])
    return {**state, "tool_results": format_search_results(results)}


def model_node(state: ChatState) -> ChatState:
    """Final synthesis: combines memory + (rag context or tool results) into an answer."""
    llm = get_llm(temperature=0.3)

    history_text = "\n".join(f"{h['role']}: {h['content']}" for h in state.get("history", []))
    prefs_text = ", ".join(f"{k}={v}" for k, v in state.get("preferences", {}).items()) or "none known"

    grounding = ""
    if state.get("route") == "rag":
        grounding = f"Static knowledge context:\n{state.get('context', '')}"
        if state.get("graph_facts"):
            grounding += f"\n\nStructured knowledge graph facts:\n{state.get('graph_facts')}"
    elif state.get("route") == "tool":
        grounding = f"Live search results:\n{state.get('tool_results', '')}"

    prompt = f"""You are a personalized, memory-aware assistant.

Known user preferences: {prefs_text}
Recent conversation:
{history_text}

{grounding}

User: {state['message']}

Respond helpfully and naturally. Use the grounding information above when relevant,
and personalize the tone/content using known preferences if appropriate."""

    answer = llm.invoke(prompt).content

    # Persist this turn + any new preferences (long-term memory write-back)
    memory_store.add_turn(state["user_id"], "user", state["message"])
    memory_store.add_turn(state["user_id"], "assistant", answer)
    extract_and_store_preferences(memory_store, state["user_id"], state["message"])

    return {**state, "answer": answer}


# ---------------------------------------------------------------------------
# Routing function (conditional edge)
# ---------------------------------------------------------------------------

def route_decision(state: ChatState) -> str | list[str]:
    """
    Returns the node(s) to run next. For "rag", we fan out to BOTH
    rag_node and kg_node in parallel - LangGraph waits for both to
    finish before running model_node.
    """
    if state["route"] == "rag":
        return ["rag_node", "kg_node"]
    elif state["route"] == "tool":
        return "tool_node"
    else:
        return "model_node"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(ChatState)

    graph.add_node("memory_node", memory_node)
    graph.add_node("router_node", router_node)
    graph.add_node("rag_node", rag_node)
    graph.add_node("kg_node", kg_node)
    graph.add_node("tool_node", tool_node)
    graph.add_node("model_node", model_node)

    graph.add_edge(START, "memory_node")
    graph.add_edge("memory_node", "router_node")

    # Single conditional edge. route_decision can return either one node
    # name or a list of node names (for the "rag" fan-out case above).
    graph.add_conditional_edges(
        "router_node",
        route_decision,
        {
            "rag_node": "rag_node",
            "kg_node": "kg_node",
            "tool_node": "tool_node",
            "model_node": "model_node",
        },
    )

    graph.add_edge("rag_node", "model_node")
    graph.add_edge("kg_node", "model_node")
    graph.add_edge("tool_node", "model_node")
    graph.add_edge("model_node", END)

    return graph.compile()


# Compiled once, reused across requests
chatbot_graph = build_graph()


def run_chat(user_id: str, message: str) -> ChatState:
    initial_state: ChatState = {"user_id": user_id, "message": message}
    return chatbot_graph.invoke(initial_state)
