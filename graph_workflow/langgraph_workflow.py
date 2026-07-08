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
      self_check_node        (flags unsupported/hallucinated answers)
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
- "rag": needs static/background knowledge that would live in a knowledge base, OR references an
  uploaded document/file/resume in any way (e.g. "what does my resume say", "tell me about the
  document I uploaded", "did you get my file", "what projects are in there") - ALWAYS classify
  these as "rag", even if phrased casually or conversationally
- "tool": needs current/real-time/live information (news, prices, weather, "latest", "today")
- "direct": small talk with no reference to documents, files, or lookups needed

Message: "{state['message']}"
User preferences known: {state.get('preferences')}

Respond with ONLY one word: rag, tool, or direct."""
    result = llm.invoke(prompt).content.strip().lower()
    route = result if result in ("rag", "tool", "direct") else "rag"
    return {**state, "route": route}


def rag_node(state: ChatState) -> ChatState:
    last_uploaded_file = state.get("preferences", {}).get("last_uploaded_file")
    context, sources = retrieve_context(state["message"], last_uploaded_file=last_uploaded_file)
    return {"context": context, "sources": sources}


def extract_candidate_entities(message: str) -> list[str]:
    """
    Asks the LLM which entity name(s) to look up in the knowledge graph
    for this message. More robust than capitalization-based guessing -
    catches lowercase mentions like "what is a knowledge graph?".
    """
    llm = get_llm(temperature=0)
    prompt = f"""Extract the main entity or entities (people, concepts, technologies,
organizations) this question is asking about, so they can be looked up in a
knowledge graph. Return ONLY a comma-separated list of short entity names,
using title case (e.g. "Knowledge Graph, Neo4j"). If there are no clear
entities, return NONE.

Question: "{message}"

Entities:"""
    result = llm.invoke(prompt).content.strip()
    if result.upper() == "NONE" or not result:
        return []
    entities = [e.strip() for e in result.split(",") if e.strip()]
    return entities[:3]


def kg_node(state: ChatState) -> ChatState:
    """
    Pulls structured (subject, relation, object) facts from Neo4j for any
    entities mentioned in the query, to complement the unstructured RAG
    context with explicit relationships. Fails gracefully if Neo4j is
    unreachable, so the chatbot still answers using RAG alone instead of
    crashing the whole request.
    """
    entities = extract_candidate_entities(state["message"])
    if not entities:
        return {"graph_facts": ""}

    facts_lines = []
    try:
        with KnowledgeGraph() as kg:
            for entity in entities:
                relations = kg.query_neighbors(entity, depth=1)
                for r in relations[:5]:
                    facts_lines.append(f"{r['subject']} --{r['relation']}--> {r['object']}")
    except Exception as e:
        print(f"[kg_node] Neo4j unavailable, falling back to RAG-only: {e}")
        return {"graph_facts": ""}

    graph_facts = "\n".join(facts_lines) if facts_lines else ""
    return {"graph_facts": graph_facts}


def tool_node(state: ChatState) -> ChatState:
    results = web_search_tool(state["message"])
    return {**state, "tool_results": format_search_results(results)}


def model_node(state: ChatState) -> ChatState:
    """Final synthesis: combines memory + (rag context or tool results) into an answer."""
    llm = get_llm(temperature=0.1)

    history_text = "\n".join(f"{h['role']}: {h['content']}" for h in state.get("history", []))
    prefs_text = ", ".join(f"{k}={v}" for k, v in state.get("preferences", {}).items()) or "none known"

    grounding = ""
    if state.get("route") == "rag":
        grounding = f"Static knowledge context:\n{state.get('context', '')}"
        if state.get("graph_facts"):
            grounding += f"\n\nStructured knowledge graph facts:\n{state.get('graph_facts')}"
    elif state.get("route") == "tool":
        grounding = f"Live search results:\n{state.get('tool_results', '')}"

    prompt = f"""You are a precise, memory-aware assistant. Follow these rules strictly:
- Every factual claim must be traceable to the grounding information below, the conversation history, or explicitly known user preferences. Do not state anything you cannot trace to one of these sources.
- Do not connect two facts into one narrative unless the source material explicitly connects them.
- If the grounding information doesn't answer the question, say so plainly rather than guessing or filling gaps.
- Be concise and exact rather than exhaustive.

Known user preferences: {prefs_text}
Recent conversation:
{history_text}

{grounding}

User: {state['message']}

Respond using only the rules above."""

    answer = llm.invoke(prompt).content

    # Persist this turn + any new preferences (long-term memory write-back)
    memory_store.add_turn(state["user_id"], "user", state["message"])
    memory_store.add_turn(state["user_id"], "assistant", answer)
    extract_and_store_preferences(memory_store, state["user_id"], state["message"])

    return {**state, "answer": answer}


def self_check_node(state: ChatState) -> ChatState:
    """
    Hallucination check: verifies the generated answer is actually supported
    by the retrieved context/graph facts, rather than trusting the model's
    output blindly. Appends a visible caveat if it isn't well-grounded.
    """
    current_answer = state.get("answer", "")

    if state.get("route") not in ("rag", "tool"):
        return {"answer": current_answer}  # nothing to check for direct/small-talk answers

    grounding_text = (
        state.get("context", "") + "\n" + state.get("graph_facts", "") + "\n" + state.get("tool_results", "")
    ).strip()
    if not grounding_text:
        return {"answer": current_answer}

    llm = get_llm(temperature=0.0)
    check_prompt = f"""Does the ANSWER below rely only on facts present in the SOURCE, or
does it include claims not supported by the SOURCE?

SOURCE:
{grounding_text}

ANSWER:
{current_answer}

Respond with ONLY one word: SUPPORTED or UNSUPPORTED."""

    verdict = llm.invoke(check_prompt).content.strip().upper()

    if "UNSUPPORTED" in verdict:
        flagged_answer = (
            current_answer
            + "\n\n⚠️ *Note: part of this answer may not be fully supported by the retrieved sources - verify independently.*"
        )
        return {"answer": flagged_answer}

    return {"answer": current_answer}


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
    graph.add_node("self_check_node", self_check_node)

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
    graph.add_edge("model_node", "self_check_node")
    graph.add_edge("self_check_node", END)

    return graph.compile()


# Compiled once, reused across requests
chatbot_graph = build_graph()


def run_chat(user_id: str, message: str) -> ChatState:
    initial_state: ChatState = {"user_id": user_id, "message": message}
    return chatbot_graph.invoke(initial_state)