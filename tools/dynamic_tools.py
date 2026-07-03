"""
Dynamic Intelligence Layer - tools for live/real-time data, used by the
LangGraph "tool node" when a query needs information that isn't in the
static knowledge base (e.g. "what's the latest news on X").
"""
from duckduckgo_search import DDGS


def web_search_tool(query: str, max_results: int = 5) -> list[dict]:
    """Live web search - used for queries needing up-to-date information."""
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))
    return [{"title": r.get("title"), "snippet": r.get("body"), "url": r.get("href")} for r in results]


def format_search_results(results: list[dict]) -> str:
    if not results:
        return "No live results found."
    lines = []
    for r in results:
        lines.append(f"- {r['title']}: {r['snippet']} ({r['url']})")
    return "\n".join(lines)


# Registry so new tools can be added without touching the LangGraph workflow code.
TOOL_REGISTRY = {
    "web_search": web_search_tool,
}
