"""
Dynamic Intelligence Layer - tools for live/real-time data, used by the
LangGraph "tool node" when a query needs information that isn't in the
static knowledge base (e.g. "what's the latest news on X").
"""
from duckduckgo_search import DDGS


def web_search_tool(query: str, max_results: int = 5) -> list[dict]:
    """Live web search - used for queries needing up-to-date information."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return [{"title": r.get("title"), "snippet": r.get("body"), "url": r.get("href")} for r in results]
    except Exception as e:
        print(f"[web_search_tool] Search failed: {e}")
        return []

import requests
from config import settings


def news_search_tool(query: str, max_results: int = 5) -> list[dict]:
    """
    Fetches real, structured, dated news articles via NewsAPI - more
    reliable than scraping a search engine, and gives proper source/date
    attribution. Falls back to an empty list on any failure so the caller
    can fall back to general web search instead.
    """
    if not settings.NEWS_API_KEY:
        return []
    try:
        resp = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": query,
                "sortBy": "publishedAt",
                "language": "en",
                "pageSize": max_results,
                "apiKey": settings.NEWS_API_KEY,
            },
            timeout=10,
        )
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
        return [
            {
                "title": a.get("title"),
                "snippet": a.get("description"),
                "url": a.get("url"),
                "source": a.get("source", {}).get("name"),
                "published": a.get("publishedAt"),
            }
            for a in articles
        ]
    except Exception as e:
        print(f"[news_search_tool] Failed: {e}")
        return []


def format_news_results(results: list[dict]) -> str:
    if not results:
        return ""
    lines = []
    for r in results:
        date = (r.get("published") or "")[:10]
        lines.append(f"- [{date}] {r['title']} ({r.get('source', 'unknown source')}): {r['snippet']} ({r['url']})")
    return "\n".join(lines)

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
