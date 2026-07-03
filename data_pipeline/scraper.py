"""
Step 1 (part 1): Web scraping with BeautifulSoup.

Fetches raw HTML for each seed URL and extracts visible text.
For larger/JS-heavy sites you can swap this for Scrapy or Playwright,
the rest of the pipeline only depends on `scrape_urls()` returning
{url: text} pairs.
"""
import os
import requests
from bs4 import BeautifulSoup
from config import settings

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; MemoryChatbotBot/1.0; +https://example.com/bot)"
}


def fetch_html(url: str, timeout: int = 15) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def extract_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # Drop non-content tags
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "form"]):
        tag.decompose()

    # Prefer <article> or <main> if present, else full body
    container = soup.find("article") or soup.find("main") or soup.body or soup
    text = container.get_text(separator="\n")
    return text


def scrape_urls(urls: list[str]) -> dict[str, str]:
    """Scrape a list of URLs, returning {url: raw_text}. Skips failures gracefully."""
    results = {}
    for url in urls:
        try:
            html = fetch_html(url)
            text = extract_text(html)
            results[url] = text
            print(f"[scraper] OK   {url}  ({len(text)} chars)")
        except Exception as e:
            print(f"[scraper] FAIL {url}  -> {e}")
    return results


def save_raw(results: dict[str, str], out_dir: str = None) -> list[str]:
    """Persist scraped text to data/raw/ as .txt files. Returns list of file paths."""
    out_dir = out_dir or settings.RAW_DATA_DIR
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i, (url, text) in enumerate(results.items()):
        path = os.path.join(out_dir, f"doc_{i:03d}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"SOURCE_URL: {url}\n\n{text}")
        paths.append(path)
    return paths


if __name__ == "__main__":
    urls = settings.SEED_URLS
    if not urls:
        print("No SEED_URLS configured in .env")
    else:
        scraped = scrape_urls(urls)
        files = save_raw(scraped)
        print(f"Saved {len(files)} files to {settings.RAW_DATA_DIR}")
