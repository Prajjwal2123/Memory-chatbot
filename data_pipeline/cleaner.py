"""
Step 1 (part 2): Data cleaning and preprocessing.
"""
import re


def clean_text(text: str) -> str:
    """Normalize whitespace, strip references/citation markers, drop empty lines."""
    # Remove Wikipedia-style citation markers like [12], [citation needed]
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\[citation needed\]", "", text, flags=re.IGNORECASE)

    # Collapse multiple blank lines
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]

    cleaned = "\n".join(lines)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


def clean_documents(raw_docs: dict[str, str]) -> dict[str, str]:
    return {url: clean_text(text) for url, text in raw_docs.items()}
