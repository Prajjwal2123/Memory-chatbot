"""
Loads text out of user-uploaded documents (PDF or plain text), so they can
be fed into the same clean -> chunk -> embed pipeline used for scraped
web pages. This lets the chatbot answer questions over documents the user
uploads directly, in addition to its scraped knowledge base.
"""
import io
from pypdf import PdfReader


def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return "\n\n".join(pages)


def extract_text_from_upload(filename: str, file_bytes: bytes) -> str:
    """Dispatch based on file extension. Add more formats here as needed."""
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    elif lower.endswith(".txt") or lower.endswith(".md"):
        return file_bytes.decode("utf-8", errors="ignore")
    else:
        raise ValueError(f"Unsupported file type: {filename}. Use .pdf, .txt, or .md")