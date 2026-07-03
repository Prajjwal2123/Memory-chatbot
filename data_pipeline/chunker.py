"""
Step 1 (part 3): Text chunking.

Uses LangChain's RecursiveCharacterTextSplitter so chunk boundaries
respect paragraph/sentence structure where possible.
"""
from dataclasses import dataclass
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import settings


@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: str


def chunk_documents(documents: dict[str, str]) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Chunk] = []
    for source, text in documents.items():
        pieces = splitter.split_text(text)
        for i, piece in enumerate(pieces):
            chunks.append(
                Chunk(text=piece, source=source, chunk_id=f"{source}::chunk_{i}")
            )
    return chunks
