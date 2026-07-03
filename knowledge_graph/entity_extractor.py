"""
Step 3 (part 1): Entity & relationship extraction.

Uses the LLM to pull a structured (subject, relation, object) triple list
out of each chunk of text, which is then loaded into Neo4j by kg_builder.py.
"""
import json
from config import settings
from models import get_llm

EXTRACTION_PROMPT = """You are an information extraction system.
Read the text below and extract factual (subject, relation, object) triples
that capture entities and the relationships between them.

Rules:
- Only extract facts explicitly stated or strongly implied in the text.
- Keep entity names short and consistent (e.g. "Knowledge Graph", not "a knowledge graph").
- relation should be a short verb phrase, e.g. "is a type of", "uses", "was developed by".
- Return STRICT JSON: a list of objects with keys "subject", "relation", "object".
- Return [] if no clear triples exist. Do not include any text outside the JSON.

Text:
\"\"\"
{text}
\"\"\"
"""


def extract_triples(text: str) -> list[dict]:
    llm = get_llm(temperature=0)
    prompt = EXTRACTION_PROMPT.format(text=text[:4000])  # guard against huge chunks
    response = llm.invoke(prompt)
    raw = response.content.strip()

    # Models sometimes wrap JSON in markdown fences; strip those defensively.
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.split("\n", 1)[-1] if raw.lower().startswith("json") else raw

    try:
        triples = json.loads(raw)
        if isinstance(triples, list):
            return [t for t in triples if {"subject", "relation", "object"} <= t.keys()]
        return []
    except json.JSONDecodeError:
        print("[entity_extractor] Could not parse LLM output as JSON, skipping chunk.")
        return []
