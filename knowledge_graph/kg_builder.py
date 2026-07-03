"""
Step 3 (part 2): Knowledge graph storage & querying (Neo4j).
"""
from neo4j import GraphDatabase
from config import settings
from data_pipeline.chunker import Chunk
from knowledge_graph.entity_extractor import extract_triples


class KnowledgeGraph:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
        )

    def close(self):
        self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def add_triple(self, subject: str, relation: str, obj: str, source: str = None):
        """MERGE keeps the graph idempotent: re-running ingestion won't duplicate nodes."""
        query = """
        MERGE (s:Entity {name: $subject})
        MERGE (o:Entity {name: $object})
        MERGE (s)-[r:RELATION {type: $relation}]->(o)
        SET r.source = $source
        """
        with self.driver.session() as session:
            session.run(query, subject=subject, object=obj, relation=relation, source=source)

    def query_neighbors(self, entity: str, depth: int = 1) -> list[dict]:
        """Return relationships within `depth` hops of `entity` (case-insensitive match)."""
        query = f"""
        MATCH (e:Entity)
        WHERE toLower(e.name) CONTAINS toLower($entity)
        MATCH path = (e)-[*1..{depth}]-(neighbor)
        UNWIND relationships(path) AS rel
        RETURN DISTINCT startNode(rel).name AS subject,
                        rel.type AS relation,
                        endNode(rel).name AS object
        LIMIT 25
        """
        with self.driver.session() as session:
            result = session.run(query, entity=entity)
            return [dict(record) for record in result]

    def run_cypher(self, cypher: str, **params) -> list[dict]:
        """Escape hatch for arbitrary read queries (used by the LangGraph KG-query tool)."""
        with self.driver.session() as session:
            result = session.run(cypher, **params)
            return [dict(record) for record in result]


def build_graph_from_chunks(chunks: list[Chunk]) -> int:
    """Step 3 end-to-end: extract triples from each chunk and load them into Neo4j."""
    total = 0
    with KnowledgeGraph() as kg:
        for chunk in chunks:
            triples = extract_triples(chunk.text)
            def _clean_field(value) -> str:
                if isinstance(value, list):
                    value = " ".join(str(v) for v in value)
                return str(value or "").strip()

            for t in triples:
                subject = _clean_field(t.get("subject"))
                relation = _clean_field(t.get("relation"))
                obj = _clean_field(t.get("object"))
                if not subject or not relation or not obj:
                    continue
                kg.add_triple(subject, relation, obj, source=chunk.source)
                total += 1
            if triples:
                print(f"[kg_builder] {chunk.chunk_id}: +{len(triples)} triples")
    return total
