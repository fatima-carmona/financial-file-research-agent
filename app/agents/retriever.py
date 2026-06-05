"""
Retriever Agent

Embeds the incoming question and does a cosine-similarity search over the
`chunks` table (pgvector) to pull back the most relevant filing sections.
"""
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agents.embeddings import get_embeddings_model
from app.db.models import Chunk

TOP_K = 6


def embed_query(question: str) -> list[float]:
    embedder = get_embeddings_model()
    return embedder.embed_query(question)


def retrieve_chunks(db: Session, question: str, top_k: int = TOP_K) -> list[Chunk]:
    """
    Return the top_k most relevant Chunk rows for `question`, ranked by
    cosine distance (pgvector's `<=>` operator via the Vector column type).
    """
    query_embedding = embed_query(question)

    stmt = (
        select(Chunk)
        .order_by(Chunk.embedding.cosine_distance(query_embedding))
        .limit(top_k)
    )
    return list(db.execute(stmt).scalars().all())


def run(db: Session, question: str) -> dict:
    """Agent entrypoint used by the LangGraph orchestrator."""
    chunks = retrieve_chunks(db, question)
    return {
        "chunks": [
            {
                "id": c.id,
                "filing_id": c.filing_id,
                "section": c.section,
                "text": c.text,
            }
            for c in chunks
        ]
    }
