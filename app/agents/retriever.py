"""
Retriever Agent

Embeds the incoming question and does a cosine-similarity search over the
`chunks` table (pgvector) to pull back the most relevant filing sections.
"""
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agents.embeddings import get_embeddings_model
from app.db.models import Chunk, Filing

TOP_K = 6


def embed_query(question: str) -> list[float]:
    embedder = get_embeddings_model()
    return embedder.embed_query(question)


def retrieve_chunks(
    db: Session,
    question: str,
    tickers: list[str] | None = None,
    top_k: int = TOP_K,
) -> list[Chunk]:
    """
    Return the top_k most relevant Chunk rows for `question`, ranked by
    cosine distance (pgvector's `<=>` operator via the Vector column type).

    If `tickers` is provided, results are restricted to filings from those
    companies — this is what makes comparison queries ("compare Citigroup's
    and JPMorgan's risk factors") reliable: without a filter, semantic search
    alone might over-favor one company's chunks over the other's.
    """
    query_embedding = embed_query(question)

    stmt = select(Chunk).join(Filing, Chunk.filing_id == Filing.id)
    if tickers:
        upper_tickers = [t.upper() for t in tickers]
        stmt = stmt.where(Filing.ticker.in_(upper_tickers))

    stmt = stmt.order_by(Chunk.embedding.cosine_distance(query_embedding)).limit(top_k)
    return list(db.execute(stmt).scalars().all())


def run(db: Session, question: str, tickers: list[str] | None = None) -> dict:
    """Agent entrypoint used by the LangGraph orchestrator."""
    chunks = retrieve_chunks(db, question, tickers=tickers)
    return {
        "chunks": [
            {
                "id": c.id,
                "filing_id": c.filing_id,
                "ticker": c.filing.ticker,
                "company_name": c.filing.company_name,
                "form_type": c.filing.form_type,
                "filing_date": c.filing.filing_date,
                "section": c.section,
                "text": c.text,
            }
            for c in chunks
        ]
    }
