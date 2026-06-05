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


def _get_target_tickers(db: Session, tickers: list[str] | None) -> list[str]:
    """
    Resolve which tickers to search across. If the caller specified tickers,
    use those. Otherwise, look at what's actually been ingested — if more
    than one company is present, we need their identities to do balanced
    per-company retrieval (see retrieve_chunks docstring for why).
    """
    if tickers:
        return [t.upper() for t in tickers]

    distinct_tickers = db.execute(select(Filing.ticker).distinct()).scalars().all()
    return list(distinct_tickers)


def _search(
    db: Session, query_embedding: list[float], ticker: str | None, limit: int
) -> list[Chunk]:
    stmt = select(Chunk).join(Filing, Chunk.filing_id == Filing.id)
    if ticker:
        stmt = stmt.where(Filing.ticker == ticker)
    stmt = stmt.order_by(Chunk.embedding.cosine_distance(query_embedding)).limit(limit)
    return list(db.execute(stmt).scalars().all())


def retrieve_chunks(
    db: Session,
    question: str,
    tickers: list[str] | None = None,
    top_k: int = TOP_K,
) -> list[Chunk]:
    """
    Return the top_k most relevant Chunk rows for `question`, ranked by
    cosine distance (pgvector's `<=>` operator via the Vector column type).

    When more than one company is in play — either because `tickers` names
    several, or because nothing was specified and multiple companies have
    been ingested — retrieval is done separately per company and then
    merged, with top_k chunks from each company (not top_k split between
    them), so every company gets full, comparable coverage.

    This matters for comparison queries: a single global top_k search ranks
    every chunk from every company together by similarity, and one company's
    phrasing can simply score higher across the board, silently crowding the
    other company out of the results entirely — the Analyst then has nothing
    to say about the missing company, even though its filing almost
    certainly covers the same topic. Searching per-company guarantees both
    companies get a fair chance to contribute chunks.
    """
    query_embedding = embed_query(question)
    target_tickers = _get_target_tickers(db, tickers)

    if len(target_tickers) <= 1:
        ticker = target_tickers[0] if target_tickers else None
        return _search(db, query_embedding, ticker, top_k)

    chunks: list[Chunk] = []
    for ticker in target_tickers:
        chunks.extend(_search(db, query_embedding, ticker, top_k))
    return chunks


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
