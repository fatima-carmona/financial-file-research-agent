"""
Retriever Agent

Embeds the incoming question and does a cosine-similarity search over the
`chunks` table (pgvector) to pull back the most relevant filing sections.
"""
import re

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agents.embeddings import get_embeddings_model
from app.db.models import Chunk, Filing

TOP_K = 6

# Total chunks sent to the Analyst is capped regardless of how many companies
# are involved. Without this, a query spanning N companies retrieves top_k
# chunks from EACH one — with 6 companies at top_k=6 that's 36 chunks in a
# single prompt, which can exceed Gemini's free-tier tokens-per-minute limit.
# Instead, per-company chunk count shrinks as company count grows, so total
# context stays roughly constant whether you're comparing 2 companies or 6.
MAX_TOTAL_CHUNKS = 24
MIN_CHUNKS_PER_COMPANY = 2

# Hand-tuned aliases for names people actually type that don't cleanly fall
# out of the registered company name via normalization alone — e.g. "Chase"
# is a far more common way to refer to JPMorgan in conversation than
# "jpmorgan chase" is, and "BofA" doesn't share a substring with "bank of
# america" at all. This is a small, maintainable list scoped to companies
# actually likely to be ingested for this project; it's a supplement to the
# automatic normalization below, not a replacement for it.
KNOWN_ALIASES: dict[str, list[str]] = {
    "JPM": ["jpmorgan", "jp morgan", "chase"],
    "BAC": ["bank of america", "bofa"],
    "GS": ["goldman sachs", "goldman"],
    "MS": ["morgan stanley"],
    "WFC": ["wells fargo"],
    "C": ["citigroup", "citi"],
}

_CORPORATE_SUFFIXES = re.compile(
    r"\b(inc|incorporated|corp|corporation|co|company|ltd|llc|plc|group|holdings|n\.a\.|na)\b\.?",
    re.IGNORECASE,
)


def embed_query(question: str) -> list[float]:
    embedder = get_embeddings_model()
    return embedder.embed_query(question)


def _normalize_company_name(name: str) -> str:
    """Strip common corporate suffixes/punctuation: 'CITIGROUP INC' -> 'citigroup'."""
    cleaned = _CORPORATE_SUFFIXES.sub("", name)
    cleaned = re.sub(r"[^\w\s]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
    return cleaned


def _resolve_tickers_from_question(db: Session, question: str) -> list[str]:
    """
    Best-effort scan of the question text for company names, so a question
    like "What does Goldman Sachs say about Marcus?" gets automatically
    scoped to GS instead of silently searching every ingested company just
    because the caller didn't pass `tickers` explicitly.

    This is a heuristic substring match, not real named-entity recognition —
    it's a safety net for the common case, not a guarantee. Passing
    `tickers` explicitly is still the reliable way to scope a query.
    """
    question_lower = question.lower()
    matched: list[str] = []

    filings = db.execute(select(Filing.ticker, Filing.company_name).distinct()).all()
    for ticker, company_name in filings:
        candidates = list(KNOWN_ALIASES.get(ticker, []))
        candidates.append(_normalize_company_name(company_name))
        if any(alias and alias in question_lower for alias in candidates):
            matched.append(ticker)

    return matched


def _get_target_tickers(
    db: Session, tickers: list[str] | None, question: str
) -> list[str]:
    """
    Resolve which tickers to search across, in priority order:
    1. Tickers the caller explicitly passed — always wins, no guessing.
    2. Company names auto-detected in the question text.
    3. Fall back to every ingested company (a genuinely open comparison,
       e.g. "compare how these banks handle climate risk").
    """
    if tickers:
        return [t.upper() for t in tickers]

    resolved = _resolve_tickers_from_question(db, question)
    if resolved:
        return resolved

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
    Return the most relevant Chunk rows for `question`, ranked by cosine
    distance (pgvector's `<=>` operator via the Vector column type).

    When more than one company is in play — either because `tickers` names
    several, because company names were auto-detected in the question text,
    or because nothing matched and every ingested company is searched —
    retrieval is done separately per company and then merged, so no single
    company's phrasing can dominate a shared ranking (see MAX_TOTAL_CHUNKS
    above for why the per-company count shrinks as company count grows).
    """
    query_embedding = embed_query(question)
    target_tickers = _get_target_tickers(db, tickers, question)

    if len(target_tickers) <= 1:
        ticker = target_tickers[0] if target_tickers else None
        return _search(db, query_embedding, ticker, top_k)

    per_company_k = max(
        MIN_CHUNKS_PER_COMPANY, min(top_k, MAX_TOTAL_CHUNKS // len(target_tickers))
    )

    chunks: list[Chunk] = []
    for ticker in target_tickers:
        chunks.extend(_search(db, query_embedding, ticker, per_company_k))
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