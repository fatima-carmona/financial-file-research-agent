"""
FastAPI entrypoint.

POST /query    { "question": "...", "tickers": ["C", "JPM"] }
                -> runs the retrieve -> analyze -> critique graph.
                   `tickers` is optional; omit it to search across every
                   company that's been ingested (useful for open-ended or
                   comparison questions).
GET  /filings  -> lists every filing currently ingested, so callers know
                   what tickers/companies are available to query.
GET  /health   -> basic liveness check
"""
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.db.models import Filing
from app.agents.graph import run_query

app = FastAPI(
    title="Financial Filing Research Agent",
    description="Multi-agent RAG system for answering questions about SEC filings "
    "across any number of ingested companies.",
    version="0.2.0",
)


class QueryRequest(BaseModel):
    question: str
    tickers: list[str] | None = None  # e.g. ["C", "JPM"]; omit to search all


class QueryResponse(BaseModel):
    answer: str
    verdict: dict
    sources: list[dict]


class FilingSummary(BaseModel):
    id: int
    ticker: str
    company_name: str
    form_type: str
    filing_date: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/filings", response_model=list[FilingSummary])
def list_filings(db: Session = Depends(get_db)):
    filings = db.query(Filing).order_by(Filing.ticker, Filing.filing_date).all()
    return [
        FilingSummary(
            id=f.id,
            ticker=f.ticker,
            company_name=f.company_name,
            form_type=f.form_type,
            filing_date=f.filing_date,
        )
        for f in filings
    ]


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, db: Session = Depends(get_db)):
    result = run_query(db, request.question, tickers=request.tickers)
    return result
