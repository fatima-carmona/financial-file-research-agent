"""
FastAPI entrypoint.

POST /query  { "question": "..." }  -> runs the retrieve -> analyze -> critique graph
GET  /health                        -> basic liveness check
"""
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.agents.graph import run_query

app = FastAPI(
    title="Financial Filing Research Agent",
    description="Multi-agent RAG system for answering questions about SEC filings.",
    version="0.1.0",
)


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    verdict: dict
    sources: list[dict]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, db: Session = Depends(get_db)):
    result = run_query(db, request.question)
    return result
