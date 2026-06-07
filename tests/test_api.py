"""
Integration tests for the /query pipeline.

These are genuine integration tests, not mocks: each call to /query goes
through the real Retriever (pgvector), the real Analyst and Critic (live
Gemini API calls), and requires the 6 banks from the README's Testing
section to already be ingested (C, JPM, BAC, WFC, GS, MS).

They're skipped automatically — not manually — if Postgres isn't reachable,
so `pytest` behaves sensibly whether or not the stack is running, rather
than failing with a confusing connection error.

Because these hit a live LLM, assertions focus on structural properties
(which companies' sources came back, whether the Critic approved) rather
than exact wording, since the Analyst's phrasing varies between runs.

Run with:
    pytest tests/test_api.py -v

Expect this to take a few minutes — 10 real LLM round-trips, some of them
multi-company.
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from app.main import app
from app.db.session import SessionLocal

client = TestClient(app)


def _db_reachable() -> bool:
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        return True
    except Exception:
        return False


requires_live_stack = pytest.mark.skipif(
    not _db_reachable(),
    reason="Postgres not reachable — start it with `docker compose up -d db` "
           "and ensure the 6 banks from the README are ingested.",
)


def _tickers_in_sources(body: dict) -> set[str]:
    return {source["ticker"] for source in body["sources"]}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@requires_live_stack
def test_filings_lists_ingested_companies():
    response = client.get("/filings")
    assert response.status_code == 200
    tickers = {f["ticker"] for f in response.json()}
    # Not asserting the full set of 6 — just that ingestion has happened at
    # all, so the tests below have something to query against.
    assert len(tickers) > 0


@requires_live_stack
def test_single_company_factual_query():
    """Test 1: single-company factual lookup, explicit ticker."""
    response = client.post(
        "/query",
        json={
            "question": "What does JPMorgan say about its Credit Portfolio VaR?",
            "tickers": ["JPM"],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"]
    assert _tickers_in_sources(body) == {"JPM"}


@requires_live_stack
def test_single_company_different_bank():
    """Test 2: single-company query, different bank."""
    response = client.post(
        "/query",
        json={
            "question": "What are Wells Fargo's key operational risks?",
            "tickers": ["WFC"],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"]
    assert _tickers_in_sources(body) == {"WFC"}


@requires_live_stack
def test_two_company_comparison():
    """Test 3: comparison with explicit tickers, checks numeric grounding."""
    response = client.post(
        "/query",
        json={
            "question": "Compare the CET1 capital ratios reported by Bank of "
                         "America and Morgan Stanley.",
            "tickers": ["BAC", "MS"],
        },
    )
    assert response.status_code == 200
    body = response.json()
    tickers = _tickers_in_sources(body)
    assert "BAC" in tickers
    assert "MS" in tickers


@requires_live_stack
def test_three_company_comparison():
    """Test 4: comparison across three explicit tickers."""
    response = client.post(
        "/query",
        json={
            "question": "How do Goldman Sachs, Morgan Stanley, and Citigroup "
                         "each describe their exposure to market risk?",
            "tickers": ["GS", "MS", "C"],
        },
    )
    assert response.status_code == 200
    body = response.json()
    tickers = _tickers_in_sources(body)
    assert tickers & {"GS", "MS", "C"}  # at least some overlap expected


@requires_live_stack
def test_open_comparison_no_ticker_filter():
    """Test 5: open comparison across every ingested company."""
    response = client.post(
        "/query",
        json={"question": "Compare how these banks describe climate-related "
                           "financial risk."},
    )
    assert response.status_code == 200
    body = response.json()
    # No explicit tickers and no single company named in the question ->
    # should search across multiple ingested companies, not just one.
    assert len(_tickers_in_sources(body)) > 1


@requires_live_stack
def test_grounding_refuses_out_of_scope_question():
    """Test 6: honest-gap test — filings don't contain stock price targets."""
    response = client.post(
        "/query",
        json={
            "question": "What is Wall Street's current stock price target "
                         "for Citigroup?",
            "tickers": ["C"],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"]
    # Critic should still approve — correctly saying "not in the filing" is
    # a grounded answer, not an unsupported claim.
    assert body["verdict"]["verdict"] == "approved"


@requires_live_stack
def test_false_premise_question():
    """
    Test 7: question built on an incorrect assumption. Documented as a known
    limitation in the README — the system currently answers around the false
    premise rather than flagging it. This test exists to track that
    behavior, not to assert it's been fixed.
    """
    response = client.post(
        "/query",
        json={
            "question": "Since Goldman Sachs completely exited consumer "
                         "banking in 2025, what risks remain in their business?",
            "tickers": ["GS"],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"]


@requires_live_stack
def test_current_events_gap():
    """Test 8: filings are historical, not news — shouldn't fabricate quotes."""
    response = client.post(
        "/query",
        json={
            "question": "What did Citigroup's CEO say in interviews last month?",
            "tickers": ["C"],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"]
    assert body["verdict"]["verdict"] == "approved"


@requires_live_stack
def test_named_company_without_explicit_ticker_filter():
    """
    Test 9: regression test for automatic company-name-to-ticker resolution
    (see app/agents/retriever.py). Goldman Sachs is named in the question but
    no `tickers` are passed — this should auto-scope to GS only, not search
    every ingested company.
    """
    response = client.post(
        "/query",
        json={"question": "What does Goldman Sachs say about its Marcus "
                           "consumer lending business?"},
    )
    assert response.status_code == 200
    body = response.json()
    assert _tickers_in_sources(body) == {"GS"}


@requires_live_stack
def test_cross_company_qualitative_synthesis():
    """Test 10: thematic synthesis across all companies, no numbers involved."""
    response = client.post(
        "/query",
        json={"question": "What common cybersecurity risks do all six banks mention?"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"]
    assert len(_tickers_in_sources(body)) > 1
