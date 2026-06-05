"""
Basic smoke tests. The /query test requires a running Postgres with ingested
data and a valid LLM API key, so it's marked to skip by default in CI/local
runs without that setup — run it manually once you've ingested a filing.
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.skip(reason="Requires ingested data + live LLM API key")
def test_query_returns_answer_with_sources():
    response = client.post(
        "/query",
        json={"question": "What are Citigroup's key risk factors?"},
    )
    assert response.status_code == 200
    body = response.json()
    assert "answer" in body
    assert "sources" in body
    assert len(body["sources"]) > 0
