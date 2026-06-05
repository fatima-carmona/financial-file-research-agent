# Financial Filing Research Agent

A multi-agent Retrieval-Augmented Generation (RAG) system that answers questions
about public company SEC filings (10-K / 10-Q) with cited, fact-checked answers.

Example query: *"What are Citigroup's key risk factors in their latest 10-K?"*

The system retrieves relevant filing sections, reasons over them, and verifies
that the final answer is actually grounded in the source text before returning it.

## Why this project

Built as a hands-on exploration of the technologies used in modern Gen AI
application development: RAG pipelines, multi-agent orchestration, vector
databases, and containerized deployment. It's intentionally scoped to the
finance domain — filings are ingested directly from SEC EDGAR, which is public,
free, and a natural fit for testing on real company data (e.g. Citigroup's own
10-K).

## Architecture

```
User Query
    │
    ▼
FastAPI endpoint (/query)
    │
    ▼
LangGraph orchestrator
    │
    ├─► Retriever Agent   — embeds the query, does similarity search over
    │                        chunked filing text stored in Postgres/pgvector
    │
    ├─► Analyst Agent     — reasons over retrieved chunks, drafts an answer
    │                        with inline citations back to source chunks
    │
    └─► Critic Agent      — checks the Analyst's answer against the retrieved
                             chunks, flags any claims that aren't supported by
                             the source text, and either approves the answer
                             or sends it back for revision
    │
    ▼
Response (answer + citations + confidence/verification notes)
```

### Why a Critic agent

Most basic RAG demos stop at "retrieve + generate." The Critic agent is the
differentiator here: it re-reads the Analyst's draft against the retrieved
source chunks and flags unsupported claims before the answer goes back to the
user. This maps directly to the "responsible AI" / "ethical AI guidelines"
emphasis that shows up in Gen AI job postings at regulated companies like banks.

## Stack

| Layer | Technology |
|---|---|
| API | Python, FastAPI |
| Orchestration | LangGraph (multi-agent graph) |
| Retrieval | LangChain |
| Ingestion / chunking | LlamaIndex |
| Vector store | PostgreSQL + pgvector |
| LLM | Configurable — OpenAI or Anthropic (Claude) via API key |
| Deployment | Docker, docker-compose |

## Project layout

```
financial-filing-agent/
├── app/
│   ├── main.py              # FastAPI app, /query and /health endpoints
│   ├── config.py            # env-based settings
│   ├── ingestion/
│   │   └── ingest.py        # chunk + embed filings, write to pgvector
│   ├── agents/
│   │   ├── graph.py         # LangGraph wiring: retriever → analyst → critic
│   │   ├── retriever.py
│   │   ├── analyst.py
│   │   └── critic.py
│   └── db/
│       ├── models.py        # SQLAlchemy models (filings, chunks)
│       └── session.py       # DB engine/session setup
├── scripts/
│   └── download_filings.py  # pulls 10-Ks from SEC EDGAR
├── tests/
│   └── test_api.py
├── docker-compose.yml        # app + Postgres/pgvector
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

## Getting started

### 1. Clone and configure

```bash
git clone <your-repo-url>
cd financial-filing-agent
cp .env.example .env
# edit .env and add your LLM API key (OPENAI_API_KEY or ANTHROPIC_API_KEY)
```

### 2. Start Postgres + pgvector

```bash
docker compose up -d db
```

### 3. Install dependencies (local dev, outside Docker)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Download and ingest a filing

```bash
python scripts/download_filings.py --ticker C --form-type 10-K --count 1
python -m app.ingestion.ingest --path data/filings
```

### 5. Run the API

```bash
uvicorn app.main:app --reload
```

Then query it:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are Citigroup'"'"'s key risk factors in their latest 10-K?"}'
```

### 6. Or run everything in Docker

```bash
docker compose up --build
```

## Status / roadmap

This is an actively-developed portfolio project. Current focus:

- [x] Project scaffold, DB models, Docker setup
- [ ] EDGAR ingestion script
- [ ] Retriever agent (pgvector similarity search)
- [ ] Analyst agent (grounded answer generation)
- [ ] Critic agent (claim verification against source chunks)
- [ ] LangGraph wiring
- [ ] FastAPI endpoint
- [ ] Tests
- [ ] Example queries + sample output in README

## License

MIT
