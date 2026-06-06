# Financial Filing Research Agent

A multi-agent Retrieval-Augmented Generation (RAG) system that answers questions
about public companies' SEC filings (10-K / 10-Q) with cited, fact-checked answers
— across any number of companies, including side-by-side comparisons.

Example queries:
- *"What are Citigroup's key risk factors in their latest 10-K?"*
- *"Compare Citigroup's and JPMorgan's exposure to credit risk."*

The system retrieves relevant filing sections, reasons over them, and verifies
that the final answer is actually grounded in the source text — and correctly
attributed to the right company — before returning it.

## Why this project

Built as a hands-on exploration of the technologies used in modern Gen AI
application development: RAG pipelines, multi-agent orchestration, vector
databases, and containerized deployment. It's scoped to the finance domain —
filings are ingested directly from SEC EDGAR, which is public, free, official,
and structured, making it a reliable source of truth for a fact-checking
pipeline. Any public company can be ingested by ticker; it's not limited to one.

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

### Multi-company support

Every part of the pipeline is company-agnostic:
- Filings are ingested per-ticker into their own subfolder and DB rows (no mixing)
- The Retriever accepts an optional `tickers` filter, so a query can be scoped to
  one company or left open to search across everything ingested
- The Analyst is instructed to attribute each claim to the specific company it
  came from, and the Critic explicitly checks for cross-company misattribution
  before approving an answer

This is what makes comparison questions ("How does Citigroup's risk disclosure
differ from JPMorgan's?") reliable rather than just concatenating both companies'
text and hoping the model sorts it out correctly.

## Stack

| Layer | Technology |
|---|---|
| API | Python, FastAPI |
| Orchestration | LangGraph (multi-agent graph) |
| Retrieval | LangChain |
| Ingestion / chunking | LlamaIndex |
| Vector store | PostgreSQL + pgvector |
| LLM | Configurable — Google Gemini (free tier, default), OpenAI, or Anthropic |
| Deployment | Docker, docker-compose |

## Project layout

```
financial-filing-agent/
├── app/
│   ├── main.py              # FastAPI app: /query, /filings, /health
│   ├── config.py            # env-based settings
│   ├── ingestion/
│   │   └── ingest.py        # chunk + embed filings, write to pgvector
│   ├── agents/
│   │   ├── graph.py         # LangGraph wiring: retriever → analyst → critic
│   │   ├── retriever.py     # pgvector similarity search, optional ticker filter
│   │   ├── analyst.py       # grounded, per-company-attributed answer generation
│   │   ├── critic.py        # fact-checks + cross-company misattribution check
│   │   ├── llm.py           # chat model factory (Gemini/OpenAI/Anthropic)
│   │   └── embeddings.py    # embedding model factory
│   └── db/
│       ├── models.py        # SQLAlchemy models (filings, chunks)
│       └── session.py       # DB engine/session setup
├── scripts/
│   └── download_filings.py  # pulls filings from SEC EDGAR, any ticker
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
# edit .env and add your Gemini API key (GOOGLE_API_KEY) — free, no card required:
# https://aistudio.google.com/apikey
```

### 2. Start Postgres + pgvector

```bash
docker compose up -d db
```

### 3. Install dependencies (local dev, outside Docker)

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Download and ingest filings

Each company gets its own subfolder under `data/filings/<TICKER>/`, so you can
ingest as many companies as you want without them mixing:

```bash
python scripts/download_filings.py --ticker C --form-type 10-K --count 1
python -m app.ingestion.ingest --path data/filings/C

python scripts/download_filings.py --ticker JPM --form-type 10-K --count 1
python -m app.ingestion.ingest --path data/filings/JPM
```

### 5. Run the API

```bash
uvicorn app.main:app --reload
```

Then query it — single company:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are Citigroup'"'"'s key risk factors?", "tickers": ["C"]}'
```

Or across companies (comparison, or just omit `tickers` to search everything ingested):

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Compare Citigroup'"'"'s and JPMorgan'"'"'s exposure to credit risk."}'
```

Check what's currently ingested:

```bash
curl http://localhost:8000/filings
```

### 6. Or run everything in Docker

```bash
docker compose up --build
```

## Testing

The system has been ingested with 10-K filings from six major banks
(Citigroup, JPMorgan, Bank of America, Wells Fargo, Goldman Sachs, Morgan
Stanley) and tested against 10 queries spanning single-company lookups,
multi-company comparisons, and adversarial cases designed to probe whether
the Critic agent actually catches problems rather than rubber-stamping
everything.

**Single-company factual queries** — correctly retrieved and cited specific
figures (e.g., JPMorgan's Credit Portfolio VaR, Wells Fargo's operational
risk disclosures), attributing every claim back to its source chunk.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What does JPMorgan say about its Credit Portfolio VaR?", "tickers": ["JPM"]}'
```

**Multi-company comparisons (2-3 companies, explicit tickers)** — correctly
pulled and compared numeric figures (e.g., CET1 capital ratios between Bank
of America and Morgan Stanley) and qualitative risk descriptions (market risk
across Goldman Sachs, Morgan Stanley, and Citigroup) without blending or
misattributing claims between companies.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Compare the CET1 capital ratios reported by Bank of America and Morgan Stanley.", "tickers": ["BAC", "MS"]}'
```

**Grounding / hallucination-resistance tests** — asked questions the filings
can't answer (a stock price target, a recent CEO interview). In both cases
the system correctly reported that the source material didn't contain the
answer instead of inventing one.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Wall Street'"'"'s current stock price target for Citigroup?", "tickers": ["C"]}'
```

**False-premise test** — asked a question built on an incorrect assumption
("since Goldman Sachs completely exited consumer banking in 2025..."). The
system answered around the false premise rather than rejecting it outright —
see Known Limitations below.

**All-six open comparison (no ticker filter)** — asked to compare climate
risk disclosures and, separately, cybersecurity risk themes across all six
banks with no `tickers` specified. This exercised the balanced per-company
retrieval logic across every ingested company at once, correctly noting when
one bank's retrieved excerpts didn't cover the topic instead of hallucinating
a claim on that bank's behalf — but also exposed a real scaling limitation
(next section).

## Known limitations

Being upfront about where this breaks is as useful as showing where it
works — these are the real edges found through testing, not hypothetical
ones:

- **Broad, filter-free queries across many companies can exceed Gemini's
  free-tier token-per-minute limit.** With 6 companies ingested, an
  unfiltered query retrieves `top_k` chunks from *each* company (36 chunks
  total) and sends all of it to the Analyst in one call. This occasionally
  triggered a rate-limit error during testing. A production version would
  scale `top_k` down as the number of companies grows, or use a map-reduce
  pattern (summarize each company's relevant chunks independently, then
  synthesize the summaries) instead of sending every raw chunk from every
  company in a single prompt.
- **No automatic company-name-to-ticker resolution.** If a question names a
  specific company (e.g., "What does Goldman Sachs say about Marcus?") but
  the caller doesn't also pass `tickers: ["GS"]`, the system has no way to
  know the question was scoped to one company — it defaults to searching
  every ingested company. The fix would be a lightweight step that extracts
  company names from the question and resolves them to tickers before
  retrieval, rather than requiring the caller to specify `tickers` manually.
- **The Critic verifies the Analyst's answer against source chunks, but
  doesn't currently challenge false premises embedded in the question
  itself.** Asking a question built on an incorrect assumption gets answered
  around the assumption rather than the assumption being flagged outright.
  A stronger version would have the Critic (or a dedicated step) check the
  question's own premises against retrieved chunks before the Analyst
  answers, not just check the Analyst's output afterward.

## Status / roadmap

This is an actively-developed portfolio project. Current focus:

- [x] Project scaffold, DB models, Docker setup
- [x] EDGAR ingestion script (multi-company, per-ticker folders)
- [x] Retriever agent (pgvector similarity search, balanced per-company search)
- [x] Analyst agent (grounded answer generation, per-company attribution)
- [x] Critic agent (claim verification + cross-company misattribution checks)
- [x] LangGraph wiring (retrieve → analyze → critique, with revision loop)
- [x] FastAPI endpoint (`/query`, `/filings`, `/health`)
- [x] Ingested 6 major banks (C, JPM, BAC, WFC, GS, MS) and ran 10 test queries
- [x] Example queries + results documented above
- [ ] Tests beyond the health-check smoke test (formalize the manual test
      queries above into actual pytest cases)
- [ ] Adaptive retrieval scope to avoid token-limit errors on broad,
      many-company queries
- [ ] Automatic company-name → ticker resolution
- [ ] Critic (or a new agent) checks question premises, not just answer claims