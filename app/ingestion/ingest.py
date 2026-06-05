"""
Ingestion pipeline:
    raw filing HTML  ->  clean text  ->  chunk (LlamaIndex)  ->  embed  ->  pgvector

Reads company name / filing dates automatically from the metadata.json file
that download_filings.py writes alongside each ticker's documents — no need
to pass --company-name manually.

Embedding runs locally by default (see app/agents/embeddings.py), so there's
no external rate limit to manage here — a full 10-K's worth of chunks embeds
in one pass. The first run will download the embedding model (~90MB) from
Hugging Face and cache it locally; subsequent runs are fast and fully offline.

Usage:
    python -m app.ingestion.ingest --path data/filings/C
    python -m app.ingestion.ingest --path data/filings/JPM
"""
import argparse
import json
import re
from pathlib import Path

from bs4 import BeautifulSoup
from llama_index.core.node_parser import SentenceSplitter

from app.agents.embeddings import get_embeddings_model
from app.db.session import SessionLocal, engine, Base
from app.db.models import Filing, Chunk

CHUNK_SIZE = 800       # tokens, roughly
CHUNK_OVERLAP = 100
EMBED_BATCH_SIZE = 100  # just for progress logging, not rate limiting


def clean_html(raw_html: str) -> str:
    """Strip tags/scripts/styles from a filing's HTML and collapse whitespace."""
    soup = BeautifulSoup(raw_html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def guess_section(chunk_text: str) -> str:
    """
    Very lightweight heuristic to tag a chunk with its likely 10-K section,
    based on common "Item N." headers. Good enough for retrieval filtering;
    not meant to be a robust parser.
    """
    match = re.search(r"(Item\s+\d+[A-Z]?\.\s*[A-Za-z ,&/-]{3,60})", chunk_text)
    return match.group(1).strip() if match else "Unclassified"


def chunk_text(text: str) -> list[str]:
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_text(text)


def embed_chunks(chunks: list[str]) -> list[list[float]]:
    embedder = get_embeddings_model()
    all_embeddings: list[list[float]] = []
    total_batches = (len(chunks) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE

    for batch_num, i in enumerate(range(0, len(chunks), EMBED_BATCH_SIZE), start=1):
        batch = chunks[i : i + EMBED_BATCH_SIZE]
        print(f"  Embedding batch {batch_num}/{total_batches} ({len(batch)} chunks)...")
        all_embeddings.extend(embedder.embed_documents(batch))

    return all_embeddings


def load_metadata(filings_dir: Path) -> dict:
    """Read the metadata.json written by download_filings.py for this ticker."""
    meta_files = list(filings_dir.glob("*_metadata.json"))
    if not meta_files:
        raise FileNotFoundError(
            f"No metadata.json found in {filings_dir}. "
            f"Run scripts/download_filings.py first."
        )
    return json.loads(meta_files[0].read_text())


def ingest_file(
    file_path: Path,
    ticker: str,
    form_type: str,
    company_name: str,
    filing_date: str,
):
    print(f"Ingesting {file_path.name} ({company_name} / {ticker})...")
    raw = file_path.read_text(errors="ignore")
    text = clean_html(raw)
    chunks = chunk_text(text)
    print(f"  {len(chunks)} chunks")

    embeddings = embed_chunks(chunks)

    db = SessionLocal()
    try:
        filing = Filing(
            ticker=ticker,
            company_name=company_name,
            form_type=form_type,
            filing_date=filing_date,
            source_url=str(file_path),
        )
        db.add(filing)
        db.flush()  # get filing.id

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            db.add(
                Chunk(
                    filing_id=filing.id,
                    section=guess_section(chunk),
                    chunk_index=i,
                    text=chunk,
                    embedding=embedding,
                )
            )
        db.commit()
        print(f"  Stored filing id={filing.id} with {len(chunks)} chunks.")
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description="Ingest downloaded SEC filings into pgvector.")
    parser.add_argument(
        "--path", required=True,
        help="Per-ticker directory from download_filings.py, e.g. data/filings/C",
    )
    args = parser.parse_args()

    Base.metadata.create_all(bind=engine)  # create tables if they don't exist

    filings_dir = Path(args.path)
    meta = load_metadata(filings_dir)
    ticker = meta["ticker"]
    company_name = meta["company_name"]

    doc_files = [
        f for f in filings_dir.glob("*")
        if f.suffix.lower() in (".htm", ".html") and "metadata" not in f.name
    ]
    if not doc_files:
        print(f"No .htm/.html filing documents found in {filings_dir}")
        return

    # Match each downloaded doc back to its filing_date/form_type from metadata
    # by accession number embedded in the filename.
    for f in doc_files:
        filing_info = next(
            (fl for fl in meta["filings"] if fl["accession_number"] in f.name), None
        )
        filing_date = filing_info["filing_date"] if filing_info else "unknown"
        # form_type isn't stored per-filing in metadata (all filings in one
        # metadata.json share the form type used at download time), so infer
        # from the calling context isn't needed — pull it from the filename's
        # sibling metadata file name instead.
        form_type = meta.get("form_type") or "10-K"

        ingest_file(
            f,
            ticker=ticker,
            form_type=form_type,
            company_name=company_name,
            filing_date=filing_date,
        )


if __name__ == "__main__":
    main()
