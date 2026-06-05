"""
Ingestion pipeline:
    raw filing HTML  ->  clean text  ->  chunk (LlamaIndex)  ->  embed  ->  pgvector

Usage:
    python -m app.ingestion.ingest --path data/filings --ticker C --form-type 10-K
"""
import argparse
import re
from pathlib import Path

from bs4 import BeautifulSoup
from llama_index.core.node_parser import SentenceSplitter
from langchain_openai import OpenAIEmbeddings

from app.config import settings
from app.db.session import SessionLocal, engine, Base
from app.db.models import Filing, Chunk

CHUNK_SIZE = 800       # tokens, roughly
CHUNK_OVERLAP = 100


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
    embedder = OpenAIEmbeddings(
        model="text-embedding-3-small", api_key=settings.openai_api_key
    )
    return embedder.embed_documents(chunks)


def ingest_file(file_path: Path, ticker: str, form_type: str, company_name: str, filing_date: str):
    print(f"Ingesting {file_path.name}...")
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
    parser.add_argument("--path", required=True, help="Directory containing downloaded filings")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--form-type", default="10-K")
    parser.add_argument("--company-name", default="")
    parser.add_argument("--filing-date", default="")
    args = parser.parse_args()

    Base.metadata.create_all(bind=engine)  # create tables if they don't exist

    filings_dir = Path(args.path)
    doc_files = [
        f for f in filings_dir.glob("*")
        if f.suffix.lower() in (".htm", ".html") and "metadata" not in f.name
    ]
    if not doc_files:
        print(f"No .htm/.html filing documents found in {filings_dir}")
        return

    for f in doc_files:
        ingest_file(
            f,
            ticker=args.ticker,
            form_type=args.form_type,
            company_name=args.company_name or args.ticker,
            filing_date=args.filing_date or "unknown",
        )


if __name__ == "__main__":
    main()
