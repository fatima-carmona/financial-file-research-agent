"""
Data model:

Filing   — one row per ingested SEC filing (e.g. Citigroup 10-K, FY2025)
Chunk    — one row per text chunk of a filing, with its embedding vector

Embedding dimension is set for Google's text-embedding-004 (768 dims), the
default free-tier embedding model used by this project. Change EMBEDDING_DIM
if you swap embedding models (e.g. 1536 for OpenAI's text-embedding-3-small).
"""
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import String, Text, ForeignKey, DateTime, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base

EMBEDDING_DIM = 768


class Filing(Base):
    __tablename__ = "filings"

    id: Mapped[int] = mapped_column(primary_key=True)
    ticker: Mapped[str] = mapped_column(String(10), index=True)
    company_name: Mapped[str] = mapped_column(String(255))
    form_type: Mapped[str] = mapped_column(String(10))  # "10-K", "10-Q"
    filing_date: Mapped[str] = mapped_column(String(20))
    source_url: Mapped[str] = mapped_column(Text)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )

    chunks: Mapped[list["Chunk"]] = relationship(back_populates="filing")


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(primary_key=True)
    filing_id: Mapped[int] = mapped_column(ForeignKey("filings.id"), index=True)
    section: Mapped[str] = mapped_column(String(255))  # e.g. "Item 1A. Risk Factors"
    chunk_index: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))

    filing: Mapped["Filing"] = relationship(back_populates="chunks")
