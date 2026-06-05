-- Enables the pgvector extension on database creation.
-- SQLAlchemy models handle table creation; this just guarantees the
-- extension exists before the app connects.
CREATE EXTENSION IF NOT EXISTS vector;
