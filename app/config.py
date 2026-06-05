"""
Central app configuration, loaded from environment variables / .env file.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM (chat/reasoning — used by Analyst and Critic agents)
    llm_provider: str = "google"  # "openai" | "anthropic" | "google"
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    google_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    anthropic_model: str = "claude-sonnet-4-6"
    google_model: str = "gemini-2.5-flash"

    # Embeddings (used by ingestion and retrieval) — separate from llm_provider
    # since embedding rate limits/costs differ a lot from chat. Default is
    # "local": runs on your machine via sentence-transformers, no API calls,
    # no rate limits, free.
    embedding_provider: str = "local"  # "local" | "google" | "openai"
    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    google_embedding_model: str = "models/gemini-embedding-001"

    # Postgres
    postgres_user: str = "filing_agent"
    postgres_password: str = "filing_agent_dev_password"
    postgres_db: str = "filings"
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    # SEC EDGAR
    sec_edgar_user_agent: str = "Financial Filing Agent contact@example.com"

    # App
    log_level: str = "INFO"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings()
