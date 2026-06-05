"""
Embeddings factory, mirroring llm.py.

Embeddings and chat use separate provider settings (EMBEDDING_PROVIDER vs
LLM_PROVIDER) because they have very different rate-limit realities: Gemini's
free tier allows generous chat usage but throttles embedding calls hard
(~100 requests/minute), and a single 10-K can produce 900+ chunks — one call
per chunk. Running embeddings locally via sentence-transformers avoids that
entirely: no API calls, no rate limits, no cost, works offline. Keep
ingestion and retrieval on the same embedding model — mixing them would make
similarity search meaningless, since vectors from different models aren't
comparable.

Note: local embeddings use sentence-transformers directly rather than the
langchain-huggingface package. That package has had major version churn and
its current release requires a much newer langchain-core than the rest of
this project's pinned LangChain stack, causing dependency resolution
conflicts. This small wrapper avoids that entirely.
"""
from app.config import settings


class _LocalEmbeddings:
    """
    Minimal wrapper around sentence-transformers exposing the same
    embed_documents / embed_query interface LangChain's Embeddings classes
    use, so it's a drop-in replacement wherever get_embeddings_model() is
    called elsewhere in the project.
    """

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode([text], show_progress_bar=False)[0].tolist()


def get_embeddings_model():
    if settings.embedding_provider == "local":
        return _LocalEmbeddings(settings.local_embedding_model)

    if settings.embedding_provider == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=settings.google_embedding_model,
            google_api_key=settings.google_api_key,
        )

    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model="text-embedding-3-small", api_key=settings.openai_api_key
    )
