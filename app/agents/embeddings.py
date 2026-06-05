"""
Embeddings factory, mirroring llm.py. Keeps ingestion and retrieval using
the same embedding model — mixing embedding models would make similarity
search meaningless, since vectors from different models aren't comparable.
"""
from app.config import settings


def get_embeddings_model():
    if settings.llm_provider == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=settings.google_embedding_model,
            google_api_key=settings.google_api_key,
        )

    if settings.llm_provider == "anthropic":
        # Anthropic doesn't offer an embeddings endpoint; fall back to OpenAI
        # if you're using Claude for chat but still need embeddings.
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model="text-embedding-3-small", api_key=settings.openai_api_key
        )

    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model="text-embedding-3-small", api_key=settings.openai_api_key
    )
