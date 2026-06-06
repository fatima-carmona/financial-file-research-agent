"""
Small factory so agents don't each need to know which LLM provider is
configured. Switch providers via LLM_PROVIDER in .env.
"""
import re
import time

from app.config import settings

MAX_RETRIES = 4
DEFAULT_RETRY_WAIT_SECONDS = 20


def get_chat_model():
    if settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=settings.anthropic_model,
            api_key=settings.anthropic_api_key,
            temperature=0,
        )

    if settings.llm_provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.google_model,
            google_api_key=settings.google_api_key,
            temperature=0,
        )

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0,
    )


def _is_rate_limit_error(exc: Exception) -> bool:
    message = str(exc)
    return "429" in message or "RESOURCE_EXHAUSTED" in message or "rate limit" in message.lower()


def _parse_retry_delay_seconds(exc: Exception) -> int:
    """Google's error message sometimes includes 'Please retry in X.Ys' — use it if present."""
    match = re.search(r"retry in (\d+(?:\.\d+)?)s", str(exc))
    if match:
        return int(float(match.group(1))) + 2  # small buffer
    return DEFAULT_RETRY_WAIT_SECONDS


def invoke_with_retry(llm, messages: list[dict]):
    """
    Wraps llm.invoke() with automatic retry/backoff on rate-limit errors.
    Even with retrieval's context budget keeping prompts reasonably sized
    (see MAX_TOTAL_CHUNKS in retriever.py), a burst of requests in a short
    window can still occasionally hit a free-tier rate limit — this retries
    instead of failing the whole query outright.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return llm.invoke(messages)
        except Exception as exc:
            if _is_rate_limit_error(exc) and attempt < MAX_RETRIES:
                wait = _parse_retry_delay_seconds(exc)
                print(f"  Rate limited on chat call (attempt {attempt}/{MAX_RETRIES}), "
                      f"waiting {wait}s before retrying...")
                time.sleep(wait)
            else:
                raise