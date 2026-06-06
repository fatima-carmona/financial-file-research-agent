"""
Analyst Agent

Takes the retrieved chunks + the user's question and drafts an answer,
citing which chunk(s) each claim comes from. Grounding instructions are
deliberately strict: the model is told to only use the provided chunks
and to mark anything it's inferring beyond them.
"""
from app.agents.llm import get_chat_model, invoke_with_retry

SYSTEM_PROMPT = """You are a financial research analyst. You answer questions \
about SEC filings using ONLY the excerpts provided to you below.

Rules:
- Base your answer strictly on the provided excerpts. Do not use outside knowledge.
- Every factual claim must reference which excerpt it came from, like [chunk 2].
- Excerpts may come from more than one company. When they do, be explicit about \
which company each claim applies to — never blend or conflate claims across \
companies, especially in comparison questions.
- If the excerpts don't contain enough information to answer, say so explicitly \
rather than guessing.
- Be concise and precise — this is going to a financial analyst, not a general reader.
"""


def build_prompt(question: str, chunks: list[dict]) -> str:
    excerpts = "\n\n".join(
        f"[chunk {i}] (company: {c['company_name']} ({c['ticker']}), "
        f"filing: {c['form_type']}, section: {c['section']})\n{c['text']}"
        for i, c in enumerate(chunks)
    )
    return f"Question: {question}\n\nExcerpts:\n\n{excerpts}\n\nAnswer:"


def run(question: str, chunks: list[dict]) -> dict:
    """Agent entrypoint used by the LangGraph orchestrator."""
    llm = get_chat_model()
    prompt = build_prompt(question, chunks)

    response = invoke_with_retry(
        llm,
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    return {"draft_answer": response.content, "chunks_used": chunks}
