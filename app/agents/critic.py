"""
Critic Agent

Re-reads the Analyst's draft answer against the retrieved chunks and checks
whether each claim is actually supported by the source text. This is the
project's "responsible AI" differentiator: it catches hallucinated or
unsupported claims before they reach the user.

Returns a structured verdict rather than free text so the graph can branch
on it (approve vs. request revision).
"""
import json

from app.agents.llm import get_chat_model

SYSTEM_PROMPT = """You are a fact-checking critic reviewing a financial analyst's \
draft answer against the source excerpts it was supposed to be based on.

For the draft answer, determine:
1. Is every claim supported by the provided excerpts?
2. List any claims that are NOT clearly supported (unsupported or overstated claims).
3. Give a final verdict: "approved" if the answer is well-grounded, or \
"needs_revision" if it contains unsupported claims.

Respond ONLY with valid JSON in this exact shape, no other text:
{
  "verdict": "approved" | "needs_revision",
  "unsupported_claims": ["..."],
  "notes": "brief explanation of your verdict"
}
"""


def build_prompt(question: str, draft_answer: str, chunks: list[dict]) -> str:
    excerpts = "\n\n".join(
        f"[chunk {i}] {c['text']}" for i, c in enumerate(chunks)
    )
    return (
        f"Original question: {question}\n\n"
        f"Draft answer:\n{draft_answer}\n\n"
        f"Source excerpts:\n\n{excerpts}"
    )


def run(question: str, draft_answer: str, chunks: list[dict]) -> dict:
    """Agent entrypoint used by the LangGraph orchestrator."""
    llm = get_chat_model()
    prompt = build_prompt(question, draft_answer, chunks)

    response = llm.invoke(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
    )

    try:
        verdict = json.loads(response.content)
    except json.JSONDecodeError:
        # Fail safe: if the critic didn't return clean JSON, don't silently
        # approve — flag it for human review instead.
        verdict = {
            "verdict": "needs_revision",
            "unsupported_claims": [],
            "notes": "Critic response was not valid JSON; flagging for review.",
        }

    return verdict
