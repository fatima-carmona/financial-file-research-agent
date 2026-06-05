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
import re

from app.agents.llm import get_chat_model

SYSTEM_PROMPT = """You are a fact-checking critic reviewing a financial analyst's \
draft answer against the source excerpts it was supposed to be based on.

For the draft answer, determine:
1. Is every claim supported by the provided excerpts?
2. If excerpts come from more than one company, does the draft correctly \
attribute each claim to the right company, without blending or conflating them?
3. List any claims that are NOT clearly supported (unsupported, overstated, or \
misattributed to the wrong company).
4. Give a final verdict: "approved" if the answer is well-grounded, or \
"needs_revision" if it contains unsupported or misattributed claims.

Respond ONLY with valid JSON in this exact shape, no other text, no markdown \
code fences, no explanation before or after the JSON:
{
  "verdict": "approved" | "needs_revision",
  "unsupported_claims": ["..."],
  "notes": "brief explanation of your verdict"
}
"""


def build_prompt(question: str, draft_answer: str, chunks: list[dict]) -> str:
    excerpts = "\n\n".join(
        f"[chunk {i}] (company: {c['company_name']} ({c['ticker']}))\n{c['text']}"
        for i, c in enumerate(chunks)
    )
    return (
        f"Original question: {question}\n\n"
        f"Draft answer:\n{draft_answer}\n\n"
        f"Source excerpts:\n\n{excerpts}"
    )


def _extract_json(raw: str) -> str:
    """
    Chat models — Gemini in particular — often wrap JSON in markdown code
    fences (```json ... ```) or add stray text around it even when told not
    to. Strip fences first, then fall back to grabbing the outermost {...}
    block if there's still leading/trailing text.
    """
    text = raw.strip()

    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()

    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        return brace_match.group(0).strip()

    return text


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
        verdict = json.loads(_extract_json(response.content))
    except json.JSONDecodeError:
        # Fail safe: if the critic still didn't return parseable JSON, don't
        # silently approve — flag it for human review instead, but keep the
        # raw response so it's possible to see what actually went wrong.
        verdict = {
            "verdict": "needs_revision",
            "unsupported_claims": [],
            "notes": "Critic response was not valid JSON; flagging for review.",
            "raw_response": response.content,
        }

    return verdict
