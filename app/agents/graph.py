"""
LangGraph orchestration.

Flow:
    retrieve -> analyze -> critique -> (approved: done | needs_revision: re-analyze, once)

The re-analysis loop is capped at one retry so a stubborn Critic can't spin
forever — if the second draft still fails, we return it anyway but mark it
as unverified so the caller/UI can show a warning.
"""
from typing import TypedDict

from langgraph.graph import StateGraph, END
from sqlalchemy.orm import Session

from app.agents import retriever, analyst, critic


class GraphState(TypedDict):
    question: str
    chunks: list[dict]
    draft_answer: str
    verdict: dict
    revision_count: int


MAX_REVISIONS = 1


def build_graph(db: Session):
    """
    db is closed over via closures below since LangGraph nodes only receive
    state, not arbitrary extra args. Each request should build a fresh graph
    bound to that request's DB session.
    """

    def retrieve_node(state: GraphState) -> GraphState:
        result = retriever.run(db, state["question"])
        return {**state, "chunks": result["chunks"]}

    def analyze_node(state: GraphState) -> GraphState:
        result = analyst.run(state["question"], state["chunks"])
        return {**state, "draft_answer": result["draft_answer"]}

    def critique_node(state: GraphState) -> GraphState:
        verdict = critic.run(state["question"], state["draft_answer"], state["chunks"])
        return {**state, "verdict": verdict}

    def route_after_critique(state: GraphState) -> str:
        if state["verdict"]["verdict"] == "approved":
            return "done"
        if state["revision_count"] >= MAX_REVISIONS:
            return "done"  # give up gracefully, caller sees the unresolved verdict
        return "revise"

    def increment_revision(state: GraphState) -> GraphState:
        return {**state, "revision_count": state["revision_count"] + 1}

    graph = StateGraph(GraphState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("analyze", analyze_node)
    graph.add_node("critique", critique_node)
    graph.add_node("increment_revision", increment_revision)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "analyze")
    graph.add_edge("analyze", "critique")
    graph.add_conditional_edges(
        "critique",
        route_after_critique,
        {"done": END, "revise": "increment_revision"},
    )
    graph.add_edge("increment_revision", "analyze")

    return graph.compile()


def run_query(db: Session, question: str) -> dict:
    app_graph = build_graph(db)
    initial_state: GraphState = {
        "question": question,
        "chunks": [],
        "draft_answer": "",
        "verdict": {},
        "revision_count": 0,
    }
    final_state = app_graph.invoke(initial_state)
    return {
        "answer": final_state["draft_answer"],
        "verdict": final_state["verdict"],
        "sources": [
            {"section": c["section"], "text": c["text"][:300] + "..."}
            for c in final_state["chunks"]
        ],
    }
