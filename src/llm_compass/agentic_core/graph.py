"""
Assembles the LangGraph.
"""

from functools import lru_cache, partial

from langgraph.graph import StateGraph, END

from .state import AgentState
from .nodes import (
    validate_intent_node,
    token_ratio_estimation_node,
    query_refiner_node,
    benchmark_discovery_node,
    benchmark_judgment_node,
    execute_ranking,
    synthesis_node,
)
from llm_compass.config import Settings, get_settings


@lru_cache(maxsize=1)
def get_graph(settings: Settings | None = None):
    """Return a cached compiled graph (built once per settings instance)."""
    return _build_graph(settings or get_settings())


def _build_graph(settings: Settings):
    workflow = StateGraph(AgentState)

    workflow.add_node("validator", partial(validate_intent_node, settings=settings))
    workflow.add_node("token_ratio", partial(token_ratio_estimation_node, settings=settings))
    workflow.add_node("refiner", partial(query_refiner_node, settings=settings))
    workflow.add_node(
        "benchmark_discovery",
        partial(benchmark_discovery_node, settings=settings),
    )
    workflow.add_node(
        "benchmark_judgment_node", partial(benchmark_judgment_node, settings=settings)
    )
    workflow.add_node("ranking", execute_ranking)
    workflow.add_node("synthesis", partial(synthesis_node, settings=settings))

    # Edges
    workflow.set_entry_point("validator")

    # Conditional edge for clarification
    def check_validity(state):
        if state.get("clarification_limit_exceeded"):
            return END
        intent_extraction = state.get("intent_extraction")
        if intent_extraction is None:
            return END
        if isinstance(intent_extraction, dict):
            is_specific = bool(intent_extraction.get("is_specific"))
        else:
            is_specific = bool(intent_extraction.is_specific)
        if is_specific is False:
            return END  # API layer handles clarification via /clarify endpoint
        return ["token_ratio", "refiner"]

    workflow.add_conditional_edges("validator", check_validity)
    workflow.add_edge(["token_ratio", "refiner"], "benchmark_discovery")
    workflow.add_edge("benchmark_discovery", "benchmark_judgment_node")
    workflow.add_edge("benchmark_judgment_node", "ranking")
    workflow.add_edge("ranking", "synthesis")
    workflow.add_edge("synthesis", END)

    return workflow.compile()
