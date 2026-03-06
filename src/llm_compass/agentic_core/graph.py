"""
Assembles the LangGraph.
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .state import AgentState
from .nodes import (
    validate_intent_node,
    token_ratio_estimation_node,
    query_refiner_node,
)


def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("validator", validate_intent_node)
    workflow.add_node("token_ratio", token_ratio_estimation_node)
    workflow.add_node("refiner", query_refiner_node)
    # workflow.add_node("benchmark_discovery", benchmark_discovery)
    # TODO: add discovery, ranking, synthesis nodes

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
            return "validator"
        return ["token_ratio", "refiner"]

    workflow.add_conditional_edges("validator", check_validity)
    # workflow.add_edge(["token_ratio", "refiner"], "benchmark_discovery")
    

    return workflow.compile(checkpointer=MemorySaver())
