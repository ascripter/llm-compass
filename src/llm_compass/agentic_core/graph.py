"""
Assembles the LangGraph.
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .state import AgentState
from .nodes import (
    validate_intent_node,
)


def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("validator", validate_intent_node)
    # TODO: add token_ratio, refiner, discovery, ranking, synthesis nodes

    # Edges
    workflow.set_entry_point("validator")

    # Conditional edge for clarification
    def check_validity(state):
        if state["clarification_limit_exceeded"]:
            return END
        elif state["intent_extraction"].is_specific == False:
            return "validator"
        return END

    workflow.add_conditional_edges("validator", check_validity)

    return workflow.compile(checkpointer=MemorySaver())
