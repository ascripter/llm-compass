"""
Assembles the LangGraph.
"""

from langgraph.graph import StateGraph, END
from .state import AgentState
from .nodes import (
    validate_intent,
    refine_query,
    execute_discovery,
    execute_ranking,
    synthesize_answer,
)


def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("validator", validate_intent)
    workflow.add_node("refiner", refine_query)
    workflow.add_node("discovery", execute_discovery)
    workflow.add_node("ranking", execute_ranking)
    workflow.add_node("synthesis", synthesize_answer)

    # Edges
    workflow.set_entry_point("validator")

    # Conditional edge for clarification
    def check_validity(state):
        return "refiner" if not state["clarification_needed"] else END

    workflow.add_conditional_edges("validator", check_validity)

    workflow.add_edge("refiner", "discovery")
    workflow.add_edge("discovery", "ranking")
    workflow.add_edge("ranking", "synthesis")
    workflow.add_edge("synthesis", END)

    return workflow.compile()
