"""
Defines the LangGraph state.
Req 2.3: Carries UI inputs and Agent intermediate steps.
"""

from typing import TypedDict, List, Dict, Any, Annotated, Optional, Literal
import operator

from langchain_core.messages import AnyMessage
from langgraph.graph import MessagesState  # implies `messages` attribute
from langgraph.graph.message import add_messages

from .schemas import IntentExtraction, TokenRatioEstimation
from ..common.schemas import Constraints
from ..common.types import Modality


class AgentState(MessagesState):
    user_query: str
    constraints: Constraints  # UI Constraints (Req 3.2: Context, Modality, Deployment, etc.)

    # From intent validation node (Req 2.3 Node 1)
    clarification_count: int  # Tracks cycles; max 3 before terminal error
    clarification_limit_exceeded: bool
    ui_mismatch_hinted: bool  # True: user was hinted at mismatch between UI/query modalities
    intent_extraction: Optional[IntentExtraction]

    # From Query Refinement (Req 2.3 Node 2)
    token_ratio_estimation: Optional[TokenRatioEstimation]
    search_queries: List[str]

    # From Benchmark Discovery (Req 2.3 Node 3)
    weighted_benchmarks: List[Dict[str, Any]]

    # From Scoring and Ranking (Req 2.3 Node 4)
    ranked_results: Dict[str, List[Any]]  # "top", "balanced", "budget" lists

    # From Synthesize Node (Req 2.3 Node 5)
    final_response: Optional[Dict]  # The JSON schema for UI

    # Traceability (Req 3.3.A)
    logs: Annotated[List[str], add_messages]


def get_initial_state():
    """Caller needs to update the keys "messages", "user_query" and "constraints" """
    return AgentState(
        {
            "messages": [],
            "user_query": "",
            "constraints": Constraints(),
            "clarification_count": 0,
            "clarification_limit_exceeded": False,
            "ui_mismatch_hinted": False,
            "intent_extraction": None,
            "token_ratio_estimation": None,
            "search_queries": [],
            "weighted_benchmarks": [],
            "ranked_results": {},
            "final_response": None,
            "logs": [],
        }
    )
