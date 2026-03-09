"""
Req 2.3 Node 2 (b): Token Ratio Estimation (LLM)

This node runs after intent validation succeeds (parallel to Refine Query)
It performs token ratio estimation (modality-aware units + normalized ratios).
For this task an LLM with good reasoning is essential since it needs to understand
the semantics of the task, needs to deduce typical prompts and then deduce their token amounts.

- Input: validated chat history
- Output: TokenRatioEstimation object + logs
"""

from typing import Any
import logging

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, AnyMessage

from llm_compass.config import Settings
from llm_compass.common.schemas import Constraints
from llm_compass.common.types import MODALITY_VALUES
from ..state import AgentState
from ..schemas import TokenRatioEstimation

logger = logging.getLogger(__name__)


def _system_prompt(chat_history: list[AnyMessage]):
    """Make system prompt dependant on chat history, i.e. whether clarification was needed."""
    if len(chat_history) == 1:
        intro = "Given the original user query"
    else:
        intro = ("Given the original user query in the first message "
        "and the consecutive clarification chat")
    return f"""You are a token volume estimation assistant in an AI routing pipeline.

{intro}:
1. Estimate realistic input units and output units for {", ".join(MODALITY_VALUES)}.
2. Keep estimates practical for a typical single LLM invocation.
3. Follow the output schema strictly.
"""

def token_ratio_estimation_node(state: AgentState, settings: Settings) -> dict[str, Any]:
    llm = settings.make_llm("moonshotai/kimi-k2.5", temperature=0)
    token_estimator = llm.with_structured_output(TokenRatioEstimation)

    history = state.get("messages", [])
    messages = [SystemMessage(content=_system_prompt(history))] + history
    token_ratio: TokenRatioEstimation = token_estimator.invoke(messages)  # type: ignore

    logger.debug(
        "token_ratio_estimation_node EXIT | input_ratios=%s | output_ratios=%s | reasoning=%r",
        token_ratio.normalized_input_ratios,
        token_ratio.normalized_output_ratios,
        getattr(token_ratio, "reasoning", "")[:120],
    )

    logs = [
        (
            "Token Ratio Estimation: "
            f"input={token_ratio.normalized_input_ratios}, "
            f"output={token_ratio.normalized_output_ratios}"
        )
    ]
    return {
        "token_ratio_estimation": token_ratio,
        "logs": logs,
    }