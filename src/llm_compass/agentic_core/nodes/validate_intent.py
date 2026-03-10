"""
Req 2.3 Node 1: Intent Validator (LLM)

This node acts as a gatekeeper for the agentic workflow. It performs three critical checks:
1. Task Description Check: Ensures the user's query is concrete and actionable.
2. Ambiguity Check (I/O Ratio): Determines if the query implies a clear input/output token ratio.
3. Consistency Check: Validates that the user's textual intent does not conflict with UI filters.

If any check fails, it sets `clarification_needed` to True and provides a `clarification_question`.
It enforces a limit of 3 consecutive clarification cycles (max 3 before terminal error).

Req 2.3 Node 1 Workflow:
- Input: user_query + constraints from UI
- Output: IntentExtraction object + clarification_count + clarification_limit_exceeded + logs
- Routing: If IntentExtraction.is_specific is False, pause and wait for user response;
           else proceed to Node 2 (a) (Query Refiner) and Node 2 (b) (Token Ratio Estimation)
"""

from typing import Any
import logging

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from llm_compass.config import Settings
from llm_compass.common.schemas import Constraints
from llm_compass.common.types import MODALITY_VALUES
from ..state import AgentState
from ..schemas import IntentExtraction

logger = logging.getLogger(__name__)

# System prompt for the LLM-based validator
INTENT_VALIDATOR_SYSTEM_PROMPT = """You are an intent validation assistant in an AI routing pipeline.

## Your Task
Analyze the user's request (and any conversation history) to:
1. Determine their intended task.
2. Extract the intended input and output modalities.
3. Assess whether the request is specific enough to proceed.

Follow the rules and examples defined in your output schema.

## Clarification Scope
When the request is unclear, ask ONLY about:
- The actual use case or task description.
- Input modalities (text, image, audio, video).
- Output modalities (text, image, audio, video).

## Strict Rules
You MUST NOT ask about or mention ANY of the following topics. This is non-negotiable:
- Open-source vs commercial or proprietary model preferences.
- Cost or pricing sensitivity.
- Model speed or latency requirements.
- General accuracy or benchmark performance.
- Specific model names or providers.
- Specific output format."""


def validate_intent_node(state: AgentState, settings: Settings) -> dict[str, Any]:
    """
    Validates user intent and UI constraint consistency (Req 2.3 Node 1).

    This node:
    1. Increments the clarification counter.
    2. Checks if the limit (3 cycles) has been exceeded.
    3. Calls an LLM to validate the query
    4. Returns state updates with clarification flags and logs.

    Args:
        state (AgentState): The current graph state containing user query and constraints.

    Returns:
        dict[str, Any]: State updates including:
            - messages (AIMessage appended)
            - clarification_count: int (incremented)
            - clarification_limit_exceeded: bool (True if threshold reached)
            - intent_extraction: IntentExtraction
            - logs: List[str] (appended)
    """
    hints_msg = (
        "- describe your task clearly\n"
        f"- input and output modalities should be clear for your task {MODALITY_VALUES}\n"
        "- make sure the modalities selected in the UI are consistent with your task\n"
        "- if possible, estimate the amount of tokens for input / output *per modality* "
        "that will occur with an average LLM invocation"
    )

    logger.debug(
        "validate_intent_node ENTRY | user_query=%r | clarification_count=%d | constraints=%s | messages=%s",
        state.get("user_query"),
        state.get("clarification_count", 0),
        state.get("constraints"),
        [
            f"{type(m).__name__}({getattr(m, 'content', '')[:60]!r})"
            for m in state.get("messages", [])
        ],
    )

    # patch: use 4o-mini since gpt-oss-120b doesn't adhere to schema consistently
    llm = settings.make_llm("openai/gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(IntentExtraction)
    messages = [SystemMessage(INTENT_VALIDATOR_SYSTEM_PROMPT)] + state["messages"]  # type:ignore
    response: IntentExtraction = structured_llm.invoke(messages)  # type: ignore

    # Pattern to correctly instantiate pydantic objects even when resumed after checkpoint
    _raw = state.get("constraints")
    constraints = Constraints(**_raw) if isinstance(_raw, dict) else _raw

    clarification_count = state.get("clarification_count", 0)
    ui_mismatch_hinted = state.get("ui_mismatch_hinted", False)
    logs = []

    state_update: dict[str, Any] = {
        "intent_extraction": response,
    }

    # Check if query is consistent with UI constraints
    ui_missing_input = [
        _ for _ in response.intended_input_modalities if _ not in constraints.modality_input
    ]
    ui_missing_output = [
        _ for _ in response.intended_output_modalities if _ not in constraints.modality_output
    ]
    ui_overspec_input = [
        _ for _ in constraints.modality_input if _ not in response.intended_input_modalities
    ]
    ui_overspec_output = [
        _ for _ in constraints.modality_output if _ not in response.intended_output_modalities
    ]
    ui_mismatch = ui_missing_input or ui_missing_output or ui_overspec_input or ui_overspec_output
    logger.debug(
        f"ui_missing_input={ui_missing_input} | ui_missing_output={ui_missing_output} | "
        f"ui_overspec_input={ui_overspec_input} | ui_overspec_output={ui_overspec_output}"
    )
    # Only trigger clarification for mismatch if user hasn't been hinted yet
    effective_ui_mismatch = ui_mismatch and not ui_mismatch_hinted

    if (not response.is_specific or effective_ui_mismatch) and clarification_count >= 3:
        # trials exceeded
        msg = (
            "I've asked for clarification multiple times, but I'm still unable to understand "
            "your request properly. Please start over and be as specific as possible:\n"
            f"{hints_msg}"
        )
        logs.append(
            (f"Intent Validator: Clarification limit exceeded (count={clarification_count}).")
        )
        state_update["clarification_limit_exceeded"] = True
        state_update["messages"] = [AIMessage(content=msg)]  # type: ignore
    elif not response.is_specific or effective_ui_mismatch:
        state_update["clarification_count"] = clarification_count + 1
        msg = ""
        if not response.is_specific:
            # Format the list of clarification questions into a single cohesive message
            if len(response.clarification_needed) == 0:
                # Shouldn't happen since LLM was instructed to provide 1 entry, but *can* happen
                msg += f"This is too vague. Please be as specific as possible:\n{hints_msg}"
            elif len(response.clarification_needed) == 1:
                msg += response.clarification_needed[0]
            else:
                msg += "Please clarify the following points:\n" + "\n".join(
                    f"- {q}" for q in response.clarification_needed
                )
            logs.append(f"Intent Validator: Response not specific:\n'''{msg}'''")
            msg += "\n\n"

        if effective_ui_mismatch:
            # Add info about modality mismatch
            msg += "Your task indicates the following modalities:\n"
            msg += f"- input: {response.intended_input_modalities}\n"
            msg += f"- output: {response.intended_output_modalities}\n\n"
            msg += "This conflicts with your selection in the UI. Please change *either* the UI "
            msg += "filters or clarify which input and output modalities you intend to use."
            logs.append(
                (
                    "Intent Validator: Modality-mismatch.\nMissing UI input: "
                    f"{ui_missing_input}; and output: {ui_missing_output}.\n"
                    "Overspec UI input: {ui_overspec_input}; and output: {ui_overspec_output}"
                )
            )
            state_update["ui_mismatch_hinted"] = True

        # Append this AIMessage to the conversation history
        state_update["messages"] = [AIMessage(content=msg)]  # type: ignore

    if logs:
        state_update["logs"] = logs

    logger.debug(
        "validate_intent_node EXIT"
        " | is_specific=%s"
        " | clarification_count=%s"
        " | clarification_limit_exceeded=%s"
        " | intended_input=%s"
        " | intended_output=%s"
        " | messages_appended=%d"
        " | logs_appended=%d"
        " | reasoning=%r",
        getattr(response, "is_specific", None),
        state_update.get("clarification_count", state.get("clarification_count", 0)),
        state_update.get("clarification_limit_exceeded", False),
        getattr(response, "intended_input_modalities", None),
        getattr(response, "intended_output_modalities", None),
        len(state_update.get("messages", [])),
        len(state_update.get("logs", [])),
        getattr(response, "reasoning", "")[:120],
    )
    return state_update
