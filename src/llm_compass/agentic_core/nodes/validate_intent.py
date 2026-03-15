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

HINTS_MSG = f"""- describe your task clearly
- input and output modalities should be clear for your task {MODALITY_VALUES}
- make sure the modalities selected in the UI are consistent with your task
- if possible, estimate the amount of tokens for input / output *per modality* that will occur with an average LLM invocation"""

# System prompt for the LLM-based validator
INTENT_VALIDATOR_SYSTEM_PROMPT = f"""You are an intent validator in an LLM benchmark-routing pipeline.

## Goal
Decide whether the user's request is specific enough to retrieve relevant benchmark descriptions.

A request is SPECIFIC if the current chat history is sufficient for a downstream retrieval step to find one or more plausible benchmark families for the user's intended task.

A request is UNSPECIFIC if the core task is still too vague, too broad, or too ambiguous to retrieve relevant benchmarks reliably.

## What to determine
1. Whether the user's intended task is clear enough.
2. The intended input modalities.
3. The intended output modalities.
4. Whether clarification is needed before benchmark retrieval.

## Decision rule
Set is_specific = true when:
- The core task/action is clear.
- At least one plausible input modality can be identified from the request or normal task defaults.
- At least one plausible output modality can be identified from the request or normal task defaults.
- The request is narrow enough that relevant benchmark descriptions could plausibly be retrieved.

Set is_specific = false when:
- The user has not actually described the task to perform.
- The request is generic or meta, e.g. "recommend a model", "best LLM", "good benchmark".
- The core task is materially ambiguous.
- Input or output modality is materially ambiguous and affects benchmark retrieval.
- Multiple benchmark families are equally plausible because the request lacks the main task intent.

## Modality defaults
Infer obvious default modalities when reasonable. Examples:
- summarize / classify / extract / translate / chat / query -> usually text input, text output
- OCR / image classification / captioning -> usually image input, text output
- questions / query AND photos / images -> text AND image input, usually text output
- speech transcription -> usually audio input, text output

Do NOT mark a request as unspecific only because the user omitted an obvious default modality.

## Clarification scope
If clarification is needed, ask only about:
- the actual use case or task
- missing input modalities: {MODALITY_VALUES}
- missing output modalities: {MODALITY_VALUES}

## Strict rules
You MUST NOT ask about or mention:
- open-source vs commercial preferences
- cost or pricing sensitivity
- speed or latency requirements
- general benchmark performance
- model names or providers
- specific output formatting
- whether the user will provide input as files or via a link

## Output rules
- Be conservative in asking clarifying questions.
- Prefer inferred default modalities over unnecessary clarification.
- If is_specific is true, clarification_needed must be empty.
- If is_specific is false, clarification_needed must contain 1-3 short questions.
- Return only data that matches the schema."""


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
    first_ui_mismatch = ui_mismatch and not ui_mismatch_hinted

    if not response.is_specific and clarification_count >= 3:
        # trials exceeded
        msg = (
            "I've asked for clarification multiple times, but I'm still unable to understand "
            "your request properly. Please start over and be as specific as possible:\n"
            f"{HINTS_MSG}"
        )
        logs.append(
            (f"Intent Validator: Clarification limit exceeded (count={clarification_count}).")
        )
        state_update["clarification_limit_exceeded"] = True
        state_update["messages"] = [AIMessage(content=msg)]  # type: ignore
    elif not response.is_specific or first_ui_mismatch:
        state_update["clarification_count"] = clarification_count + 1
        msg = ""
        if not response.is_specific:
            # Format the list of clarification questions into a single cohesive message
            if len(response.clarification_needed) == 0:
                # Shouldn't happen since LLM was instructed to provide 1 entry, but *can* happen
                msg += f"This is too vague. Please be as specific as possible:\n{HINTS_MSG}"
            elif len(response.clarification_needed) == 1:
                msg += response.clarification_needed[0]
            else:
                msg += "Please clarify the following points:\n" + "\n".join(
                    f"- {q}" for q in response.clarification_needed
                )
            logs.append(f"Intent Validator: Response not specific:\n'''{msg}'''")
            msg += "\n\n"

        if first_ui_mismatch:
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
            response.is_specific = False  # patch it to trigger iteration of node
        else:
            # consecutive ui mismatch -> only soft hint
            msg += "I've mentioned that already, but want to make sure your modality filter "
            msg += "is set correctly.\nYour task description currently indicates the following "
            msg += "modalities:\n"
            msg += f"- input: {response.intended_input_modalities}\n"
            msg += f"- output: {response.intended_output_modalities}\n\n"
            msg += "This currently conflicts with your selection in the UI. Eventually the "
            msg += "UI filters will take precedence and determine the recommendations you get."
            logs.append(
                (
                    "Intent Validator: Consecutive modality-mismatch.\nMissing UI input: "
                    f"{ui_missing_input}; and output: {ui_missing_output}.\n"
                    "Overspec UI input: {ui_overspec_input}; and output: {ui_overspec_output}"
                )
            )

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
        " | logs_appended=%d",
        # " | reasoning=%r",
        getattr(response, "is_specific", None),
        state_update.get("clarification_count", state.get("clarification_count", 0)),
        state_update.get("clarification_limit_exceeded", False),
        getattr(response, "intended_input_modalities", None),
        getattr(response, "intended_output_modalities", None),
        len(state_update.get("messages", [])),
        len(state_update.get("logs", [])),
        # getattr(response, "reasoning", "")[:120],
    )
    return state_update
