"""
Unit tests for the Intent Validator node (Req 2.3 Node 1).

The node calls an LLM via settings.make_llm() internally, so tests pass a mock
Settings object to return controlled IntentExtraction objects and verify state_update contents.

Tests cover:
1. LLM returns is_specific=False  → clarification message + count increment
2. LLM returns is_specific=False at limit (count >= 3) → limit-exceeded flag
3. LLM returns is_specific=True, modalities match UI  → no messages returned
4. LLM returns is_specific=True, modalities conflict UI → clarification message
5. Clarification question formatting (0, 1, many questions)
"""

from typing import cast
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage

from llm_compass.agentic_core.nodes.validate_intent import validate_intent_node
from llm_compass.agentic_core.schemas.validate_intent import IntentExtraction
from llm_compass.agentic_core.state import AgentState
from llm_compass.common.schemas import Constraints
from llm_compass.common.types import Modality
from llm_compass.config import Settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(
    messages: list | None = None,
    clarification_count: int = 0,
    constraints: Constraints | None = None,
) -> AgentState:
    """Minimal AgentState dict for the validate_intent_node."""
    return cast(AgentState, {
        "messages": messages or [HumanMessage(content="test query")],
        "clarification_count": clarification_count,
        "constraints": constraints or Constraints(min_context_window=0),
    })


def _make_response(
    is_specific: bool,
    input_modalities: list[Modality] | None = None,
    output_modalities: list[Modality] | None = None,
    clarification_needed: list[str] | None = None,
) -> IntentExtraction:
    return IntentExtraction(
        reasoning="test reasoning",
        is_specific=is_specific,
        intended_input_modalities=input_modalities or (["text"] if is_specific else []),
        intended_output_modalities=output_modalities or (["text"] if is_specific else []),
        clarification_needed=clarification_needed or ([] if is_specific else ["Please clarify your task."]),
    )


def _make_settings(response: IntentExtraction) -> MagicMock:
    """Returns a mock Settings whose make_llm returns an LLM that yields *response*."""
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = response
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured
    mock_settings = MagicMock(spec=Settings)
    mock_settings.make_llm.return_value = mock_llm
    return mock_settings


# ---------------------------------------------------------------------------
# 1. Not specific → clarification requested
# ---------------------------------------------------------------------------

class TestNotSpecific:

    def test_appends_ai_message(self):
        response = _make_response(is_specific=False, clarification_needed=["What is your use case?"])
        result = validate_intent_node(_make_state(), settings=_make_settings(response))

        assert "messages" in result
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)  # type: ignore[arg-type]
        assert "What is your use case?" in result["messages"][0].content  # type: ignore[union-attr]

    def test_increments_clarification_count(self):
        response = _make_response(is_specific=False, clarification_needed=["Clarify please."])
        result = validate_intent_node(_make_state(clarification_count=1), settings=_make_settings(response))

        assert result["clarification_count"] == 2

    def test_intent_extraction_always_returned(self):
        response = _make_response(is_specific=False, clarification_needed=["Clarify please."])
        result = validate_intent_node(_make_state(), settings=_make_settings(response))

        assert result["intent_extraction"] is response

    def test_single_clarification_question_no_bullet(self):
        """A single question should be returned as-is, not wrapped in a list."""
        question = "What data modality do you need?"
        response = _make_response(is_specific=False, clarification_needed=[question])
        result = validate_intent_node(_make_state(), settings=_make_settings(response))

        msg_content = result["messages"][0].content  # type: ignore[union-attr]
        assert msg_content == question

    def test_multiple_questions_formatted_as_list(self):
        questions = ["What is your input?", "What is your output?"]
        response = _make_response(is_specific=False, clarification_needed=questions)
        result = validate_intent_node(_make_state(), settings=_make_settings(response))

        msg_content = result["messages"][0].content  # type: ignore[union-attr]
        assert "Please clarify the following points:" in msg_content
        for q in questions:
            assert q in msg_content

    def test_zero_clarification_questions_fallback_message(self):
        """Edge case: LLM returns is_specific=False but empty clarification list."""
        response = _make_response(is_specific=False, clarification_needed=[])
        result = validate_intent_node(_make_state(), settings=_make_settings(response))

        msg_content = result["messages"][0].content  # type: ignore[union-attr]
        assert msg_content == "Please clarify your task."

    def test_log_entry_added(self):
        response = _make_response(is_specific=False, clarification_needed=["Clarify."])
        result = validate_intent_node(_make_state(), settings=_make_settings(response))

        assert "logs" in result
        assert any("Intent Validator" in log for log in result["logs"])


# ---------------------------------------------------------------------------
# 2. Clarification limit exceeded (count >= 3 and not specific)
# ---------------------------------------------------------------------------

class TestClarificationLimit:

    def test_limit_exceeded_sets_flag(self):
        response = _make_response(is_specific=False, clarification_needed=["Clarify."])
        result = validate_intent_node(_make_state(clarification_count=3), settings=_make_settings(response))

        assert result.get("clarification_limit_exceeded") is True

    def test_limit_exceeded_appends_message(self):
        response = _make_response(is_specific=False, clarification_needed=["Clarify."])
        result = validate_intent_node(_make_state(clarification_count=3), settings=_make_settings(response))

        assert "messages" in result
        assert isinstance(result["messages"][0], AIMessage)  # type: ignore[arg-type]

    def test_limit_not_triggered_when_specific(self):
        """count >= 3 but LLM says is_specific=True → no limit flag."""
        response = _make_response(is_specific=True)
        result = validate_intent_node(_make_state(clarification_count=3), settings=_make_settings(response))

        assert result.get("clarification_limit_exceeded") is not True

    def test_limit_not_triggered_at_count_2(self):
        """count=2 → still under limit, normal clarification flow."""
        response = _make_response(is_specific=False, clarification_needed=["Clarify."])
        result = validate_intent_node(_make_state(clarification_count=2), settings=_make_settings(response))

        assert result.get("clarification_limit_exceeded") is not True
        assert result["clarification_count"] == 3


# ---------------------------------------------------------------------------
# 3. Specific + modalities match UI constraints → clean pass
# ---------------------------------------------------------------------------

class TestSpecificNoConflict:

    def test_no_messages_appended(self):
        response = _make_response(
            is_specific=True,
            input_modalities=["text"],
            output_modalities=["text"],
        )
        constraints = Constraints(min_context_window=0, modality_input=["text"], modality_output=["text"])
        result = validate_intent_node(_make_state(constraints=constraints), settings=_make_settings(response))

        assert "messages" not in result

    def test_intent_extraction_returned(self):
        response = _make_response(is_specific=True, input_modalities=["text"], output_modalities=["text"])
        constraints = Constraints(min_context_window=0, modality_input=["text"], modality_output=["text"])
        result = validate_intent_node(_make_state(constraints=constraints), settings=_make_settings(response))

        assert result["intent_extraction"] is response

    def test_no_logs_on_clean_pass(self):
        response = _make_response(is_specific=True, input_modalities=["text"], output_modalities=["text"])
        constraints = Constraints(min_context_window=0, modality_input=["text"], modality_output=["text"])
        result = validate_intent_node(_make_state(constraints=constraints), settings=_make_settings(response))

        assert "logs" not in result


# ---------------------------------------------------------------------------
# 4. Specific + modality conflicts with UI constraints
# ---------------------------------------------------------------------------

class TestModalityConflict:

    def test_missing_input_modality_triggers_clarification(self):
        """LLM says image input needed, but UI only has text → mismatch."""
        response = _make_response(
            is_specific=True,
            input_modalities=["image"],
            output_modalities=["text"],
        )
        constraints = Constraints(min_context_window=0, modality_input=["text"], modality_output=["text"])
        result = validate_intent_node(_make_state(constraints=constraints), settings=_make_settings(response))

        assert "messages" in result
        assert isinstance(result["messages"][0], AIMessage)  # type: ignore[arg-type]

    def test_overspec_input_modality_triggers_clarification(self):
        """UI has text+image, but LLM only detects text input → overspec."""
        response = _make_response(
            is_specific=True,
            input_modalities=["text"],
            output_modalities=["text"],
        )
        constraints = Constraints(min_context_window=0, modality_input=["text", "image"], modality_output=["text"])
        result = validate_intent_node(_make_state(constraints=constraints), settings=_make_settings(response))

        assert "messages" in result

    def test_mismatch_message_mentions_modalities(self):
        response = _make_response(
            is_specific=True,
            input_modalities=["image"],
            output_modalities=["text"],
        )
        constraints = Constraints(min_context_window=0, modality_input=["text"], modality_output=["text"])
        result = validate_intent_node(_make_state(constraints=constraints), settings=_make_settings(response))

        msg = result["messages"][0].content  # type: ignore[union-attr]
        assert "modali" in msg.lower()

    def test_mismatch_patches_is_specific_false(self):
        response = _make_response(
            is_specific=True,
            input_modalities=["audio"],
            output_modalities=["text"],
        )
        constraints = Constraints(min_context_window=0, modality_input=["text"], modality_output=["text"])
        result = validate_intent_node(_make_state(constraints=constraints), settings=_make_settings(response))

        # The node patches response.is_specific = False
        assert result["intent_extraction"].is_specific is False  # type: ignore[union-attr]

    def test_mismatch_log_entry_added(self):
        response = _make_response(
            is_specific=True,
            input_modalities=["video"],
            output_modalities=["text"],
        )
        constraints = Constraints(min_context_window=0, modality_input=["text"], modality_output=["text"])
        result = validate_intent_node(_make_state(constraints=constraints), settings=_make_settings(response))

        assert "logs" in result
        assert any("Modality-mismatch" in log for log in result["logs"])
