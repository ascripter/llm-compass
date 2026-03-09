"""
Structured output schemas for Req 2.3
- Node 1: Qualitative intent validation and extraction of intended modalities
"""

from typing import Literal, Self
from pydantic import BaseModel, Field, ConfigDict, model_validator
from pydantic.json_schema import SkipJsonSchema

from llm_compass.common.types import Modality, MODALITY_VALUES



class IntentExtraction(BaseModel):
    """Analyzes a user's task to extract intended input/output modalities
    and determines if the request is specific enough to proceed.
    """

    # Field ordering matters for LLM reasoning
    # 1. Reasoning comes FIRST so the model thinks before judging
    reasoning: str = Field(
        description=(
            "Briefly explain your thought process. Analyze what the user is asking, identify "
            "potential inputs and outputs, and assess if the request lacks critical details."
        )
    )

    # 2. Then the model commits to a boolean based on its reasoning
    is_specific: bool = Field(
        description=(
            "True only if the chat history has enough detail to describe the intended LLM usage "
            "AND derive modality volumes for input and output. False if chat history is vague "
            "or lacking specifics."
        )
    )

    # 3. Then it extracts the modalities
    intended_input_modalities: list[Literal[Modality]] = Field(
        description="The data modalities the user intends to provide as INPUT to the model."
    )
    intended_output_modalities: list[Literal[Modality]] = Field(
        description="The data modalities the user expects the model to generate as OUTPUT."
    )
    clarification_needed: list[str] = Field(
        default_factory=list,
        description=(
            "If is_specific is False, provide a list of distinct questions explaining exactly "
            "what information is missing. Focus on the following aspects primarily:\n"
            "- Is the actual use case clear?\n"
            "- Are input and output modalities clear?\n"
            "Leave empty if is_specific is True, otherwise you MUST add least add one entry."
        ),
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "reasoning": (
                        "The user gave a highly detailed request around 'making videos from "
                        "text descriptions and style images'. The input clearly consists of "
                        "text and image modalities. The expected output is a video. "
                    ),
                    "is_specific": True,
                    "intended_input_modalities": ["text", "image"],
                    "intended_output_modalities": ["video"],
                    "clarification_needed": [],
                },
                {
                    "reasoning": (
                        "The user said to 'summarize things'. They did not specify what 'this' "
                        "is (a text document, an image, an audio or video file), nor did they "
                        "specify what format they want the analysis in. The request is too vague."
                    ),
                    "is_specific": False,
                    "intended_input_modalities": [],
                    "intended_output_modalities": [],
                    "clarification_needed": [
                        (
                            "Your task description is too vague. Can you provide more details "
                            "about what you want to achieve?"
                        ),
                        (
                            "What is the source material you want to summarize? Text, images, "
                            "audio, or video?"
                        ),
                        (
                            "What is the output modality you want for the summary? Most likely "
                            "a summary is a text summary, but it could also be audio or video?"
                        ),
                    ],
                },
            ]
        }
    )