"""
Structured output schemas for Req 2.3
- Node 1: Qualitative intent validation and extraction of intended modalities
"""

from enum import StrEnum
from typing import Literal, Self, Optional, get_args
from pydantic import BaseModel, Field, ConfigDict, model_validator
from pydantic.json_schema import SkipJsonSchema

from llm_compass.common.types import Modality, MODALITY_VALUES


class IntentExtraction(BaseModel):
    """Analyzes a user's task to extract intended input/output modalities
    and determines if the request is specific enough to proceed.
    """

    # Field ordering matters for LLM reasoning
    # 1. Reasoning comes FIRST so the model thinks before judging
    # reasoning: str = Field(
    #     description=(
    #         "Brief rationale in 1-2 sentences. State whether the core task is clear "
    #         "enough for benchmark retrieval, and mention only the key missing detail "
    #         "if the request is not specific. Do not provide step-by-step reasoning."
    #     )
    # )

    # 2. Then the model commits to a boolean based on its reasoning
    is_specific: bool = Field(
        description=(
            "True if the current chat history is sufficient to retrieve one or more "
            "plausible benchmark families for the user's intended task. False if the "
            "core task or materially relevant modality information is still too vague."
        )
    )

    # 3. Then it extracts the modalities
    intended_input_modalities: list[Modality] = Field(
        default_factory=list,
        description=(
            "Input modalities the user intends to provide. Infer obvious defaults when "
            "reasonable from the task."
        ),
    )
    intended_output_modalities: list[Modality] = Field(
        default_factory=list,
        description=(
            "Output modalities the user expects. Infer obvious defaults when reasonable "
            "from the task."
        ),
    )
    clarification_needed: list[str] = Field(
        default_factory=list,
        description=(
            "1-3 short clarification questions only if is_specific is False. Ask only "
            "about the task/use case, input modalities, or output modalities. Leave "
            "empty if is_specific is True."
        ),
    )

    @model_validator(mode="after")
    def validate_consistency(self):
        if self.is_specific and self.clarification_needed:
            raise ValueError("clarification_needed must be empty when is_specific is True")
        if not self.is_specific:
            if not self.clarification_needed:
                raise ValueError(
                    "clarification_needed must contain 1-3 questions when is_specific is False"
                )
            if len(self.clarification_needed) > 3:
                raise ValueError("clarification_needed must contain at most 3 questions")
        return self

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    # "reasoning": (
                    #     "The user described a concrete task: extracting structured data "
                    #     "from invoices. The task is clear enough for benchmark retrieval."
                    # ),
                    "is_specific": True,
                    "intended_input_modalities": ["image", "text"],
                    "intended_output_modalities": ["text"],
                    "clarification_needed": [],
                },
                {
                    # "reasoning": (
                    #     "The request is too generic because the user asked for a model "
                    #     "recommendation without describing the actual task to perform."
                    # ),
                    "is_specific": False,
                    "intended_input_modalities": [],
                    "intended_output_modalities": [],
                    "clarification_needed": [
                        "What task do you want the model to perform? A concise description suffices",
                        "What input modality will you provide: text, image, audio, or video?",
                        "What output modality do you expect from the model?",
                    ],
                },
            ]
        }
    )
