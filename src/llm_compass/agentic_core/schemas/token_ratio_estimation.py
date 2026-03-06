"""
Structured output schemas for Req 2.3
- Node 2 (b): Token ratio estimation based on user intent.
"""

from typing import Literal, Self
from pydantic import BaseModel, Field, ConfigDict, model_validator
from pydantic.json_schema import SkipJsonSchema

from llm_compass.common.types import Modality, MODALITY_VALUES


class ModalityUnits(BaseModel):
    text_word_count: int = Field(
        default=0,
        description="Estimated number of words. E.g., a short query is 20, a long essay is 1000.",
    )
    image_count: float = Field(default=0.0, description="Average number of images involved.")
    audio_minutes: float = Field(default=0.0, description="Minutes of audio.")
    video_minutes: float = Field(default=0.0, description="Minutes of video.")


class TokenRatioEstimation(BaseModel):
    """Estimates the real-world units (words, minutes, counts) for the user's
    intended input and output modalities.
    """

    reasoning: str = Field(
        description=(
            "Briefly explain your thought process for estimating the units. Break down "
            "the user's request and justify the numbers for word count, image count, "
            "audio and video minutes BEFORE providing them."
        )
    )
    input_units: ModalityUnits = Field(description="Estimated units for the user's input.")
    output_units: ModalityUnits = Field(
        description="Estimated units for the model's requested output."
    )

    # Hide calculated ratios from the LLM as they are derived fields
    # all ratio (across input AND output) sum to 1.0000, i.e. represent the total cost fraction
    normalized_input_ratios: SkipJsonSchema[dict[str, float]] = Field(default_factory=dict)
    normalized_output_ratios: SkipJsonSchema[dict[str, float]] = Field(default_factory=dict)

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                # Example 1: Summarize a 10-minute video into a short 100-word paragraph
                {
                    "reasoning": (
                        "The user wants a summary of a 10-minute video. The input is roughly "
                        "10 minutes of video. A standard summary is a short paragraph, "
                        "roughly 100 words. No images or audio are explicitly mentioned."
                    ),
                    "input_units": {
                        "text_word_count": 0,
                        "image_count": 0,
                        "audio_minutes": 0.0,
                        "video_minutes": 10.0,
                    },
                    "output_units": {
                        "text_word_count": 100,
                        "image_count": 0,
                        "audio_minutes": 0.0,
                        "video_minutes": 0.0,
                    },
                },
                # Example 2: Images with descriptions (100 words), expecting a 150-word answer
                {
                    "reasoning": (
                        "The user asked to identify dog breeds from 1 to 4 images and a short "
                        "description of behavioral traits. The input is on average 2.5 images "
                        "and a short text paragraph (approx 100 words). The expected output is "
                        "a short text answer explaining the breed, roughly 150 words."
                    ),
                    "input_units": {
                        "text_word_count": 100,
                        "image_count": 2.5,
                        "audio_minutes": 0.0,
                        "video_minutes": 0.0,
                    },
                    "output_units": {
                        "text_word_count": 150,
                        "image_count": 0,
                        "audio_minutes": 0.0,
                        "video_minutes": 0.0,
                    },
                },
            ]
        }
    )

    @model_validator(mode="after")
    def compute_ratios(self) -> Self:
        # Input ratios calculation
        HEURISTICS = {
            "text": 1.3,  # tokens per word
            "image": 1000.0,  # tokens per image (average)
            "audio": 750.0,  # tokens per minute
            "video": 50000.0,  # tokens per minute
        }

        token_total = 0
        for mode in ("input", "output"):
            units = getattr(self, f"{mode}_units")
            token_total += sum(
                [
                    units.text_word_count * HEURISTICS["text"],
                    units.image_count * HEURISTICS["image"],
                    units.audio_minutes * HEURISTICS["audio"],
                    units.video_minutes * HEURISTICS["video"],
                ]
            )

        # Edge case: LLM guessed 0 for absolutely everything
        if token_total == 0:
            # Default to 100% input text, 0% everything else
            self.normalized_input_ratios = {"text": 0.5, "image": 0.0, "audio": 0.0, "video": 0.0}
            self.normalized_output_ratios = {
                "text": 0.5,
                "image": 0.0,
                "audio": 0.0,
                "video": 0.0,
            }
            return self

        for mode in ("input", "output"):
            units = getattr(self, f"{mode}_units")
            setattr(
                self,
                f"normalized_{mode}_ratios",
                {
                    "text": round((units.text_word_count * HEURISTICS["text"]) / token_total, 4),
                    "image": round((units.image_count * HEURISTICS["image"]) / token_total, 4),
                    "audio": round((units.audio_minutes * HEURISTICS["audio"]) / token_total, 4),
                    "video": round((units.video_minutes * HEURISTICS["video"]) / token_total, 4),
                },
            )

        return self
