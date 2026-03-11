"""Extendable type definitions for model attributes, state variables or API schemas."""

from typing import Literal, get_args


type ModelType = Literal["base", "instruct", "thinking", "generator"]
type Modality = Literal["text", "image", "audio", "video"]  # Extendable for future modalities
# NOTE on future Modality extension: agentic_core/nodes/ranking ONLY works as long as
# enum strings are not substrings of each other. Otherwise modality filter needs to be adapted
type SpeedClass = Literal["fast", "medium", "slow"]  # For categorizing model inference speed
type ReasoningType = Literal["none", "standard", "native cot"]  # For categorizing reasoning
type ToolCalling = Literal["none", "standard", "agentic"]  # For categorizing tool calling

MODEL_TYPE_VALUES: tuple[str, ...] = get_args(ModelType.__value__)
MODALITY_VALUES: tuple[str, ...] = get_args(Modality.__value__)
SPEED_CLASS_VALUES: tuple[str, ...] = get_args(SpeedClass.__value__)
REASONING_TYPE_VALUES: tuple[str, ...] = get_args(ReasoningType.__value__)
TOOL_CALLING_VALUES: tuple[str, ...] = get_args(ToolCalling.__value__)

type DeploymentType = Literal["any", "cloud", "local"]
DEPLOYMENT_TYPE_VALUES: tuple[str, ...] = get_args(DeploymentType.__value__)
