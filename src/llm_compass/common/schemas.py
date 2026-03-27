"""
Shared Pydantic schemas used across API and agentic core layers.
This module provides a single source of truth for constraint definitions.
"""

from typing import List, Optional

from pydantic import BaseModel, Field

from .types import Modality, ReasoningType, ToolCalling, SpeedClass, DeploymentType


class Constraints(BaseModel):
    """Common constraint definition used by API and graph.

    Maps 1:1 to UI constraint panel (Req 3.2).
    """

    min_context_window: int = Field(default=0, ge=0)
    modality_input: List[Modality] = Field(default_factory=lambda: ["text"])
    modality_output: List[Modality] = Field(default_factory=lambda: ["text"])
    deployment: DeploymentType = "any"
    min_reasoning_type: Optional[ReasoningType] = None
    min_tool_calling: Optional[ToolCalling] = None
    min_speed_class: Optional[SpeedClass] = None
    balanced_perf_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    budget_perf_weight: float = Field(default=0.2, ge=0.0, le=1.0)
