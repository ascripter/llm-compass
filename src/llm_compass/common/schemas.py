"""
Shared Pydantic schemas used across API and agentic core layers.
This module provides a single source of truth for constraint definitions.
"""

from typing import List, Optional

from pydantic import BaseModel, Field

from ..data.models import ReasoningType
from ..api.schemas.common import DeploymentType, Modality, SpeedClass


class Constraints(BaseModel):
    """Common constraint definition used by API and graph.

    Maps 1:1 to UI constraint panel (Req 3.2).
    """

    min_context_window: int = Field(0, ge=0)
    modality_input: List[Modality] = Field(default_factory=lambda: ["text"])
    modality_output: List[Modality] = Field(default_factory=lambda: ["text"])
    deployment: DeploymentType = DeploymentType.ANY
    reasoning_type: Optional[ReasoningType] = None
    require_tool_calling: bool = False
    min_speed_class: Optional[SpeedClass] = None
