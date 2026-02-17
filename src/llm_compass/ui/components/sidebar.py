"""
Req 3.2: Input Interface ("Setup Panel").
"""

import streamlit as st


def render_sidebar() -> dict:
    """
    Renders constraints and returns a dictionary.
    Includes:
    1. Min Context Window (int)
    2. Modality Selectors (multiselect)
    3. Deployment (Radio: Any/Cloud/Local)
    4. Capabilities (Toggles: Reasoning, Tool Calling)
    """
    st.sidebar.header("Constraints")
    # ... implementation of widgets ...
    return {"min_context": ..., "modalities": ..., "deployment": ...}
