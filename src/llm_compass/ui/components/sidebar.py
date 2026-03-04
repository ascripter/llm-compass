import streamlit as st

def render_sidebar():
    with st.sidebar:
        st.header("⚙ Setup")
        st.subheader("Constraints")

        min_context = st.number_input("Min Context", min_value=0, value=0)

        input_modalities = st.multiselect(
            "Input Modalities",
            ["Text", "Image", "Audio", "Video"],
            default=["Text"]
        )

        output_modalities = st.multiselect(
            "Output Modalities",
            ["Text", "Image", "Audio", "Video"],
            default=["Text"]
        )

        deployment = st.segmented_control(
            "Deployment",
            ["Any", "Cloud API", "Local / Open Weights"],
            default="Any"
        )

        reasoning = st.checkbox("Reasoning Model")
        tool_calling = st.checkbox("Tool Calling")

        speed = st.segmented_control(
            "Minimum Speed",
            ["Any/Slow+", "Balanced+", "Fast only"],
            default="Any/Slow+"
        )

        with st.expander("Global Ranking Settings"):
            perf_cost = st.slider("Balanced Profile (Performance ↔ Cost)", 0, 100, 50)
            budget_profile = st.slider("Budget Profile (Performance ↔ Cost)", 0, 100, 20)

    return {
        "Min Context": min_context,
        "Inputs": input_modalities,
        "Outputs": output_modalities,
        "Deployment": deployment,
        "Reasoning": reasoning,
        "Tool Calling": tool_calling,
        "Speed": speed,
        "Perf vs Cost": perf_cost,
        "Budget Profile": budget_profile
    }
