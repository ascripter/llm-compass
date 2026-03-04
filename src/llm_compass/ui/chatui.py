import streamlit as st
import pandas as pd
import time
import random

# ==========================
# Page + Theme
# ==========================
st.set_page_config(
    page_title="The Benchmark Analyst",
    page_icon="📊",
    layout="wide"
)

# Force dark + monospace vibe
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: monospace;
}
.stChatMessage {
    background-color: #0f172a10;
}
.metric-card {
    padding: 1rem;
    border-radius: 12px;
    background: #020617;
    border: 1px solid #1e293b;
}
.winner {
    border: 2px solid #22c55e;
}
</style>
""", unsafe_allow_html=True)

st.title("📊 The Benchmark Analyst")
st.caption("Mission Control for Model Selection")

# ==========================
# Session State
# ==========================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "trace" not in st.session_state:
    st.session_state.trace = []

# ==========================
# Sidebar = Setup Panel
# ==========================
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

# ==========================
# Layout
# ==========================
left, right = st.columns([0.55, 0.45])

# ==========================
# Mock Agent
# ==========================
def run_agent(query):
    trace = [
        "> Analyzing intent...",
        "> Searching benchmarks: Legal reasoning, Long context retrieval...",
        "> Found 3 benchmark clusters",
        "> Filtering models based on constraints...",
        "> Calibrating scores..."
    ]

    for t in trace:
        st.session_state.trace.append(t)
        time.sleep(0.3)

    models = [
        {"Model": "Model-A", "Score": 0.92, "Cost": 8, "HumanEval": 85},
        {"Model": "Model-B", "Score": 0.88, "Cost": 3, "HumanEval": None},
        {"Model": "Model-C", "Score": 0.81, "Cost": 2, "HumanEval": 70},
    ]

    df = pd.DataFrame(models)
    return df

# ==========================
# LEFT: Chat + Active Constraints
# ==========================
with left:
    st.subheader("💬 Chat")

    # Active constraints summary
    with st.expander("Active Constraints", expanded=True):
        st.write({
            "Min Context": min_context,
            "Inputs": input_modalities,
            "Outputs": output_modalities,
            "Deployment": deployment,
            "Reasoning": reasoning,
            "Tool Calling": tool_calling,
            "Speed": speed
        })

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Describe your task (e.g., 'I need a model for RAG on legal documents')")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        st.session_state.trace = []

        df = run_agent(prompt)

        # Edge case
        if df.empty:
            assistant_text = "❌ No models found. Try relaxing your constraints."
        else:
            assistant_text = "### ✅ Executive Summary\nModel-A is the best overall performer, while Model-C wins on budget."

        st.session_state.messages.append({"role": "assistant", "content": assistant_text})

        with st.chat_message("assistant"):
            st.markdown(assistant_text)

        st.session_state.results = df

# ==========================
# RIGHT: Traceability + Results
# ==========================
with right:
    st.subheader("🧠 Traceability View")

    with st.expander("Live Agent Logs", expanded=True):
        for t in st.session_state.trace:
            st.code(t)

    st.subheader("📊 Results")

    if "results" in st.session_state:
        df = st.session_state.results.copy()

        df["Est?"] = df["HumanEval"].isna()

        st.dataframe(
            df,
            use_container_width=True
        )

        st.subheader("🏆 Recommendations")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div class="metric-card winner">
            <h4>Performance Winner</h4>
            <p>Model-A</p>
            <button>Copy Name</button>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="metric-card">
            <h4>Budget Winner</h4>
            <p>Model-C</p>
            <button>Copy Name</button>
            </div>
            """, unsafe_allow_html=True)
