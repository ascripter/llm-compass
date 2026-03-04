import streamlit as st
import pandas as pd
import time

from ui.components.sidebar import render_sidebar
from ui.components.chat import render_chat
from ui.components.traceability import render_traceability
from ui.components.tables import render_results


st.set_page_config(
    page_title="The Benchmark Analyst",
    page_icon="📊",
    layout="wide"
)

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

if "messages" not in st.session_state:
    st.session_state.messages = []

if "trace" not in st.session_state:
    st.session_state.trace = []

if "results" not in st.session_state:
    st.session_state.results = None


def run_agent(query):
    trace_steps = [
        "> Analyzing intent...",
        "> Searching benchmarks...",
        "> Found 3 benchmark clusters",
        "> Filtering models based on constraints...",
        "> Calibrating scores..."
    ]

    st.session_state.trace = []
    for step in trace_steps:
        st.session_state.trace.append(step)
        time.sleep(0.3)

    models = [
        {"Model": "Model-A", "Score": 0.92, "Cost": 8, "HumanEval": 85},
        {"Model": "Model-B", "Score": 0.88, "Cost": 3, "HumanEval": None},
        {"Model": "Model-C", "Score": 0.81, "Cost": 2, "HumanEval": 70},
    ]

    return pd.DataFrame(models)


constraints = render_sidebar()

left, right = st.columns([0.55, 0.45])

with left:
    prompt = render_chat(constraints)

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        df = run_agent(prompt)

        if df.empty:
            assistant_text = "❌ No models found. Try relaxing your constraints."
        else:
            assistant_text = "### ✅ Executive Summary\nModel-A is best overall. Model-C wins on budget."

        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_text}
        )

        st.session_state.results = df

with right:
    render_traceability()
    render_results(st.session_state.results)
