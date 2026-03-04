import streamlit as st

def render_traceability():
    st.subheader("🧠 Traceability View")

    with st.expander("Live Agent Logs", expanded=True):
        for step in st.session_state.trace:
            st.code(step)
