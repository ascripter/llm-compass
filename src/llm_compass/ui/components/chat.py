import streamlit as st


def render_chat(constraints, *, debug: bool = False):
    if debug:
        with st.expander("Active Constraints", expanded=True):
            st.write(constraints)

    st.subheader("💬 LLM Compass")

    if not st.session_state.messages:
        st.info("What task do you want to accomplish?")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Describe your task (e.g., 'RAG on legal documents')")

    return prompt
