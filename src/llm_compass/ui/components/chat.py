import streamlit as st

def render_chat(constraints):
    st.subheader("💬 Chat")

    with st.expander("Active Constraints", expanded=True):
        st.write(constraints)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input(
        "Describe your task (e.g., 'I need a model for RAG on legal documents')"
    )

    return prompt
