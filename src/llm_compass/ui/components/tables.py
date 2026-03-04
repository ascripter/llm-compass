import streamlit as st

def render_results(df):
    st.subheader("📊 Results")

    if df is None:
        st.info("Run a query to see benchmark results.")
        return

    df = df.copy()
    df["Est?"] = df["HumanEval"].isna()

    st.dataframe(df, use_container_width=True)

    st.subheader("🏆 Recommendations")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="metric-card winner">
        <h4>Performance Winner</h4>
        <p>Model-A</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
        <h4>Budget Winner</h4>
        <p>Model-C</p>
        </div>
        """, unsafe_allow_html=True)
