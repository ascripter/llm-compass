"""
Req 3.3.B: Structured Response rendering.
"""

import streamlit as st
import pandas as pd


def render_comparison_table(ranked_data: list):
    """
    Renders the interactive comparison table.
    Columns: Model | Rank Score | Cost | Benchmarks | Est?
    """
    df = pd.DataFrame(ranked_data)
    st.dataframe(df, use_container_width=True)
