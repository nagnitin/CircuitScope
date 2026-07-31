"""
pages/05_Attention_Explorer.py
==============================
Interactive Attention Head Explorer & Importance Grid for CircuitScope.
"""

import streamlit as st
import pandas as pd
import numpy as np
from components.charts import build_head_importance_heatmap
from components.cards import render_callout, render_head_role_pill
from components.utils import load_css, load_exp_csv

load_css()

st.title("🎯 Attention Head Explorer & Importance Grid")
st.markdown("### Systematic 144-Head Ablation Sweep (GPT-2 Small)")

# Load Head Ablation Data
df_heads = load_exp_csv("04_head_ablation", "head_ablation.csv")

if df_heads is None:
    st.warning("⚠️ Could not load outputs/04_head_ablation/results/head_ablation.csv. Using fallback display.")
    # Build fallback dataframe
    df_heads = pd.DataFrame({
        "head_label": ["L8H6", "L8H10", "L5H5", "L7H9", "L6H9", "L3H0", "L0H10", "L1H10", "L5H9", "L9H7", "L7H3", "L10H0"],
        "layer": [8, 8, 5, 7, 6, 3, 0, 1, 5, 9, 7, 10],
        "head": [6, 10, 5, 9, 9, 0, 10, 10, 9, 7, 3, 0],
        "importance": [0.34316, 0.281559, 0.241703, 0.224678, 0.123334, 0.123044, 0.121647, 0.117772, 0.117301, 0.108209, 0.092756, 0.08166],
        "head_type": ["Name Mover", "Name Mover", "Name Mover", "Name Mover", "Helper", "Helper", "Helper", "Helper", "Helper", "Helper", "Helper", "Helper"]
    })

# Render Plotly Heatmap
st.plotly_chart(build_head_importance_heatmap(df_heads), use_container_width=True)

st.markdown("---")

# Per-Head Inspector
st.header("🔍 Individual Head Inspector")

col_sel1, col_sel2 = st.columns(2)

with col_sel1:
    selected_head_label = st.selectbox(
        "Select Head to Inspect:",
        options=df_heads["head_label"].tolist() if "head_label" in df_heads.columns else ["L8H6", "L8H10", "L5H5", "L7H9"]
    )

row_head = df_heads[df_heads["head_label"] == selected_head_label].iloc[0]

with col_sel2:
    role = str(row_head.get("head_type", "Name Mover" if selected_head_label in ["L8H6", "L8H10", "L5H5", "L7H9"] else "Neutral"))
    st.markdown(f"**Head Category:** {render_head_role_pill(role)}", unsafe_allow_html=True)
    st.markdown(f"**Importance Score:** `{float(row_head['importance']):.4f}` Logit Diff Drop")

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Layer Index", f"Layer {row_head.get('layer', selected_head_label[1:2])}")
with c2:
    st.metric("Head Index", f"Head {row_head.get('head', selected_head_label[3:])}")
with c3:
    rank = df_heads[df_heads["head_label"] == selected_head_label].index[0] + 1 if selected_head_label in df_heads["head_label"].values else 1
    st.metric("Importance Rank", f"#{rank} / 144")

st.markdown("<br>", unsafe_allow_html=True)

# Top 15 Heads Table
st.header("📋 Top 15 Ranked Attention Heads")
df_top15 = df_heads.head(15)[["head_label", "importance", "head_type"]] if "head_type" in df_heads.columns else df_heads.head(15)
st.dataframe(df_top15, use_container_width=True)

render_callout(
    title="Name Mover Heads Identification",
    text="The top 4 heads by importance score — <strong>L8H6</strong> (0.3432), <strong>L8H10</strong> (0.2816), <strong>L5H5</strong> (0.2417), and <strong>L7H9</strong> (0.2247) — form the primary <em>Name Mover group</em> that directly writes Indirect Object names to the output residual stream.",
    category="success",
    icon="★"
)
