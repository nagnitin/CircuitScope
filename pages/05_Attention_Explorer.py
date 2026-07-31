"""
pages/05_Attention_Explorer.py
==============================
Interactive Attention Head Explorer & Matrix Visualizer for GPT-2 Small.
"""

import streamlit as st
import pandas as pd
import numpy as np

from components.charts import build_head_importance_heatmap, build_attention_matrix_heatmap
from components.cards import render_callout, render_head_role_pill
from components.model_runner import run_live_inference
from components.utils import load_css, load_exp_csv

load_css()

st.title("🎯 Attention Head Explorer & Matrix Visualizer")
st.markdown("### Interactive 144-Head Ablation Grid & Live Token-to-Token Attention Matrices")

# Load Head Ablation Data
df_heads = load_exp_csv("04_head_ablation", "head_ablation.csv")

if df_heads is None:
    df_heads = pd.DataFrame({
        "head_label": ["L8H6", "L8H10", "L5H5", "L7H9", "L6H9", "L3H0", "L0H10", "L1H10"],
        "layer": [8, 8, 5, 7, 6, 3, 0, 1],
        "head": [6, 10, 5, 9, 9, 0, 10, 10],
        "importance": [0.34316, 0.281559, 0.241703, 0.224678, 0.123334, 0.123044, 0.121647, 0.117772],
        "head_type": ["Name Mover", "Name Mover", "Name Mover", "Name Mover", "Helper", "Helper", "Helper", "Helper"]
    })

# 1. Top 12x12 Head Importance Heatmap
st.header("1. Global 12×12 Attention Head Importance Grid")
st.plotly_chart(build_head_importance_heatmap(df_heads), use_container_width=True)

st.markdown("---")

# 2. Interactive Head & Prompt Selection
st.header("2. Live Attention Matrix Inspector")

prompt_text = st.text_input(
    "Prompt for Attention Matrix:",
    value="When John and Mary went to the store, John gave the book to"
)

with st.spinner("Computing Attention Matrix..."):
    results = run_live_inference(prompt_text, target_name="Mary", distractor_name="John")

str_tokens = results["tokens"]
attn_patterns = results["attn_patterns"] # (12, 12, seq, seq)

col_h1, col_h2, col_h3 = st.columns(3)

with col_h1:
    selected_layer = st.slider("Select Layer Index:", min_value=0, max_value=11, value=8)
with col_h2:
    selected_head = st.slider("Select Head Index:", min_value=0, max_value=11, value=6)
with col_h3:
    head_label = f"L{selected_layer}H{selected_head}"
    st.markdown(f"**Selected Head:** `{head_label}`")
    
    # Determine Role and Importance
    h_row = df_heads[(df_heads["layer"] == selected_layer) & (df_heads["head"] == selected_head)] if "layer" in df_heads.columns else df_heads[df_heads["head_label"] == head_label]
    if not h_row.empty:
        role = str(h_row.iloc[0].get("head_type", "Name Mover" if head_label in ["L8H6", "L8H10", "L5H5", "L7H9"] else "Neutral"))
        imp = float(h_row.iloc[0]["importance"])
    else:
        role = "Name Mover" if head_label in ["L8H6", "L8H10", "L5H5", "L7H9"] else "Neutral"
        imp = 0.3432 if head_label == "L8H6" else 0.05

    st.markdown(f"**Role:** {render_head_role_pill(role)}", unsafe_allow_html=True)
    st.markdown(f"**Importance Score:** `{imp:.4f}`")

# 3. Interactive N_seq x N_seq Attention Matrix Plotly Heatmap
selected_matrix = attn_patterns[selected_layer, selected_head]
fig_attn = build_attention_matrix_heatmap(
    selected_matrix,
    str_tokens,
    title=f"Attention Pattern Matrix for Head {head_label} ({role})"
)
st.plotly_chart(fig_attn, use_container_width=True)

# 4. Research Mode Details
if st.session_state.get("research_mode", False):
    st.markdown("---")
    st.header(f"🔬 Research Mode: Full Layer {selected_layer} Attention Grid")
    st.caption(f"Showing query-key attention distribution for all 12 heads in Layer {selected_layer}")
    
    cols = st.columns(4)
    for h_idx in range(12):
        with cols[h_idx % 4]:
            st.markdown(f"**Head L{selected_layer}H{h_idx}**")
            sub_mat = attn_patterns[selected_layer, h_idx]
            fig_sub = px.imshow(sub_mat, color_continuous_scale="Viridis", labels=dict(x="Key", y="Query"))
            fig_sub.update_layout(height=180, margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False)
            st.plotly_chart(fig_sub, use_container_width=True)
