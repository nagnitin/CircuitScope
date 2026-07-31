"""
pages/04_Prompt_Playground.py
=============================
Interactive Live Prompt Playground, Logit Lens, Token Attribution, and What-If Explorer for GPT-2 Small.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from components.cards import render_callout
from components.charts import build_logit_lens_curve, build_token_attribution_chart
from components.model_runner import run_live_inference
from components.utils import load_css

load_css()

st.title("⚡ Live Prompt Playground & Reasoning Inspector")
st.markdown("### Test Custom IOI Prompts, Track Layer Evolution, Token Attribution & What-If Experiments")

# 1. Prompt Selection & Configuration
st.header("1. Enter or Select an IOI Prompt")

col_sel, col_t1, col_t2 = st.columns([2, 1, 1])

with col_sel:
    preset = st.selectbox(
        "Choose Preset Template:",
        [
            "When John and Mary went to the store, John gave the book to",
            "When Alice and Bob visited the restaurant, Bob offered the package to",
            "John gave the book to Mary because she wanted it for",
            "Sarah met James at the café. She bought a gift for",
            "Custom Prompt Input"
        ]
    )

if preset == "Custom Prompt Input":
    default_prompt = "When David and Emma went to the park, David gave the ball to"
    default_target = "Emma"
    default_s = "David"
else:
    default_prompt = preset
    if "John and Mary" in preset and "John gave" in preset:
        default_target = "Mary"
        default_s = "John"
    elif "Alice and Bob" in preset:
        default_target = "Alice"
        default_s = "Bob"
    elif "because she" in preset:
        default_target = "Mary"
        default_s = "John"
    elif "Sarah met James" in preset:
        default_target = "James"
        default_s = "Sarah"
    else:
        default_target = "Emma"
        default_s = "David"

prompt_text = st.text_area("Prompt Text:", value=default_prompt, height=80)

with col_t1:
    target_name = st.text_input("Target IO Name:", value=default_target)
with col_t2:
    distractor_name = st.text_input("Distractor S Name:", value=default_s)

# Run Live Forward Pass
with st.spinner("Running GPT-2 Small Live Inference..."):
    results = run_live_inference(prompt_text, target_name=target_name, distractor_name=distractor_name)

st.markdown("---")

# 2. Tokenization & Token Pill Tags
st.header("2. Tokenization & Token Roles")
str_tokens = results["tokens"]

token_html = ""
for t in str_tokens:
    t_clean = t.strip()
    if t_clean == target_name.strip():
        token_html += f'<span class="pill-tag pill-name-mover">{t} [IO]</span> '
    elif t_clean == distractor_name.strip():
        token_html += f'<span class="pill-tag pill-s-inhibition">{t} [S]</span> '
    else:
        token_html += f'<span class="pill-tag pill-neutral">{t}</span> '

st.markdown(f'<div style="margin: 10px 0;">{token_html}</div>', unsafe_allow_html=True)

st.markdown("---")

# Tabbed Demonstrations
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Predictions & Logits",
    "🔍 Logit Lens Layer Progression",
    "🧲 Token Attribution",
    "🔀 What-If Name Playground"
])

# Tab 1: Top 10 Predictions
with tab1:
    st.subheader("Top-10 Next Token Predictions (GPT-2 Small)")
    st.caption("Live computed predictions, probabilities, and unnormalized logits at the final prompt token position.")
    
    df_top = results["top_tokens"]
    
    col_p1, col_p2 = st.columns([2, 1])
    with col_p1:
        st.dataframe(df_top[["Rank", "Token", "Probability", "Logit"]], use_container_width=True)
    with col_p2:
        top1_row = df_top.iloc[0]
        st.metric("Top-1 Predicted Token", top1_row["Token"], delta=f"Confidence: {top1_row['Probability']}")
        
        # Check target vs distractor
        target_row = df_top[df_top["Token"].str.contains(target_name.strip())]
        if not target_row.empty:
            st.metric(f"Target IO ({target_name}) Rank", f"#{target_row.iloc[0]['Rank']}", delta=f"Logit: {target_row.iloc[0]['Logit']}")

# Tab 2: Logit Lens
with tab2:
    st.subheader("Logit Lens: Layer-by-Layer Prediction Evolution")
    st.markdown("Projecting intermediate residual stream vectors directly onto vocabulary logits reveals where the model resolves the indirect object name:")
    
    df_lens = results["logit_lens"]
    st.plotly_chart(build_logit_lens_curve(df_lens), use_container_width=True)
    
    with st.expander("📋 View Layer-by-Layer Prediction Table"):
        st.dataframe(df_lens[["Layer", "Top_Prediction", "Top_Probability", "Logit_Diff", "Target_Logit", "Distractor_Logit"]], use_container_width=True)

# Tab 3: Token Attribution
with tab3:
    st.subheader("Direct Token Attribution Analysis")
    st.markdown("Measures how much each prompt token's residual state contributes to the final logit difference $logit(\\text{IO}) - logit(\\text{S})$:")
    
    fig_attr = build_token_attribution_chart(results["token_attributions"])
    st.plotly_chart(fig_attr, use_container_width=True)

# Tab 4: What-If Name Swap Playground
with tab4:
    st.subheader("🔀 What-If Name Swap Experiment")
    st.markdown("Modify subject and object names to test how prediction and logit diff change in real time:")
    
    col_w1, col_w2 = st.columns(2)
    with col_w1:
        new_target = st.text_input("New Target Name:", value="Emma" if target_name != "Emma" else "Sophia")
    with col_w2:
        new_s = st.text_input("New Distractor Name:", value="Lucas" if distractor_name != "Lucas" else "Oliver")
        
    swapped_prompt = prompt_text.replace(target_name, new_target).replace(distractor_name, new_s)
    st.code(f"Swapped Prompt: '{swapped_prompt}'", language="text")
    
    if st.button("⚡ Recompute Swapped Prediction"):
        with st.spinner("Running Live Inference on Swapped Prompt..."):
            swap_results = run_live_inference(swapped_prompt, target_name=new_target, distractor_name=new_s)
            
        c_orig, c_swap = st.columns(2)
        with c_orig:
            st.markdown("**Original Prompt:**")
            st.metric("Original Top Token", results["top_tokens"].iloc[0]["Token"], delta=f"Prob: {results['top_tokens'].iloc[0]['Probability']}")
            st.metric("Original Logit Diff", f"{results['logit_lens'].iloc[-1]['Logit_Diff']:+.2f}")
            
        with c_swap:
            st.markdown("**Swapped Prompt:**")
            st.metric("Swapped Top Token", swap_results["top_tokens"].iloc[0]["Token"], delta=f"Prob: {swap_results['top_tokens'].iloc[0]['Probability']}")
            st.metric("Swapped Logit Diff", f"{swap_results['logit_lens'].iloc[-1]['Logit_Diff']:+.2f}")
            
        render_callout(
            title="Circuit Flexibility",
            text=f"The circuit successfully adapts to new names (<em>{new_target}</em> vs <em>{new_s}</em>), demonstrating abstract structural reasoning.",
            category="success"
        )

# 5. Research Mode Section (If Enabled)
if st.session_state.get("research_mode", False):
    st.markdown("---")
    st.header("🔬 Research Mode: Deep Tensor Activations")
    
    r_norms = results["residual_norms"]
    m_norms = results["mlp_norms"]
    
    fig_res = go.Figure()
    fig_res.add_trace(go.Scatter(x=[f"L{i}" for i in range(12)], y=r_norms, name="Residual Stream L2 Norm", line=dict(color="#38BDF8", width=3)))
    fig_res.add_trace(go.Scatter(x=[f"L{i}" for i in range(12)], y=m_norms, name="MLP Activation L2 Norm", line=dict(color="#F59E0B", width=2, dash="dash")))
    fig_res.update_layout(
        title="Layer-by-Layer Activation Norms",
        xaxis_title="Layer",
        yaxis_title="L2 Norm",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.6)",
        font=dict(color="#F8FAFC", family="Inter"),
        height=380
    )
    st.plotly_chart(fig_res, use_container_width=True)
