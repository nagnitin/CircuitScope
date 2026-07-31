"""
pages/07_Experiments.py
=======================
Master Experiment Hub for all 12 Pipeline Experiments in CircuitScope.
"""

import streamlit as st
import pandas as pd
import json
import streamlit.components.v1 as components
from components.cards import render_callout
from components.utils import load_css, load_exp_csv, load_exp_json, load_html_figure, get_primary_outputs_path

load_css()

st.title("🧪 Master Experiments Hub")
st.markdown("### Interactive Results Inspector for all 12 Pipeline Experiments")

# Select Experiment
exp_choice = st.selectbox(
    "Select Experiment View:",
    [
        "01. Baseline IOI Evaluation",
        "02. Logit Lens Analysis",
        "03. Layer Ablation (Mean & Resample Control)",
        "04. Attention Head Ablation Sweep",
        "05. Activation Patching (Residual, Attn, MLP)",
        "06. Path Patching & Circuit Graph",
        "07. Full Automated Pipeline Runner",
        "08. Circuit Validation (Necessity & Sufficiency)",
        "09. Pronoun Resolution Extension",
        "10. Statistical Analysis & Bootstrap CIs",
        "11. Single-Head Cross-Task Patching",
        "12. Multi-Head Group Sub-Circuit Patching",
    ]
)

st.markdown("---")

if "01." in exp_choice:
    st.header("01. Baseline IOI Evaluation")
    st.markdown("**Goal:** Measure baseline model performance (accuracy, logit difference, P(IO)) across 1,000 generated IOI prompts.")
    
    df_res = load_exp_csv("01_baseline", "ioi_results.csv")
    meta = load_exp_json("01_baseline", "experiment_metadata.json")
    
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Overall Accuracy", "96.6%")
    with c2: st.metric("Mean Logit Diff", "+3.1293")
    with c3: st.metric("Mean P(IO)", "0.3617")
    
    if df_res is not None:
        st.subheader("Sample Results Data (First 10 Prompts)")
        st.dataframe(df_res.head(10), use_container_width=True)

elif "02." in exp_choice:
    st.header("02. Logit Lens Analysis")
    st.markdown("**Goal:** Project intermediate residual stream states directly to vocabulary logits to pinpoint where preference emerges.")
    
    df_lens = load_exp_csv("02_logit_lens", "logit_lens_by_layer.csv")
    if df_lens is not None:
        st.dataframe(df_lens, use_container_width=True)
    
    fig_html = load_html_figure("02_logit_lens", "02_logit_lens_layer_progression.html")
    if fig_html:
        components.html(fig_html, height=550)

elif "03." in exp_choice:
    st.header("03. Layer Ablation (Mean & Resample Control)")
    st.markdown("**Goal:** Measure drop in logit difference when ablating full layers, attention, or MLP blocks.")
    
    df_resample = load_exp_csv("03_layer_ablation", "layer_ablation_resample.csv")
    if df_resample is not None:
        st.subheader("Resample Ablation Control Results")
        st.dataframe(df_resample, use_container_width=True)
        
    render_callout(
        title="Resample Control Finding",
        text="Layer 0 MLP exhibits a resample normalized drop of <strong>1.0927</strong>, proving its critical role is a genuine forward-pass dependency.",
        category="success"
    )

elif "04." in exp_choice:
    st.header("04. Attention Head Ablation Sweep")
    st.markdown("**Goal:** Systematically ablate all 144 attention heads to rank head importance.")
    
    df_heads = load_exp_csv("04_head_ablation", "head_ablation.csv")
    if df_heads is not None:
        st.dataframe(df_heads, use_container_width=True)

elif "05." in exp_choice:
    st.header("05. Activation Patching")
    st.markdown("**Goal:** Patch residual, attention, and MLP activations from clean runs into corrupted runs.")
    
    df_resid = load_exp_csv("05_activation_patching", "patching_resid.csv")
    if df_resid is not None:
        st.subheader("Residual Patching Matrix")
        st.dataframe(df_resid.head(10), use_container_width=True)

elif "06." in exp_choice:
    st.header("06. Path Patching & Circuit Graph")
    st.markdown("**Goal:** Identify direct causal connections between sender and receiver heads.")
    
    summary_6 = load_exp_json("06_path_patching", "circuit_summary.json")
    if summary_6:
        st.json(summary_6)

elif "07." in exp_choice:
    st.header("07. Full Automated Pipeline Runner")
    st.markdown("Orchestrates all analysis parts sequentially with shared model loading.")
    st.code("python experiments/07_full_pipeline.py --full-patching", language="bash")

elif "08." in exp_choice:
    st.header("08. Circuit Validation")
    st.markdown("**Goal:** Validate necessity (1.0728) and sufficiency (0.8477) of the 14-head circuit.")
    
    df_val = load_exp_csv("08_circuit_validation", "circuit_validation.csv")
    if df_val is not None:
        st.dataframe(df_val, use_container_width=True)

elif "09." in exp_choice:
    st.header("09. Pronoun Resolution Extension")
    st.markdown("**Goal:** Test head-importance correlation between IOI and Pronoun tasks.")
    
    comp_9 = load_exp_json("09_novel_extension", "task_comparison.json")
    if comp_9:
        st.json(comp_9)
        st.success(f"Pearson r = {comp_9['head_importance_correlation']['pearson_r']:.4f} (p = {comp_9['head_importance_correlation']['p_value']:.2e})")

elif "10." in exp_choice:
    st.header("10. Statistical Analysis & Bootstrap CIs")
    st.markdown("**Goal:** Compute 2,000 bootstrap CIs, Cohen's d effect sizes, and permutation tests.")
    
    df_eff = load_exp_csv("10_statistical_analysis", "stats_effect_sizes.csv")
    if df_eff is not None:
        st.dataframe(df_eff, use_container_width=True)

elif "11." in exp_choice:
    st.header("11. Single-Head Cross-Task Patching")
    st.markdown("**Goal:** Test cross-task transfer by patching individual Name Mover heads.")
    
    sum_11 = load_exp_json("11_cross_task_patching", "cross_task_summary.json")
    if sum_11:
        st.json(sum_11)
        render_callout(title="Verdict", text=f"Verdict: <strong>{sum_11.get('causal_transfer_verdict')}</strong> (Mean recovery = {sum_11.get('name_mover_cross_recovery'):.4f})", category="danger")

elif "12." in exp_choice:
    st.header("12. Multi-Head Group Sub-Circuit Patching")
    st.markdown("**Goal:** Jointly patch full 4-head Name Mover group (Group A) and 5-head NM+SI group (Group B).")
    
    sum_12 = load_exp_json("12_multihead_patching", "multihead_summary.json")
    if sum_12:
        st.json(sum_12)
        render_callout(title="Verdict", text=f"Verdict: <strong>{sum_12.get('causal_transfer_verdict')}</strong> (Group A = {sum_12.get('group_a_name_mover_cross_recovery'):.4f}, Group B = {sum_12.get('group_b_nm_si_cross_recovery'):.4f})", category="danger")
