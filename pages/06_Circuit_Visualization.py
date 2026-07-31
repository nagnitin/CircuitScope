"""
pages/06_Circuit_Visualization.py
==================================
Interactive 14-Head IOI Sub-Circuit Graph, Node Inspector, and Validation Dashboard.
"""

import streamlit as st
import pandas as pd
from components.charts import build_circuit_graph_figure
from components.cards import render_callout, render_head_role_pill
from components.utils import load_css, load_exp_csv

load_css()

st.title("🗺️ Discovered IOI Computational Sub-Circuit")
st.markdown("### Interactive Circuit Graph, Node Inspector, and Quantitative Validation")

# Load Circuit Validation Data
df_val = load_exp_csv("08_circuit_validation", "circuit_validation.csv")

# Necessity & Sufficiency KPI Cards
c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Circuit Size", "14 Heads / 144 Total", delta="9.7% of model heads", delta_color="normal")
    
with c2:
    nec_val = df_val[df_val["test_name"]=="necessity"]["score"].values[0] if df_val is not None and "test_name" in df_val.columns else 1.0728
    st.metric("Circuit Necessity Score", f"{nec_val:.4f}", delta="55.3% Accuracy Drop", delta_color="inverse")
    
with c3:
    suf_val = df_val[df_val["test_name"]=="sufficiency"]["score"].values[0] if df_val is not None and "test_name" in df_val.columns else 0.8477
    st.metric("Circuit Sufficiency Score", f"{suf_val:.4f}", delta="84.8% Logit Diff Retained", delta_color="normal")

st.markdown("---")

# 1. Interactive Graph Visualization
st.header("1. Sub-Circuit Architecture Node Graph")
st.plotly_chart(build_circuit_graph_figure(), use_container_width=True)

st.markdown("---")

# 2. Interactive Node Detail Inspector
st.header("2. Clickable Node & Layer Detail Inspector")
st.markdown("Select a component node from the discovered circuit to inspect its internal mechanics, importance score, and related experiments:")

selected_node = st.selectbox(
    "Select Circuit Node:",
    [
        "L8H6 (Name Mover Head)",
        "L8H10 (Name Mover Head)",
        "L5H5 (Name Mover Head)",
        "L7H9 (Name Mover & S-Inhibition)",
        "L0H10 (Helper / Duplicate Token Head)",
        "L1H10 (Helper / Duplicate Token Head)",
        "L0 MLP (Early Layer Transformation)",
        "L5 MLP (Mid Layer Non-Linearity)",
        "Residual Stream (Memory Bus)",
    ]
)

st.markdown("<br>", unsafe_allow_html=True)

if "L8H6" in selected_node:
    st.markdown("""
    <div class="paper-card">
        <h3 style="color: #818CF8 !important;">★ L8H6 (Primary Name Mover Head)</h3>
        <p><strong>Layer:</strong> 8 | <strong>Head Index:</strong> 6</p>
        <p><strong>Functional Role:</strong> <span class="pill-tag pill-name-mover">Name Mover</span> Directly reads the Indirect Object (IO) name representation and writes its directional vector into the residual stream towards the unembedding target token.</p>
        <p><strong>Ablation Importance:</strong> <code>0.3432</code> (Rank #1 out of 144 heads)</p>
        <p><strong>Related Experiments:</strong> Exp 04 (Head Ablation), Exp 05 (Activation Patching), Exp 11 (Cross-Task Patching), Exp 12 (Group Patching).</p>
    </div>
    """, unsafe_allow_html=True)

elif "L8H10" in selected_node and "S-Inhibition" not in selected_node:
    st.markdown("""
    <div class="paper-card">
        <h3 style="color: #818CF8 !important;">★ L8H10 (Name Mover Head)</h3>
        <p><strong>Layer:</strong> 8 | <strong>Head Index:</strong> 10</p>
        <p><strong>Functional Role:</strong> <span class="pill-tag pill-name-mover">Name Mover</span> Secondary Name Mover head operating in parallel with L8H6 to reinforce IO token projection.</p>
        <p><strong>Ablation Importance:</strong> <code>0.2816</code> (Rank #2 out of 144 heads)</p>
        <p><strong>Related Experiments:</strong> Exp 04, Exp 05, Exp 12.</p>
    </div>
    """, unsafe_allow_html=True)

elif "L7H9" in selected_node:
    st.markdown("""
    <div class="paper-card">
        <h3 style="color: #FBBF24 !important;">L7H9 (S-Inhibition & Name Mover)</h3>
        <p><strong>Layer:</strong> 7 | <strong>Head Index:</strong> 9</p>
        <p><strong>Functional Role:</strong> <span class="pill-tag pill-s-inhibition">S-Inhibition</span> Suppresses attention to the duplicate Subject name (S2) so Name Movers attend selectively to IO.</p>
        <p><strong>Ablation Importance:</strong> <code>0.2247</code> (Rank #4 out of 144 heads)</p>
        <p><strong>Related Experiments:</strong> Exp 04, Exp 06 (Path Patching), Exp 12.</p>
    </div>
    """, unsafe_allow_html=True)

elif "L0H10" in selected_node or "L1H10" in selected_node:
    st.markdown("""
    <div class="paper-card">
        <h3 style="color: #22D3EE !important;">L0H10 / L1H10 (Helper & Duplicate Token Heads)</h3>
        <p><strong>Layer:</strong> 0 or 1 | <strong>Head Index:</strong> 10</p>
        <p><strong>Functional Role:</strong> <span class="pill-tag pill-helper">Helper</span> Detects duplicated name tokens (S1 and S2) early in the sequence and passes positional signals downstream.</p>
        <p><strong>Ablation Importance:</strong> <code>0.1216</code> (Rank #7 out of 144 heads)</p>
        <p><strong>Related Experiments:</strong> Exp 04, Exp 06 (Path Patching).</p>
    </div>
    """, unsafe_allow_html=True)

elif "MLP" in selected_node:
    st.markdown("""
    <div class="paper-card">
        <h3 style="color: #38BDF8 !important;">⚡ Layer MLP Blocks (L0 MLP & L5 MLP)</h3>
        <p><strong>Functional Role:</strong> Computes key non-linear transformations required for signal propagation.</p>
        <p><strong>Resample Control Verdict:</strong> Resample ablation (drop = <code>1.0927</code>) proves Layer 0 MLP is a genuine forward-pass requirement rather than a mean-ablation artifact.</p>
        <p><strong>Related Experiments:</strong> Exp 03 (Layer Ablation), Exp 05 (Activation Patching).</p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="paper-card">
        <h3 style="color: #38BDF8 !important;">🔄 Residual Stream Memory Bus</h3>
        <p><strong>Dimension:</strong> $d_{model} = 768$</p>
        <p><strong>Functional Role:</strong> Acts as the shared 768-dimensional communication backbone where attention heads and MLPs write linear updates.</p>
        <p><strong>Related Experiments:</strong> Exp 02 (Logit Lens), Exp 05 (Residual Activation Patching).</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# 3. Quantitative Circuit Validation Results
st.header("3. Quantitative Circuit Validation Results")

if df_val is not None:
    st.dataframe(df_val, use_container_width=True)
