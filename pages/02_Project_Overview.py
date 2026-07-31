"""
pages/02_Project_Overview.py
=============================
Project Overview Page — Detailed Explanations of Mechanistic Interpretability Methods & Tasks.
"""

import streamlit as st
from components.cards import render_callout
from components.utils import load_css

load_css()

st.title("📖 Project Overview & Methodology")
st.markdown("### Mechanistic Interpretability Framework, Tasks, and Analytical Techniques")

st.markdown("""
**Mechanistic Interpretability** seeks to reverse-engineer trained neural networks into human-understandable algorithms. 
Rather than treating language models as black-box predictors, CircuitScope inspects individual transformer components — residual streams, attention heads, and MLP blocks — to trace how structural information flows from raw tokens to final logit output.
""")

# Tasks Explanation Tabs
st.header("🎯 Benchmark Reasoning Tasks")

tab_ioi, tab_pronoun = st.tabs(["1. Indirect Object Identification (IOI)", "2. Pronoun Resolution (Novel Extension)"])

with tab_ioi:
    st.markdown("""
    #### Indirect Object Identification (IOI)
    The **IOI task** tests how a model identifies the indirect object in sentences with two distinct names:
    
    > *"When **John** [S1] and **Mary** [IO] went to the store, **John** [S2] gave the book to **___**"* $\\rightarrow$ Target: **Mary**
    
    - **S1 & S2**: The subject name that appears twice (*John*).
    - **IO**: The indirect object name that appears once (*Mary*).
    - **Target Output**: The model must predict the IO token over the duplicate S token.
    - **Metric (Logit Difference)**:
      $$\\text{Logit Diff} = \\text{Logit}(\\text{IO}) - \\text{Logit}(\\text{S})$$
    """)

with tab_pronoun:
    st.markdown("""
    #### Pronoun Resolution Task
    To test whether discovered sub-circuits generalize across tasks, CircuitScope introduces **Pronoun Resolution**:
    
    > *"**Sarah** [N1] met **James** [N2] at the café. **She** [Pronoun] bought a gift for **___**"* $\\rightarrow$ Target: **James**
    
    - **Gendered Pronoun Match**: *She* matches *Sarah*, so the object of the action must be *James*.
    - **Question**: Do the same "Name Mover" attention heads that extract the IO name in IOI also extract *James* in pronoun resolution?
    - **Correlation**: Yes ($r = 0.5750, p = 4.78 \\times 10^{-14}$).
    - **Causal Transferability**: No (patching activations fails: $-5.97\\%$ recovery).
    """)

st.markdown("---")

# Analytical Methods Breakdown
st.header("🛠️ Core Analytical Techniques")

with st.expander("🔍 1. Logit Lens", expanded=True):
    st.markdown("""
    **Concept:** Projects the intermediate residual stream at layer $l$ directly onto the vocabulary via the unembedding matrix $W_U$:
    $$\\text{Logits}^{(l)} = \\text{LayerNorm}(x^{(l)}) W_U$$
    **Purpose:** Reveals the exact layer where the model begins to prefer the correct Indirect Object name over the duplicate Subject name.
    **Finding:** In GPT-2 Small, IO preference first emerges at **Layer 0** (residual stream) and undergoes massive refinement in **Layers 8–11**.
    """)

with st.expander("🔨 2. Layer & Head Ablation (Mean vs. Resample Control)", expanded=False):
    st.markdown("""
    **Concept:** Zeroing out or replacing activations at a specific layer or attention head to measure performance degradation.
    - **Mean Ablation:** Replaces activation with dataset-wide mean activation $\\bar{z}$.
    - **Resample Ablation (Control):** Replaces activation with activation from a mismatched prompt in the same batch.
    
    **Why Resample Control Matters:** Mean ablation can introduce out-of-distribution (OOD) activations. Resample ablation proves that Layer 0 MLP's large drop (resample drop = **1.0927**) is a genuine forward-pass dependency, not a mean-ablation artifact.
    """)

with st.expander("🧲 3. Causal Activation Patching", expanded=False):
    st.markdown("""
    **Concept:** Replaces specific activations in a *corrupted* prompt run (e.g. name swapped) with clean activations from a *clean* prompt run.
    $$\\text{Recovery Score} = \\frac{\\text{LogitDiff}_{patched} - \\text{LogitDiff}_{corrupted}}{\\text{LogitDiff}_{clean} - \\text{LogitDiff}_{corrupted}}$$
    **Cross-Task Patching:** Replaces IOI head activations with clean Pronoun activations. Tests whether head representations are causally interchangeable across tasks.
    """)

with st.expander("🗺️ 4. Path Patching & Sub-Circuit Validation", expanded=False):
    st.markdown("""
    **Concept:** Isolates direct information flow along specific edges (sender head $\\rightarrow$ receiver head) without altering collateral paths.
    **Circuit Validation Metrics:**
    - **Necessity Score:** Performance drop when ablating the candidate circuit while keeping the rest intact.
    - **Sufficiency Score:** Performance retained when keeping ONLY the candidate circuit and ablating everything else.
    """)

render_callout(
    title="Key Methodological Insight",
    text="Correlational ranking methods (like head ablation importance) tell you <em>where</em> computational activity is concentrated, but only <strong>causal intervention methods (activation patching)</strong> can prove whether two sub-circuits perform interchangeable computations.",
    category="info",
    icon="💡"
)
