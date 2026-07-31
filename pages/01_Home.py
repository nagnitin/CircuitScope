"""
pages/01_Home.py
================
Home Page — Executive Summary, Research Highlights, and Pipeline Overview.
"""

import streamlit as st
from components.metrics import render_headline_kpis
from components.cards import render_callout, render_paper_summary_card
from components.utils import load_css

load_css()

st.title("🏠 CircuitScope: Home & Research Highlights")
st.markdown("### Executive Summary, Benchmark Datasets, and Pipeline Flow")

# Top KPI Metric Cards
render_headline_kpis()

st.markdown("<br>", unsafe_allow_html=True)

# Main Summary Card
render_paper_summary_card()

# Pipeline Flow Stepper
st.header("⚡ The CircuitScope 12-Stage Pipeline Flow")
st.markdown("CircuitScope executes a 12-stage sequential analysis pipeline to map and validate internal transformer mechanics:")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("""
    #### Stage 1: Localization
    1. **01. Baseline IOI**: Baseline accuracy (96.6%) & logit diff (+3.1293).
    2. **02. Logit Lens**: Layer-by-layer residual stream projection.
    3. **03. Layer Ablation**: Mean & resample ablation controls.
    """)

with c2:
    st.markdown("""
    #### Stage 2: Isolation
    4. **04. Head Ablation**: 144-head importance sweep.
    5. **05. Activation Patching**: Residual, Attn, & MLP patching.
    6. **06. Path Patching**: Directed sender-receiver circuit graph.
    """)

with c3:
    st.markdown("""
    #### Stage 3: Validation
    7. **07. Full Pipeline**: Master automated runner.
    8. **08. Circuit Validation**: Necessity (1.0728) & Sufficiency (0.8477).
    9. **09. Pronoun Extension**: Cross-task generalization.
    """)

with c4:
    st.markdown("""
    #### Stage 4: Rigor
    10. **10. Statistics**: Bootstrap CIs, Cohen's d (+4.90), permutation tests.
    11. **11. Single-Head Patch**: Cross-task patching (`NO_TRANSFER`).
    12. **12. Multi-Head Patch**: Group patching (`NO_TRANSFER_EVEN_AT_GROUP`).
    """)

st.markdown("---")

# Benchmark Datasets Overview
st.header("📊 Benchmark Datasets Overview")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("""
    <div class="paper-card">
        <h4>1. Indirect Object Identification (IOI) Dataset</h4>
        <p><strong>Size:</strong> 1,000 generated prompts (50% ABB, 50% BAB templates)</p>
        <p><strong>Example (ABB):</strong> <em>"When Alice and Bob visited the store, Alice gave the book to ___"</em> (Target: <strong>Bob</strong>)</p>
        <p><strong>Example (BAB):</strong> <em>"When Alice and Bob visited the store, Bob gave the book to ___"</em> (Target: <strong>Alice</strong>)</p>
        <p><strong>Name Pool:</strong> 30+ single-token names evaluated to prevent tokenization artifacts.</p>
    </div>
    """, unsafe_allow_html=True)

with col_b:
    st.markdown("""
    <div class="paper-card">
        <h4>2. Pronoun Resolution Dataset (Novel Extension)</h4>
        <p><strong>Size:</strong> 500 generated prompts</p>
        <p><strong>Example:</strong> <em>"Sarah met James at the café. She bought a gift for ___"</em> (Target: <strong>James</strong>)</p>
        <p><strong>Goal:</strong> Test whether Name Mover heads performing IOI structural reasoning causally transfer to pronoun-based name prediction.</p>
        <p><strong>Finding:</strong> Pearson correlation <em>r</em> = 0.5750 (<em>p</em> = 4.78e-14), but zero causal transfer.</p>
    </div>
    """, unsafe_allow_html=True)

# Main Contributions Callout Box
render_callout(
    title="Core Contributions of CircuitScope",
    text="""
    <ul>
        <li><strong>Resample Ablation Control:</strong> Resolved the Layer 0 MLP anomaly using resample ablation controls (replace activation with mismatched prompt in batch), confirming a genuine forward-pass requirement.</li>
        <li><strong>First Multi-Head Group Transplantation:</strong> Jointly patched the full 4-head Name Mover group (Group A) and 5-head NM+SI group (Group B), proving sub-circuit non-transferability at group level.</li>
        <li><strong>Methodological Caution:</strong> Empirically demonstrated that head-importance correlation across tasks does NOT imply causal circuit sharing.</li>
        <li><strong>100% Deterministic Reproducibility:</strong> Fixed seed 42 with verified clean-room reproduction report (zero numerical drift across all metrics).</li>
    </ul>
    """,
    category="success",
    icon="✨"
)
