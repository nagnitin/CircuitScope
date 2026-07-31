"""
pages/10_Downloads.py
=====================
Download Center for Research Papers, Data CSVs, and JSON Summaries.
"""

import streamlit as st
from pathlib import Path
from components.cards import render_callout
from components.utils import load_css, PROJECT_ROOT

load_css()

st.title("📥 Research Artifacts & Data Download Center")
st.markdown("### Export Codebase Artifacts, Paper Drafts, CSV Results, and JSON Summaries")

st.header("📄 1. Paper & Documentation Manuscripts")

col_doc1, col_doc2, col_doc3 = st.columns(3)

with col_doc1:
    paper_path = PROJECT_ROOT / "paper" / "research_paper.md"
    if paper_path.exists():
        st.download_button(
            label="📄 Download Research Paper (Markdown)",
            data=paper_path.read_text(encoding="utf-8"),
            file_name="CircuitScope_Research_Paper.md",
            mime="text/markdown"
        )
        st.caption("Full 450+ line paper with math formulations, figures, and references.")

with col_doc2:
    audit_path = PROJECT_ROOT / "docs" / "REPRODUCIBILITY_VERIFICATION.md"
    if audit_path.exists():
        st.download_button(
            label="📋 Download Verification Report",
            data=audit_path.read_text(encoding="utf-8"),
            file_name="REPRODUCIBILITY_VERIFICATION.md",
            mime="text/markdown"
        )
        st.caption("Clean-room reproduction audit report and environment package pins.")

with col_doc3:
    port_path = PROJECT_ROOT / "docs" / "portfolio.md"
    if port_path.exists():
        st.download_button(
            label="💼 Download Portfolio Summary",
            data=port_path.read_text(encoding="utf-8"),
            file_name="CircuitScope_Portfolio.md",
            mime="text/markdown"
        )
        st.caption("Executive overview, architecture diagram, and resume bullets.")

st.markdown("---")

st.header("📊 2. Experiment Data CSVs")

col_csv1, col_csv2 = st.columns(2)

with col_csv1:
    # 01 Baseline
    p_b = PROJECT_ROOT / "outputs" / "01_baseline" / "results" / "ioi_results.csv"
    if not p_b.exists(): p_b = PROJECT_ROOT / "outputs" / "results" / "ioi_results.csv"
    if p_b.exists():
        st.download_button("📊 Download Baseline IOI Results CSV (1,000 Prompts)", data=p_b.read_bytes(), file_name="ioi_results.csv", mime="text/csv")
        
    # 04 Head Ablation
    p_h = PROJECT_ROOT / "outputs" / "04_head_ablation" / "results" / "head_ablation.csv"
    if p_h.exists():
        st.download_button("📊 Download 144-Head Ablation Sweep CSV", data=p_h.read_bytes(), file_name="head_ablation.csv", mime="text/csv")

    # 08 Circuit Validation
    p_v = PROJECT_ROOT / "outputs" / "08_circuit_validation" / "results" / "circuit_validation.csv"
    if p_v.exists():
        st.download_button("📊 Download Circuit Validation Results CSV", data=p_v.read_bytes(), file_name="circuit_validation.csv", mime="text/csv")

with col_csv2:
    # 03 Layer Ablation Resample
    p_l = PROJECT_ROOT / "outputs" / "03_layer_ablation" / "results" / "layer_ablation_resample.csv"
    if p_l.exists():
        st.download_button("📊 Download Resample Layer Ablation CSV", data=p_l.read_bytes(), file_name="layer_ablation_resample.csv", mime="text/csv")
        
    # 11 Cross Task Patching
    p_c11 = PROJECT_ROOT / "outputs" / "11_cross_task_patching" / "results" / "cross_task_patching.csv"
    if p_c11.exists():
        st.download_button("📊 Download Single-Head Cross-Task Patching CSV", data=p_c11.read_bytes(), file_name="cross_task_patching.csv", mime="text/csv")

    # 12 Multihead Patching
    p_c12 = PROJECT_ROOT / "outputs" / "12_multihead_patching" / "results" / "multihead_patching.csv"
    if p_c12.exists():
        st.download_button("📊 Download Multi-Head Group Patching CSV", data=p_c12.read_bytes(), file_name="multihead_patching.csv", mime="text/csv")

st.markdown("---")

st.header("🔑 3. JSON Summary Artifacts")

col_j1, col_j2 = st.columns(2)

with col_j1:
    p_j11 = PROJECT_ROOT / "outputs" / "11_cross_task_patching" / "results" / "cross_task_summary.json"
    if p_j11.exists():
        st.download_button("🔑 Download Single-Head Cross Patching Summary (JSON)", data=p_j11.read_bytes(), file_name="cross_task_summary.json", mime="application/json")

    p_j09 = PROJECT_ROOT / "outputs" / "09_novel_extension" / "results" / "task_comparison.json"
    if p_j09.exists():
        st.download_button("🔑 Download Pronoun Task Comparison (JSON)", data=p_j09.read_bytes(), file_name="task_comparison.json", mime="application/json")

with col_j2:
    p_j12 = PROJECT_ROOT / "outputs" / "12_multihead_patching" / "results" / "multihead_summary.json"
    if p_j12.exists():
        st.download_button("🔑 Download Multi-Head Group Patching Summary (JSON)", data=p_j12.read_bytes(), file_name="multihead_summary.json", mime="application/json")

render_callout(
    title="Data License & Open Access",
    text="All dataset CSVs, experiment metadata JSONs, and visualization HTML figures are released under the MIT Open Source License.",
    category="success",
    icon="🔓"
)
