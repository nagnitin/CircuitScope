"""
pages/09_Reproducibility.py
===========================
Clean-Room Reproducibility Verification Dashboard for CircuitScope.
"""

import streamlit as st
import pandas as pd
from components.cards import render_callout
from components.utils import load_css

load_css()

st.title("📋 Clean-Room Reproducibility Verification")
st.markdown("### Independent Audit, Environment Pinning, and Deterministic Replication Report")

# Verdict Banner
render_callout(
    title="Verification Status: REPRODUCED (100% EXACT MATCH ACROSS ALL METRICS)",
    text="Independent clean-room re-run from scratch into an isolated output directory (<code>outputs_verify/</code>) achieved <strong>100% exact match (Delta = 0.0000)</strong> across all 14 core evaluation metrics.",
    category="success",
    icon="✅"
)

st.markdown("---")

# Environment Pinning
st.header("1. Environment & Package Pinning")
st.markdown("Verified runtime environment (`Python 3.14.6`) and exact package constraints in `requirements.txt`:")

df_env = pd.DataFrame([
    {"Package": "torch", "Pinned Version": "2.12.1+cpu", "Role": "Core Deep Learning Tensor Operations"},
    {"Package": "transformer_lens", "Pinned Version": "3.6.0", "Role": "Mechanistic Interpretability Hooking Library"},
    {"Package": "numpy", "Pinned Version": "2.5.1", "Role": "Numerical Array & Matrix Computations"},
    {"Package": "pandas", "Pinned Version": "3.0.3", "Role": "Data Structure Management & CSV Exports"},
    {"Package": "scipy", "Pinned Version": "1.18.0", "Role": "Statistical Significance & Pearson/Spearman Tests"},
    {"Package": "plotly", "Pinned Version": "6.9.0", "Role": "Interactive Visualization Engine"},
])

st.table(df_env)

st.markdown("---")

# Detailed Comparison Table
st.header("2. Committed (`outputs/`) vs. Freshly Reproduced (`outputs_verify/`) Comparison")
st.markdown("Floating-point comparison tolerance: `1e-4`")

df_comp = pd.DataFrame([
    {"Experiment": "Exp 01: Baseline IOI", "Metric": "Overall Accuracy", "Committed": "0.966", "Reproduced": "0.966", "Delta": "0.0000", "Status": "EXACT MATCH"},
    {"Experiment": "Exp 01: Baseline IOI", "Metric": "Mean Logit Difference", "Committed": "+3.129324", "Reproduced": "+3.129324", "Delta": "0.0000", "Status": "EXACT MATCH"},
    {"Experiment": "Exp 02: Logit Lens", "Metric": "Layer 7 Logit Diff", "Committed": "+0.093688", "Reproduced": "+0.093688", "Delta": "0.0000", "Status": "EXACT MATCH"},
    {"Experiment": "Exp 03: Layer Ablation", "Metric": "Layer 0 MLP Resample Drop", "Committed": "1.092652", "Reproduced": "1.092652", "Delta": "0.0000", "Status": "EXACT MATCH"},
    {"Experiment": "Exp 04: Head Ablation", "Metric": "L8H6 Importance", "Committed": "0.343160", "Reproduced": "0.343160", "Delta": "0.0000", "Status": "EXACT MATCH"},
    {"Experiment": "Exp 08: Validation", "Metric": "Circuit Necessity Score", "Committed": "1.072836", "Reproduced": "1.072836", "Delta": "0.0000", "Status": "EXACT MATCH"},
    {"Experiment": "Exp 08: Validation", "Metric": "Circuit Sufficiency Score", "Committed": "0.847699", "Reproduced": "0.847699", "Delta": "0.0000", "Status": "EXACT MATCH"},
    {"Experiment": "Exp 09: Novel Extension", "Metric": "Pearson r (Head Importance)", "Committed": "0.575044", "Reproduced": "0.575044", "Delta": "0.0000", "Status": "EXACT MATCH"},
    {"Experiment": "Exp 10: Statistics", "Metric": "Name Mover Cohen's d", "Committed": "+4.897581", "Reproduced": "+4.897581", "Delta": "0.0000", "Status": "EXACT MATCH"},
    {"Experiment": "Exp 11: Cross Patching", "Metric": "Name Mover Cross Recovery", "Committed": "-0.059700", "Reproduced": "-0.059700", "Delta": "0.0000", "Status": "EXACT MATCH"},
    {"Experiment": "Exp 11: Cross Patching", "Metric": "Causal Transfer Verdict", "Committed": "NO_TRANSFER", "Reproduced": "NO_TRANSFER", "Delta": "0", "Status": "EXACT MATCH"},
    {"Experiment": "Exp 12: Multihead Patching", "Metric": "Group A Cross Recovery", "Committed": "-1.122500", "Reproduced": "-1.122500", "Delta": "0.0000", "Status": "EXACT MATCH"},
    {"Experiment": "Exp 12: Multihead Patching", "Metric": "Group B Cross Recovery", "Committed": "-1.094700", "Reproduced": "-1.094700", "Delta": "0.0000", "Status": "EXACT MATCH"},
    {"Experiment": "Exp 12: Multihead Patching", "Metric": "Causal Transfer Verdict", "Committed": "NO_TRANSFER_EVEN_AT_GROUP_LEVEL", "Reproduced": "NO_TRANSFER_EVEN_AT_GROUP_LEVEL", "Delta": "0", "Status": "EXACT MATCH"},
])

st.dataframe(df_comp, use_container_width=True)

st.markdown("---")

st.header("3. How to Execute Reproducibility Verification")
st.code("""
# PowerShell / Bash Execution
$env:PYTHONIOENCODING="utf-8"

# Run all experiments using verification configuration
python experiments/baseline_ioi.py --config config/experiment_config_verify.yaml
python experiments/02_logit_lens.py --config config/experiment_config_verify.yaml --n-samples 200
python experiments/03_layer_ablation.py --config config/experiment_config_verify.yaml --n-samples 200
python experiments/04_head_ablation.py --config config/experiment_config_verify.yaml --n-samples 200
python experiments/05_activation_patching.py --config config/experiment_config_verify.yaml --n-samples 50
python experiments/06_path_patching.py --config config/experiment_config_verify.yaml --n-samples 50
python experiments/08_circuit_validation.py --config config/experiment_config_verify.yaml --n-samples 150 --threshold 0.05
python experiments/09_novel_extension.py --config config/experiment_config_verify.yaml --n-prompts 500 --n-samples 200
python experiments/10_statistical_analysis.py --config config/experiment_config_verify.yaml --n-bootstrap 2000
python experiments/11_cross_task_patching.py --config config/experiment_config_verify.yaml --n-samples 150
python experiments/12_multihead_patching.py --config config/experiment_config_verify.yaml --n-samples 150
""", language="powershell")
