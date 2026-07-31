"""
components/metrics.py
====================
Custom styled metric cards and KPI summary layouts for CircuitScope.
"""

import streamlit as st
from components.utils import get_key_metrics_summary

def render_metric_card(title: str, value: str, subtitle: str = "", help_text: str = None):
    """Renders a single styled NeurIPS-level KPI metric card."""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-subtitle">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)
    if help_text:
        st.caption(help_text)

def render_headline_kpis():
    """Renders the top 5 core research KPI metric cards."""
    metrics = get_key_metrics_summary()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        render_metric_card(
            title="IOI Baseline Accuracy",
            value=f"{metrics['accuracy']:.1%}",
            subtitle="GPT-2 Small (1,000 prompts)",
            help_text="Mean accuracy predicting correct IO token over distractors."
        )
        
    with col2:
        render_metric_card(
            title="Head Correlation",
            value=f"r = {metrics['pearson_r']:.4f}",
            subtitle=f"p = {metrics['pearson_p']:.2e} (n=144)",
            help_text="Pearson correlation between IOI & Pronoun head importances."
        )
        
    with col3:
        render_metric_card(
            title="Single-Head Transfer",
            value=f"{metrics['single_head_transfer']:+.2f}%",
            subtitle=f"Verdict: {metrics['single_head_verdict']}",
            help_text="Mean cross-task logit-diff recovery at Name Mover heads."
        )
        
    with col4:
        render_metric_card(
            title="Group-Level Transfer",
            value=f"{metrics['group_head_transfer']:+.2f}%",
            subtitle="NO_TRANSFER_EVEN_AT_GROUP",
            help_text="Simultaneous 4-head Name Mover group transplantation."
        )
        
    with col5:
        render_metric_card(
            title="Circuit Necessity",
            value=f"{metrics['necessity_score']:.4f}",
            subtitle="55.3% Accuracy Drop",
            help_text="Score > 1.0 indicates critical circuit dependence."
        )
