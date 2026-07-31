"""
components/charts.py
====================
Interactive Plotly visualizations for CircuitScope Streamlit App.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

THEME_DARK = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(15,23,42,0.6)",
    "font_color": "#F8FAFC",
    "grid_color": "rgba(51,65,85,0.4)",
}

def build_head_importance_heatmap(df_heads: pd.DataFrame) -> go.Figure:
    """Builds a 12x12 Plotly heatmap of attention head importance scores."""
    matrix = np.zeros((12, 12))
    
    for _, row in df_heads.iterrows():
        l = int(row["layer"]) if "layer" in row else int(row["head_label"][1:row["head_label"].find("H")])
        h = int(row["head"]) if "head" in row else int(row["head_label"][row["head_label"].find("H")+1:])
        imp = float(row["importance"])
        matrix[l, h] = imp
        
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=[f"H{h}" for h in range(12)],
        y=[f"L{l}" for l in range(12)],
        colorscale="Viridis",
        colorbar=dict(title="Importance (Logit Diff Drop)", titleside="right"),
        hovertemplate="Layer: %{y}<br>Head: %{x}<br>Importance: %{z:.4f}<extra></extra>"
    ))
    
    # Annotate top Name Mover heads
    nm_heads = [("L8", "H6"), ("L8", "H10"), ("L5", "H5"), ("L7", "H9")]
    for y_val, x_val in nm_heads:
        fig.add_annotation(
            x=x_val, y=y_val, text="★",
            showarrow=False, font=dict(color="#FFD700", size=18)
        )
        
    fig.update_layout(
        title="Attention Head Importance Grid (12 × 12)",
        xaxis_title="Head Index (0–11)",
        yaxis_title="Layer Index (0–11)",
        paper_bgcolor=THEME_DARK["paper_bgcolor"],
        plot_bgcolor=THEME_DARK["plot_bgcolor"],
        font=dict(color=THEME_DARK["font_color"], family="Inter"),
        margin=dict(l=50, r=50, t=50, b=50),
        height=500,
    )
    return fig

def build_logit_lens_curve(df_lens: pd.DataFrame) -> go.Figure:
    """Builds logit lens progression line chart across transformer layers."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_lens["layer_label"],
        y=df_lens["logit_diff"],
        mode="lines+markers",
        name="Mean Logit Diff",
        line=dict(color="#38BDF8", width=3),
        marker=dict(size=8, color="#38BDF8")
    ))
    
    if "prob_io" in df_lens.columns:
        fig.add_trace(go.Scatter(
            x=df_lens["layer_label"],
            y=df_lens["prob_io"],
            mode="lines+markers",
            name="P(IO Token)",
            yaxis="y2",
            line=dict(color="#10B981", width=2, dash="dash"),
            marker=dict(size=6, color="#10B981")
        ))
        
    fig.update_layout(
        title="Logit Lens: Residual Stream Accumulation Across Layers",
        xaxis_title="Transformer Layer",
        yaxis=dict(title="Mean Logit Difference", title_font=dict(color="#38BDF8")),
        yaxis2=dict(title="P(IO Token)", title_font=dict(color="#10B981"), overlaying="y", side="right", range=[0, 1]),
        paper_bgcolor=THEME_DARK["paper_bgcolor"],
        plot_bgcolor=THEME_DARK["plot_bgcolor"],
        font=dict(color=THEME_DARK["font_color"], family="Inter"),
        margin=dict(l=50, r=50, t=50, b=50),
        height=450,
        legend=dict(x=0.02, y=0.98),
    )
    return fig

def build_causal_transfer_comparison_chart() -> go.Figure:
    """Builds bar chart comparing single-head vs group-level cross-task recovery."""
    categories = [
        "Neutral Control (Single)",
        "Name Mover (Single - Exp 11)",
        "Name Mover Group A (Exp 12)",
        "NM + S-Inhib Group B (Exp 12)"
    ]
    recovery = [-2.19, -5.97, -1.12, -1.09]
    colors = ["#64748B", "#F43F5E", "#E11D48", "#BE123C"]
    
    fig = go.Figure(data=[go.Bar(
        x=categories,
        y=recovery,
        marker_color=colors,
        text=[f"{v:+.2f}%" for v in recovery],
        textposition="outside"
    )])
    
    fig.add_hline(y=0, line_dash="dash", line_color="#94A3B8", annotation_text="0% Neutral Baseline")
    
    fig.update_layout(
        title="Cross-Task Recovery Verdict: NO_TRANSFER Across All Granularities",
        xaxis_title="Patching Condition",
        yaxis_title="Mean Cross-Task Recovery (%)",
        paper_bgcolor=THEME_DARK["paper_bgcolor"],
        plot_bgcolor=THEME_DARK["plot_bgcolor"],
        font=dict(color=THEME_DARK["font_color"], family="Inter"),
        margin=dict(l=50, r=50, t=50, b=50),
        height=450,
        yaxis=dict(range=[-8, 2])
    )
    return fig

def build_circuit_graph_figure() -> go.Figure:
    """Builds a Plotly node diagram representing the 14-head IOI sub-circuit graph."""
    # Nodes: Early, Middle, Late heads
    nodes = {
        "S1/S2 Token Input": (0.1, 0.5, "#94A3B8"),
        "L0H10 (Helper)": (0.3, 0.8, "#06B6D4"),
        "L1H10 (Helper)": (0.3, 0.2, "#06B6D4"),
        "L7H9 (S-Inhibition)": (0.6, 0.7, "#F59E0B"),
        "L8H10 (S-Inhibition)": (0.6, 0.3, "#F59E0B"),
        "L8H6 (Name Mover)": (0.85, 0.85, "#6366F1"),
        "L8H10 (Name Mover)": (0.85, 0.60, "#6366F1"),
        "L5H5 (Name Mover)": (0.85, 0.35, "#6366F1"),
        "L7H9 (Name Mover)": (0.85, 0.10, "#6366F1"),
        "Logit Diff Output": (1.1, 0.5, "#10B981"),
    }
    
    fig = go.Figure()
    
    # Add Edges
    edges = [
        ("S1/S2 Token Input", "L0H10 (Helper)"),
        ("S1/S2 Token Input", "L1H10 (Helper)"),
        ("L0H10 (Helper)", "L7H9 (S-Inhibition)"),
        ("L1H10 (Helper)", "L8H10 (S-Inhibition)"),
        ("L7H9 (S-Inhibition)", "L8H6 (Name Mover)"),
        ("L8H10 (S-Inhibition)", "L8H10 (Name Mover)"),
        ("L8H6 (Name Mover)", "Logit Diff Output"),
        ("L8H10 (Name Mover)", "Logit Diff Output"),
        ("L5H5 (Name Mover)", "Logit Diff Output"),
        ("L7H9 (Name Mover)", "Logit Diff Output"),
    ]
    
    for src, dst in edges:
        x0, y0, _ = nodes[src]
        x1, y1, _ = nodes[dst]
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1],
            mode="lines",
            line=dict(width=2, color="rgba(148, 163, 184, 0.4)"),
            hoverinfo="none",
            showlegend=False
        ))
        
    # Add Nodes
    for label, (x, y, color) in nodes.items():
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode="markers+text",
            marker=dict(size=28, color=color, line=dict(width=2, color="#FFFFFF")),
            text=[label],
            textposition="top center",
            textfont=dict(color="#F8FAFC", size=11),
            name=label,
            showlegend=False
        ))
        
    fig.update_layout(
        title="Discovered 14-Head IOI Computational Sub-Circuit Graph",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 1.25]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.1, 1.1]),
        paper_bgcolor=THEME_DARK["paper_bgcolor"],
        plot_bgcolor=THEME_DARK["plot_bgcolor"],
        font=dict(color=THEME_DARK["font_color"], family="Inter"),
        margin=dict(l=20, r=20, t=50, b=20),
        height=480,
    )
    return fig
