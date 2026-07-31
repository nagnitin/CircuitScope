"""
components/cards.py
===================
Reusable paper-style info callouts and concept cards for CircuitScope Streamlit App.
"""

import streamlit as st

def render_callout(title: str, text: str, category: str = "info", icon: str = "💡"):
    """
    Renders a styled callout card.
    category: 'info' | 'success' | 'warning' | 'danger'
    """
    st.markdown(f"""
    <div class="callout-box {category}">
        <div class="callout-title">{icon} {title}</div>
        <div style="font-size: 0.9rem; color: #CBD5E1; line-height: 1.5;">
            {text}
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_paper_summary_card():
    """Renders the executive paper summary card."""
    st.markdown("""
    <div class="paper-card">
        <h3 style="margin-top: 0; color: #38BDF8 !important;">🔬 Executive Summary & Key Research Finding</h3>
        <p style="font-size: 0.95rem; color: #E2E8F0; line-height: 1.6;">
            <strong>CircuitScope</strong> presents a complete mechanistic interpretability analysis of the Indirect Object Identification (IOI) circuit in <strong>GPT-2 Small</strong> (12 layers, 144 heads). 
            Using logit lens, resample ablation controls, activation patching, and path patching, we reverse-engineer the exact 14-head sub-circuit responsible for structural reasoning.
        </p>
        <div style="background: rgba(99, 102, 241, 0.15); border-left: 3px solid #6366F1; padding: 12px 16px; border-radius: 6px; margin: 12px 0;">
            <strong style="color: #818CF8;">Central Finding:</strong> Head-importance correlation between tasks (Pearson <em>r</em> = 0.5750, <em>p</em> = 4.78 × 10<sup>-14</sup>) 
            <strong>does NOT imply causal circuit transferability</strong>. Bidirectional activation patching reveals no functional transfer at either the single-head level (-5.97% recovery) or the full group level (-1.12% recovery).
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_head_role_pill(role: str) -> str:
    """Returns HTML pill tag for an attention head role."""
    role_clean = role.lower().strip()
    if "name mover" in role_clean:
        return '<span class="pill-tag pill-name-mover">Name Mover</span>'
    elif "inhibition" in role_clean or "s-inhibition" in role_clean:
        return '<span class="pill-tag pill-s-inhibition">S-Inhibition</span>'
    elif "helper" in role_clean:
        return '<span class="pill-tag pill-helper">Helper</span>'
    elif "suppressor" in role_clean:
        return '<span class="pill-tag pill-suppressor">Suppressor</span>'
    else:
        return '<span class="pill-tag pill-neutral">Neutral</span>'
