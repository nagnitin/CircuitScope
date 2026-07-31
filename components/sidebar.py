"""
components/sidebar.py
====================
Sidebar branding, navigation, and research mode toggle for CircuitScope Streamlit App.
"""

import streamlit as st
from components.utils import load_css

def render_sidebar():
    """Renders the common sidebar elements across pages."""
    load_css()
    
    with st.sidebar:
        st.markdown("## 🔬 CircuitScope")
        st.markdown("*GPT-2 Small IOI Mechanistic Interpretability*")
        st.markdown("---")
        
        # Research Mode Toggle
        if "research_mode" not in st.session_state:
            st.session_state["research_mode"] = False
            
        research_mode = st.toggle(
            "🔬 Research Mode",
            value=st.session_state["research_mode"],
            help="Enables advanced mechanistic diagnostics: residual stream norms, head vector projections, and full attention matrices."
        )
        st.session_state["research_mode"] = research_mode
        
        if research_mode:
            st.info("🔬 **Research Mode Enabled**: Exposing deep tensor activations & head projections.")
            
        st.markdown("---")
        st.markdown("### 📊 Model & Dataset")
        st.markdown("""
        - **Target Model**: `GPT-2 Small` (85M params)
        - **Layers / Heads**: 12 Layers / 144 Heads
        - **IOI Dataset**: 1,000 Prompts (ABB/BAB)
        - **Pronoun Dataset**: 500 Prompts
        - **Random Seed**: `42` (Fixed)
        """)
        
        st.markdown("---")
        st.markdown("### 📄 Quick Links")
        st.markdown("- [🔗 GitHub Repository](https://github.com/nagnitin/CircuitScope)")
        st.markdown("- [📄 Read Paper PDF/MD](https://github.com/nagnitin/CircuitScope/blob/main/paper/research_paper.md)")
        st.markdown("- [📋 Verification Report](https://github.com/nagnitin/CircuitScope/blob/main/docs/REPRODUCIBILITY_VERIFICATION.md)")
        
        st.markdown("---")
        st.caption("CircuitScope v1.0 | NeurIPS Interactive Demo")
        st.caption("Google DeepMind & Community Research")
