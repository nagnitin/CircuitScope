"""
app.py
======
CircuitScope: Mechanistic Interpretability Research Web Application.
Master entry point using Streamlit 1.30+ Navigation Architecture.

To run:
    streamlit run app.py
"""

import sys
from pathlib import Path
import streamlit as st

# Project Root Setup
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from components.sidebar import render_sidebar
from components.utils import load_css

# Global Page Config
st.set_page_config(
    page_title="CircuitScope — Mechanistic Interpretability",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Global CSS Styling & Sidebar
load_css()
render_sidebar()

# Define Streamlit Navigation Structure across pages/
pages = {
    "Research Core": [
        st.Page("pages/01_Home.py", title="01. Home & Executive Summary", icon="🏠", default=True),
        st.Page("pages/02_Project_Overview.py", title="02. Project Overview & Methods", icon="📖"),
        st.Page("pages/03_GPT2_Architecture.py", title="03. GPT-2 Transformer Architecture", icon="🏗️"),
        st.Page("pages/04_Prompt_Playground.py", title="04. Live Prompt Playground", icon="⚡"),
        st.Page("pages/05_Attention_Explorer.py", title="05. Attention Head Explorer", icon="🎯"),
    ],
    "Experiments & Evidence": [
        st.Page("pages/06_Circuit_Visualization.py", title="06. 14-Head Circuit Visualization", icon="🗺️"),
        st.Page("pages/07_Experiments.py", title="07. Master Experiments Hub", icon="🧪"),
        st.Page("pages/08_Statistics.py", title="08. Statistical Rigor & Bootstrap CIs", icon="📈"),
        st.Page("pages/09_Reproducibility.py", title="09. Clean-Room Reproducibility", icon="📋"),
        st.Page("pages/10_Downloads.py", title="10. Downloads & Open Data", icon="📥"),
    ]
}

# Run Streamlit Router
pg = st.navigation(pages)
pg.run()
