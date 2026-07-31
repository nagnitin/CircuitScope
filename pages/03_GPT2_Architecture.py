"""
pages/03_GPT2_Architecture.py
=============================
GPT-2 Small Interactive Transformer Architecture Explorer.
"""

import streamlit as st
from components.cards import render_callout
from components.utils import load_css

load_css()

st.title("🏗️ GPT-2 Small Transformer Architecture Explorer")
st.markdown("### Interactive Structural Breakdown & Residual Stream Inspection")

st.markdown("""
GPT-2 Small is an autoregressive decoder-only transformer with **85 Million parameters**, **12 Transformer Layers**, **12 Attention Heads per layer** (144 total), and a hidden dimension of $d_{model} = 768$.
""")

# Architecture Specifications Cards
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Model Dimension (d_model)", "768")
with c2:
    st.metric("Transformer Layers", "12")
with c3:
    st.metric("Total Attention Heads", "144 (12 × 12)")
with c4:
    st.metric("Head Dimension (d_head)", "64")

st.markdown("---")

st.header("🔍 Interactive Component Inspector")
st.markdown("Click on any architecture block below to view its mathematical formulation, internal dimensions, and role in the IOI sub-circuit:")

# Select Component Block
selected_block = st.radio(
    "Select Architecture Component Block:",
    [
        "🔤 1. Token & Position Embedding (W_E & W_POS)",
        "🔄 2. Residual Stream Memory Bus",
        "🎯 3. Multi-Head Self-Attention (MHSA)",
        "⚡ 4. Multilayer Perceptron (MLP) Blocks",
        "📏 5. Layer Normalization (LN)",
        "🔠 6. Unembedding & Vocab Logits (W_U)",
    ],
    horizontal=False
)

st.markdown("<br>", unsafe_allow_html=True)

if "Embedding" in selected_block:
    st.markdown("""
    <div class="paper-card">
        <h3 style="color: #38BDF8 !important;">🔤 1. Token & Position Embedding</h3>
        <p><strong>Formula:</strong> $x_0 = t W_E + p W_{POS}$</p>
        <p><strong>Dimensions:</strong> $W_E \\in \\mathbb{R}^{50257 \\times 768}$, $W_{POS} \\in \\mathbb{R}^{1024 \\times 768}$</p>
        <p><strong>Role in IOI Circuit:</strong> Converts token IDs for names (e.g. <em>John</em>, <em>Mary</em>) and structural tokens (e.g. <em>When</em>, <em>to</em>) into initial 768-dimensional vectors in the residual stream.</p>
        <p><strong>Key Finding:</strong> Early token embeddings at Layer 0 already encode position markers necessary for distinguishing S1 vs S2 positions.</p>
    </div>
    """, unsafe_allow_html=True)

elif "Residual Stream" in selected_block:
    st.markdown("""
    <div class="paper-card">
        <h3 style="color: #38BDF8 !important;">🔄 2. Residual Stream Memory Bus</h3>
        <p><strong>Formula:</strong> $x_{l} = x_{l-1} + \\text{Attn}_l(x_{l-1}) + \\text{MLP}_l(x_{l-1} + \\text{Attn}_l(x_{l-1}))$</p>
        <p><strong>Dimensions:</strong> $x_l \\in \\mathbb{R}^{seq \\times 768}$</p>
        <p><strong>Role in IOI Circuit:</strong> Acts as a shared linear memory communication channel across all 12 layers. Attention heads and MLPs read from and write to this stream via linear additions.</p>
        <p><strong>Key Finding:</strong> Logit lens shows information accumulates sequentially, jumping sharply in logit diff at Layers 7–8 and 10.</p>
    </div>
    """, unsafe_allow_html=True)

elif "Self-Attention" in selected_block:
    st.markdown("""
    <div class="paper-card">
        <h3 style="color: #38BDF8 !important;">🎯 3. Multi-Head Self-Attention (MHSA)</h3>
        <p><strong>Formula:</strong> $\\text{Head}_h(x) = \\text{Softmax}\\left(\\frac{x W_Q^h (x W_K^h)^T}{\\sqrt{d_k}}\\right) x W_V^h W_O^h$</p>
        <p><strong>Dimensions:</strong> 12 heads per layer, $d_{head} = 64$. $W_Q, W_K, W_V \\in \\mathbb{R}^{768 \\times 64}$, $W_O \\in \\mathbb{R}^{64 \\times 768}$</p>
        <p><strong>Functional Roles in IOI Circuit:</strong>
            <ul>
                <li><strong style="color: #818CF8;">Name Mover Heads (L8H6, L8H10, L5H5, L7H9):</strong> Move the Indirect Object name vector from its original position to the final token position.</li>
                <li><strong style="color: #FBBF24;">S-Inhibition Heads (L7H9, L8H10):</strong> Attend to the duplicate Subject name (S2) and suppress its attention weight in Name Movers.</li>
                <li><strong style="color: #22D3EE;">Helper / Duplicate Token Heads (L0H10, L1H10):</strong> Detect repeated tokens early in the sequence.</li>
            </ul>
        </p>
    </div>
    """, unsafe_allow_html=True)

elif "MLP" in selected_block:
    st.markdown("""
    <div class="paper-card">
        <h3 style="color: #38BDF8 !important;">⚡ 4. Multilayer Perceptron (MLP) Blocks</h3>
        <p><strong>Formula:</strong> $\\text{MLP}(x) = \\text{GELU}(x W_{in} + b_{in}) W_{out} + b_{out}$</p>
        <p><strong>Dimensions:</strong> Expansion factor 4x: $W_{in} \\in \\mathbb{R}^{768 \\times 3072}$, $W_{out} \\in \\mathbb{R}^{3072 \\times 768}$</p>
        <p><strong>Role in IOI Circuit:</strong> Computes non-linear transformations. Activation patching reveals that **Layer 0 MLP** and **Layer 5 MLP** contribute over 1.0 logit diff restoration.</p>
        <p><strong>Resample Ablation Control:</strong> Proves Layer 0 MLP is a genuine forward-pass requirement rather than a mean-ablation artifact.</p>
    </div>
    """, unsafe_allow_html=True)

elif "Layer Normalization" in selected_block:
    st.markdown("""
    <div class="paper-card">
        <h3 style="color: #38BDF8 !important;">📏 5. Layer Normalization (LN)</h3>
        <p><strong>Formula:</strong> $\\text{LN}(x) = \\frac{x - \\mu}{\\sigma} \\odot \\gamma + \\beta$</p>
        <p><strong>Role in IOI Circuit:</strong> Normalizes residual stream variance before each Attention and MLP sub-layer. Crucial to account for LN scaling factors when measuring head activations.</p>
    </div>
    """, unsafe_allow_html=True)

elif "Unembedding" in selected_block:
    st.markdown("""
    <div class="paper-card">
        <h3 style="color: #38BDF8 !important;">🔠 6. Unembedding & Vocab Logits</h3>
        <p><strong>Formula:</strong> $\\text{Logits} = \\text{LN}(x_{final}) W_U$</p>
        <p><strong>Dimensions:</strong> $W_U \\in \\mathbb{R}^{768 \\times 50257}$</p>
        <p><strong>Role in IOI Circuit:</strong> Maps the final residual stream vector at the last prompt token position to token logits over the 50,257 vocabulary entries.</p>
    </div>
    """, unsafe_allow_html=True)

render_callout(
    title="Architectural Insight",
    text="GPT-2 Small's residual stream acts as a 768-dimensional shared memory bus where attention heads read specific linear subspaces and write target token directional vectors directly towards the unembedding matrix $W_U$.",
    category="info",
    icon="💡"
)
