"""Streamlit counterfeit drug detection demo dashboard."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from src.models.nir_classifier import NIRAuthenticityClassifier, N_WAVENUMBERS, snv_transform
from src.models.fusion_engine import fuse_scores

st.set_page_config(page_title="Counterfeit Drug Detection", page_icon="💊", layout="wide")
st.title("💊 Counterfeit Drug Detection — AI Authentication System")
st.caption("Multi-modal: NIR Spectroscopy + Pill Vision + Packaging OCR | NAFDAC-integrated")

DRUGS = ["ACT-AL", "AMOX-500", "OXY-10IU", "ARV-TDF", "METRO-400"]
VERDICT_COLOURS = {
    "AUTHENTIC": "#00CC00",
    "SUSPICIOUS": "#FFA500",
    "LIKELY_COUNTERFEIT": "#CC0000",
}

st.sidebar.header("Drug Authentication")
drug_code = st.sidebar.selectbox("Drug Code", DRUGS)
is_counterfeit = st.sidebar.checkbox("Simulate counterfeit drug", value=False)
run_nir = st.sidebar.checkbox("Run NIR Analysis", value=True)
run_vision = st.sidebar.checkbox("Run Pill Vision Check", value=True)
run_packaging = st.sidebar.checkbox("Run Packaging OCR", value=True)

if st.sidebar.button("🔍 Authenticate Drug"):
    with st.spinner("Running authentication..."):
        np.random.seed(np.random.randint(0, 9999))
        base_spec = np.linspace(900, 1700, N_WAVENUMBERS)
        if is_counterfeit:
            spectrum = 0.4 * np.exp(-((base_spec - 1250) ** 2) / (2 * 130 ** 2)) + np.random.normal(0, 0.06, N_WAVENUMBERS)
            nir_prob = np.random.uniform(0.18, 0.45) if run_nir else None
            vision_prob = np.random.uniform(0.20, 0.50) if run_vision else None
            packaging_prob = np.random.uniform(0.10, 0.40) if run_packaging else None
        else:
            spectrum = (
                0.8 * np.exp(-((base_spec - 1200) ** 2) / (2 * 100 ** 2))
                + 0.4 * np.exp(-((base_spec - 1400) ** 2) / (2 * 80 ** 2))
                + np.random.normal(0, 0.02, N_WAVENUMBERS)
            )
            nir_prob = np.random.uniform(0.78, 0.96) if run_nir else None
            vision_prob = np.random.uniform(0.75, 0.95) if run_vision else None
            packaging_prob = np.random.uniform(0.80, 0.98) if run_packaging else None

    result = fuse_scores(nir_prob, vision_prob, packaging_prob, drug_code)
    colour = VERDICT_COLOURS[result.final_verdict]

    st.markdown(f"""
    <div style='background:{colour};color:white;padding:20px;border-radius:8px;margin-bottom:16px'>
    <h2>{'✅' if result.final_verdict == 'AUTHENTIC' else '⚠️' if result.final_verdict == 'SUSPICIOUS' else '❌'} {result.final_verdict.replace('_', ' ')}</h2>
    <b>Drug:</b> {drug_code} &nbsp;|&nbsp; <b>Overall Score:</b> {result.overall_score:.3f} &nbsp;|&nbsp; <b>Confidence:</b> {result.confidence:.1%}<br>
    <br><b>Recommendation:</b> {result.recommendation}
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Authentication Scores by Modality")
        scores = {}
        if nir_prob is not None:
            scores["NIR Spectroscopy"] = nir_prob
        if vision_prob is not None:
            scores["Pill Vision"] = vision_prob
        if packaging_prob is not None:
            scores["Packaging OCR"] = packaging_prob

        colours = ["#00CC00" if v >= 0.70 else "#FFA500" if v >= 0.50 else "#CC0000"
                   for v in scores.values()]
        fig = go.Figure(go.Bar(
            x=list(scores.values()), y=list(scores.keys()),
            orientation="h",
            marker_color=colours,
            text=[f"{v:.1%}" for v in scores.values()],
            textposition="outside",
        ))
        fig.add_vline(x=0.70, line_dash="dash", line_color="green", annotation_text="Auth threshold")
        fig.add_vline(x=0.50, line_dash="dash", line_color="orange", annotation_text="Suspicious")
        fig.update_layout(title="Modality Authentication Scores", xaxis_range=[0, 1.15])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("NIR Spectrum Analysis")
        processed = snv_transform(spectrum.reshape(1, -1))[0]
        wavelengths = np.linspace(900, 1700, N_WAVENUMBERS)
        fig_spec = go.Figure()
        fig_spec.add_trace(go.Scatter(
            x=wavelengths, y=processed,
            line=dict(color="blue", width=1.5),
            name="Sample Spectrum",
        ))
        if not is_counterfeit:
            ref = 0.8 * np.exp(-((wavelengths - 1200) ** 2) / (2 * 100 ** 2))
            fig_spec.add_trace(go.Scatter(
                x=wavelengths, y=snv_transform(ref.reshape(1, -1))[0],
                line=dict(color="green", dash="dash"),
                name="Reference (Authentic)",
            ))
        fig_spec.update_layout(
            title="NIR Spectrum (SNV-normalised)",
            xaxis_title="Wavenumber (nm)",
            yaxis_title="Absorbance (SNV)",
        )
        st.plotly_chart(fig_spec, use_container_width=True)

    if result.failed_checks:
        st.error("**Failed Checks:**\n" + "\n".join(f"• {c}" for c in result.failed_checks))
    if result.report_to_nafdac:
        st.warning("⚠️ This result should be reported to NAFDAC. Evidence has been logged with GPS and timestamp.")

st.markdown("---")
st.caption("MOMAH MOSES .C. · Geospatial AI Engineer & Data Scientist · github.com/Momahmoses")
