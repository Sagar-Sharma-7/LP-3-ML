from __future__ import annotations

from pathlib import Path
import io
import math
import tempfile

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from pe_screening import create_report, load_image, predict_image, simple_heatmap

st.set_page_config(page_title="PE Screening Workspace", page_icon="", layout="wide", initial_sidebar_state="expanded")
MODEL_PATH = Path("models/pe_screening_model.joblib")

st.markdown("""
<style>
    :root { --ink: #edf0f5; --muted: #a4acb9; --paper: #0e1117; --panel: #171b23; --sand: #1d222c; --line: #2b3340; --accent: #a8b8ff; --accent-dark: #c5d0ff; }
    .stApp { background: var(--paper); color: var(--ink); }
    [data-testid="stHeader"] { background: rgba(14, 17, 23, 0.94); border-bottom: 1px solid rgba(43, 51, 64, 0.8); }
    [data-testid="stToolbar"], [data-testid="stHeaderActionElements"] { display: none; }
    [data-testid="stSidebar"] { background: #131821; border-right: 1px solid var(--line); }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.3rem; }
    .block-container { max-width: 1240px; padding: 2.4rem 3rem 3rem; }
    h1, h2, h3, p, label, span, div { font-family: Geist, Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    code, pre, [data-testid="stCode"] { font-family: "Geist Mono", "SFMono-Regular", Consolas, "Liberation Mono", monospace !important; }
    h1 { color: var(--ink); font-size: 2.35rem !important; font-weight: 650 !important; letter-spacing: -0.035em; margin-bottom: 0.25rem !important; }
    h2 { color: var(--ink); font-size: 1.42rem !important; font-weight: 620 !important; padding-top: 1rem; }
    h3 { color: var(--ink); font-weight: 600 !important; }
    .eyebrow { color: var(--accent); font-family: Geist, Inter, ui-sans-serif, sans-serif; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.14em; margin-bottom: 0.45rem; }
    .subtitle { color: var(--muted); font-size: 1.04rem; max-width: 700px; margin-bottom: 1.6rem; }
    .disclaimer { background: #1a202b; border-left: 3px solid var(--accent); color: #cbd3df; padding: 0.85rem 1rem; border-radius: 4px; font-size: 0.92rem; margin: 0.4rem 0 1.8rem; }
    .sidebar-label { color: var(--accent); font-family: Geist, Inter, ui-sans-serif, sans-serif; font-size: 0.68rem; font-weight: 700; letter-spacing: 0.12em; margin: 0.7rem 0 0.35rem; }
    .sidebar-title { color: var(--ink); font-size: 1.22rem; font-weight: 600; margin-bottom: 0.25rem; }
    .sidebar-copy { color: var(--muted); font-size: 0.84rem; line-height: 1.45; margin-bottom: 1.1rem; }
    [data-testid="stMetric"] { background: var(--panel); border: 1px solid var(--line); border-radius: 7px; padding: 1rem 1.1rem; }
    [data-testid="stMetricLabel"] { color: var(--muted); font-family: Geist, Inter, ui-sans-serif, sans-serif; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; }
    [data-testid="stMetricValue"] { color: var(--ink); font-family: Geist, Inter, ui-sans-serif, sans-serif; }
    [data-testid="stFileUploader"] { background: var(--panel); border: 1px dashed #54627b; border-radius: 7px; padding: 0.65rem; }
    .stButton > button, .stDownloadButton > button { background: var(--accent); color: #182033; border: 0; border-radius: 4px; font-family: Geist, Inter, ui-sans-serif, sans-serif; font-size: 0.84rem; font-weight: 650; padding: 0.55rem 0.9rem; }
    .stButton > button:hover, .stDownloadButton > button:hover { background: var(--accent-dark); color: #111827; border: 0; }
    [data-baseweb="radio"] label { color: var(--ink); font-family: Geist, Inter, ui-sans-serif, sans-serif; font-size: 0.9rem; }
    [data-testid="stExpander"] { background: var(--panel); border: 1px solid var(--line); border-radius: 6px; }
    hr { border-color: var(--line); }
    #MainMenu, footer { visibility: hidden; }
    .section-note { color: var(--muted); font-size: 0.94rem; margin-top: -0.25rem; margin-bottom: 1.2rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_model(path: str, modified: float):
    return joblib.load(path)


def current_model():
    if not MODEL_PATH.exists():
        return None
    return get_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)


def probability_label(probability: float) -> tuple[str, str]:
    if probability >= 0.65:
        return "Higher model score", "This image should be prioritised for qualified clinical review."
    if probability >= 0.40:
        return "Intermediate model score", "The prototype is uncertain; interpret only alongside clinical review."
    return "Lower model score", "A low model score cannot rule out pulmonary embolism."


def render_image_prediction(name: str, loaded, model) -> dict:
    probability = predict_image(model, loaded.image)
    label, description = probability_label(probability)
    st.subheader(name)
    left, middle, right = st.columns([1, 1, 0.8])
    with left:
        st.image(loaded.image, clamp=True, caption="Normalised input image", use_container_width=True)
    with middle:
        heatmap = simple_heatmap(model, loaded.image)
        figure, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(loaded.image, cmap="gray")
        ax.imshow(heatmap, cmap="jet", alpha=0.38)
        ax.axis("off")
        ax.set_title("Occlusion sensitivity aid (not lesion localization)")
        st.pyplot(figure, clear_figure=True)
    with right:
        st.metric("PE likelihood score", f"{probability:.1%}")
        st.write(f"**{label}**")
        st.caption(description)
        with st.expander("Input metadata"):
            st.json(loaded.metadata)
    return {"image": name, "pe_probability": probability, "classification": label}


def load_uploaded(uploaded) -> tuple[str, object]:
    suffix = Path(uploaded.name).suffix or ".img"
    if suffix.lower() == ".dcm":
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(uploaded.getvalue())
            temp_path = handle.name
        try:
            loaded = load_image(temp_path, uploaded.name)
        finally:
            Path(temp_path).unlink(missing_ok=True)
    else:
        loaded = load_image(io.BytesIO(uploaded.getvalue()), uploaded.name)
    return uploaded.name, loaded


st.markdown("<div class='eyebrow'>ACADEMIC RESEARCH PROJECT</div>", unsafe_allow_html=True)
st.title("Pulmonary Embolism Screening")
st.markdown("<div class='subtitle'>A compact CT-slice analysis workspace for reviewing model scores, test results, and supporting visual evidence.</div>", unsafe_allow_html=True)
st.markdown("<div class='disclaimer'>This academic prototype is not a medical device. It must not be used to diagnose, exclude, or guide treatment for pulmonary embolism.</div>", unsafe_allow_html=True)

model = current_model()
if model is None:
    st.error("No trained model found. Run `python generate_demo_dataset.py`, then `python train_model.py`, or prepare an approved dataset subset first.")
    st.stop()

metrics = model.get("metrics", {})
with st.sidebar:
    st.markdown("<div class='sidebar-label'>PE SCREENING WORKSPACE</div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-title'>Local model review</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='sidebar-copy'>{metrics.get('samples', 'Unknown')} labelled slices · patient-level evaluation</div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-label'>SECTIONS</div>", unsafe_allow_html=True)
    page = st.radio("Navigation", ["Screening workspace", "Performance review", "Method and data"], label_visibility="collapsed")

if page == "Screening workspace":
    st.header("Image review")
    st.markdown("<div class='section-note'>Upload individual DICOM slices or prepared CT images. For responsive CPU analysis, up to 40 images are reviewed at one time.</div>", unsafe_allow_html=True)
    files = st.file_uploader("Add CT images", type=["dcm", "png", "jpg", "jpeg"], accept_multiple_files=True)
    if files:
        with st.spinner("Loading and scoring selected images…"):
            loaded_files = []
            for uploaded in files[:40]:
                try:
                    loaded_files.append(load_uploaded(uploaded))
                except Exception as exc:
                    st.error(f"Could not load {uploaded.name}: {exc}")
        if loaded_files:
            st.divider()
            records = []
            for name, loaded in loaded_files:
                records.append(render_image_prediction(name, loaded, model))
            report_frame, pdf_data = create_report("Uploaded images", records, metrics)
            st.divider()
            st.subheader("Study summary")
            highest = report_frame.pe_probability.max()
            top_mean = report_frame.nlargest(min(3, len(report_frame)), "pe_probability").pe_probability.mean()
            study_score = 0.70 * highest + 0.30 * top_mean
            summary_label, summary_description = probability_label(study_score)
            first, second, third = st.columns(3)
            first.metric("Aggregate study score", f"{study_score:.1%}")
            second.metric("Highest slice score", f"{highest:.1%}")
            third.metric("Images analysed", len(report_frame))
            st.markdown(f"<div class='disclaimer'><strong>{summary_label}.</strong> {summary_description}</div>", unsafe_allow_html=True)
            csv = report_frame.to_csv(index=False).encode("utf-8")
            download_left, download_right = st.columns(2)
            download_left.download_button("Download prediction CSV", csv, "prediction_results.csv", "text/csv")
            download_right.download_button("Download academic PDF report", pdf_data, "pe_screening_report.pdf", "application/pdf")
    else:
        st.markdown("<div class='disclaimer'>Select prepared FUMPE images from <code>data/prepared/images</code> to begin a real-data demonstration.</div>", unsafe_allow_html=True)

elif page == "Performance review":
    st.header("Performance review")
    st.markdown(f"<div class='section-note'>Metrics are from {metrics.get('validation_scheme', 'the latest evaluation')}. They measure this small research dataset only and do not establish clinical performance.</div>", unsafe_allow_html=True)
    columns = st.columns(5)
    for column, (name, value) in zip(columns, [("Accuracy", metrics.get("accuracy", 0)), ("Precision", metrics.get("precision", 0)), ("Recall", metrics.get("recall", 0)), ("F1", metrics.get("f1", 0)), ("ROC-AUC", metrics.get("roc_auc", 0))]):
        column.metric(name, f"{value:.1%}")
    st.subheader("Evaluation context")
    left, right = st.columns(2)
    left.metric("Labelled CT slices", metrics.get("samples", 0))
    right.metric("Patients held out across folds", metrics.get("held_out_studies", 0))
    st.markdown("<div class='disclaimer'>The model uses fixed CTPA windowing, image-intensity statistics, gradient features, histograms, and a compact 32 × 32 image representation with logistic regression. Patient-level separation prevents neighbouring slices from the same person appearing in both training and evaluation.</div>", unsafe_allow_html=True)

else:
    st.header("Method and data")
    st.markdown("""
    ### Data and storage
    This project uses a compact FUMPE subset: twelve real CT pulmonary-angiography studies with expert PE masks. The working image set is kept small enough for local CPU training.

    ### Processing approach
    CT intensities are displayed with a fixed pulmonary-angiography window. Expert masks create slice labels, and all patient slices remain together within either a training or evaluation fold.

    ### Appropriate use
    The system is designed for academic demonstration of medical-image preprocessing and machine learning. It is not a clinical diagnostic workflow.
    """)
