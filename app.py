"""Streamlit web app for interactive crop disease detection.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import numpy as np
import streamlit as st
from PIL import Image

import config
from src.treatments import get_treatment, is_healthy

st.set_page_config(
    page_title="Crop Disease Detector",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Confidence below this is flagged as uncertain.
UNCERTAIN_THRESHOLD = 0.50

# --- Styling -----------------------------------------------------------------
CUSTOM_CSS = """
<style>
    /* Hide default Streamlit chrome for a cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1100px;}

    /* Hero header */
    .hero {
        background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 45%, #43a047 100%);
        padding: 2.2rem 2.4rem;
        border-radius: 18px;
        color: #ffffff;
        box-shadow: 0 10px 30px rgba(27, 94, 32, 0.25);
        margin-bottom: 1.8rem;
    }
    .hero h1 {
        color: #ffffff;
        font-size: 2.2rem;
        margin: 0 0 .4rem 0;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .hero p {
        color: rgba(255,255,255,0.92);
        font-size: 1.05rem;
        margin: 0;
        max-width: 640px;
    }

    /* Generic card */
    .card {
        background: #ffffff;
        border: 1px solid #e6e9ec;
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        box-shadow: 0 4px 14px rgba(0,0,0,0.05);
        margin-bottom: 1.1rem;
    }

    /* Result banner */
    .result-banner {
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.1rem;
        color: #ffffff;
    }
    .result-banner .label {font-size: 0.95rem; opacity: 0.9; margin-bottom: .2rem;}
    .result-banner .value {font-size: 1.9rem; font-weight: 700; line-height: 1.15;}
    .banner-healthy  {background: linear-gradient(135deg, #2e7d32, #43a047);}
    .banner-disease  {background: linear-gradient(135deg, #c62828, #ef5350);}
    .banner-uncertain{background: linear-gradient(135deg, #ef6c00, #fb8c00);}

    /* Confidence bar */
    .conf-wrap {margin-top: .9rem;}
    .conf-track {
        background: rgba(255,255,255,0.3);
        border-radius: 999px;
        height: 10px;
        width: 100%;
        overflow: hidden;
    }
    .conf-fill {
        background: #ffffff;
        height: 100%;
        border-radius: 999px;
    }
    .conf-text {font-size: 0.85rem; opacity: 0.95; margin-top: .35rem;}

    /* Treatment steps */
    .treat-step {
        display: flex;
        align-items: flex-start;
        gap: .6rem;
        padding: .5rem 0;
        border-bottom: 1px solid #f0f2f4;
        font-size: 0.97rem;
    }
    .treat-step:last-child {border-bottom: none;}
    .treat-step .dot {
        color: #2e7d32;
        font-weight: 700;
        margin-top: 1px;
    }
    .pill {
        display: inline-block;
        background: #eef7ee;
        color: #1b5e20;
        padding: .25rem .7rem;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: .6rem;
    }
    .muted {color: #6b7280; font-size: 0.85rem;}
    /* Treatment card text */
    .treat-summary {font-weight: 600; font-size: 1.08rem; margin-bottom: .3rem; color: #1f2937;}
    .treat-cause {color: #6b7280; font-size: 0.9rem; margin-bottom: 1rem;}
    .treat-steps-label {
        font-size: 0.8rem; font-weight: 700; letter-spacing: .5px;
        text-transform: uppercase; color: #2e7d32; margin-bottom: .3rem;
    }
    .uncertain-note {
        background: #fff4e5; color: #9a4a00; border-radius: 10px;
        padding: .6rem .8rem; font-size: 0.9rem; margin-bottom: .9rem;
    }

    /* Empty state */
    .empty-state {text-align: center; padding: 3rem 1.6rem;}
    .empty-icon {font-size: 3rem;}
    .empty-title {font-weight: 600; font-size: 1.1rem; margin-top: .5rem; color: #1f2937;}

    /* Ranked probabilities with mini bars */
    .rank-row {
        display: grid; grid-template-columns: 1fr 90px 48px; align-items: center;
        gap: .7rem; font-size: 0.9rem; padding: .35rem 0;
    }
    .rank-name {color: #374151;}
    .rank-bar {background: #eef2f5; border-radius: 999px; height: 8px; overflow: hidden;}
    .rank-bar-fill {display: block; height: 100%; background: linear-gradient(90deg, #43a047, #2e7d32); border-radius: 999px;}
    .rank-val {color: #6b7280; font-variant-numeric: tabular-nums; text-align: right;}
</style>
"""


@st.cache_resource(show_spinner=False)
def _load_model_and_classes():
    """Load the trained model and class names once, cached across reruns."""
    import tensorflow as tf

    from src.data_loader import load_class_names

    model = tf.keras.models.load_model(config.MODEL_PATH)
    class_names = load_class_names()
    return model, class_names


def _preprocess(pil_image: Image.Image) -> np.ndarray:
    img = pil_image.convert("RGB").resize(config.IMG_SIZE)
    arr = np.asarray(img, dtype=np.float32)  # [0, 255]
    return np.expand_dims(arr, axis=0)


def _pretty(label: str) -> str:
    """Turn 'Tomato___Early_blight' into 'Tomato · Early blight'."""
    return label.replace("___", " · ").replace("_", " ")


def _render_hero() -> None:
    html = (
        '<div class="hero">'
        "<h1>🌱 Crop Disease Detection</h1>"
        "<p>Upload a photo of a crop leaf and get an instant AI diagnosis with "
        "tailored treatment guidance. Powered by deep learning "
        "(MobileNetV2 transfer learning).</p>"
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def _render_sidebar(class_names: list[str]) -> None:
    with st.sidebar:
        st.markdown("### ℹ️ About")
        st.write(
            "This tool classifies crop leaf images and recommends treatment. "
            "It is for educational use — confirm serious cases with a local "
            "agricultural extension service."
        )
        st.markdown("### 🧪 How to use")
        st.markdown(
            "1. Upload a clear, well-lit leaf photo.\n"
            "2. Read the diagnosis and confidence.\n"
            "3. Follow the recommended steps."
        )
        st.markdown(f"### 🗂️ Detectable classes ({len(class_names)})")
        for name in class_names:
            st.markdown(f"<span class='muted'>• {_pretty(name)}</span>", unsafe_allow_html=True)


def _render_result_banner(top_label: str, top_conf: float) -> None:
    """Color-coded banner: healthy / disease / uncertain."""
    if top_conf < UNCERTAIN_THRESHOLD:
        banner_cls, label_txt = "banner-uncertain", "Uncertain — please verify"
    elif is_healthy(top_label):
        banner_cls, label_txt = "banner-healthy", "Looks healthy 🌿"
    else:
        banner_cls, label_txt = "banner-disease", "Disease detected 🧪"

    pct = top_conf * 100
    html = (
        f'<div class="result-banner {banner_cls}">'
        f'<div class="label">{label_txt}</div>'
        f'<div class="value">{_pretty(top_label)}</div>'
        f'<div class="conf-wrap">'
        f'<div class="conf-track"><div class="conf-fill" style="width:{pct:.1f}%"></div></div>'
        f'<div class="conf-text">{pct:.1f}% confidence</div>'
        f'</div></div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def _render_treatment(top_label: str, uncertain: bool) -> None:
    t = get_treatment(top_label)
    steps_html = "".join(
        f'<div class="treat-step"><span class="dot">✓</span><span>{step}</span></div>'
        for step in t["steps"]
    )
    note = ""
    if uncertain:
        note = (
            '<div class="uncertain-note">⚠️ Low confidence — treat this as a '
            "suggestion and verify with another photo or an expert.</div>"
        )
    html = (
        '<div class="card">'
        '<span class="pill">🩺 Recommended treatment</span>'
        f"{note}"
        f'<div class="treat-summary">{t["summary"]}</div>'
        f'<div class="treat-cause"><em>Cause:</em> {t["cause"]}</div>'
        '<div class="treat-steps-label">What to do</div>'
        f"{steps_html}"
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def _render_probabilities(class_names: list[str], probs: np.ndarray) -> None:
    ranked = sorted(zip(class_names, probs), key=lambda x: x[1], reverse=True)[:5]
    rows = "".join(
        f'<div class="rank-row"><span class="rank-name">{_pretty(n)}</span>'
        f'<span class="rank-bar"><span class="rank-bar-fill" style="width:{p*100:.1f}%"></span></span>'
        f'<span class="rank-val">{p*100:.1f}%</span></div>'
        for n, p in ranked
    )
    html = (
        '<div class="card">'
        '<span class="pill">📊 Top matches</span>'
        f"{rows}"
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def main() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    _render_hero()

    if not config.MODEL_PATH.exists() or not config.CLASS_NAMES_PATH.exists():
        st.error(
            "No trained model found. Train one first:\n\n"
            "```\npython -m src.download_data --crop Tomato\npython -m src.train\n```"
        )
        st.stop()

    with st.spinner("Loading model..."):
        model, class_names = _load_model_and_classes()

    _render_sidebar(class_names)

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown("#### 📤 Upload a leaf image")
        uploaded = st.file_uploader(
            "Drag and drop or browse",
            type=["png", "jpg", "jpeg", "bmp", "webp"],
            label_visibility="collapsed",
        )
        if uploaded is not None:
            image = Image.open(uploaded)
            st.image(image, caption="Uploaded image", width="stretch")

    with right:
        if uploaded is None:
            empty_html = (
                '<div class="card empty-state">'
                '<div class="empty-icon">🍃</div>'
                '<div class="empty-title">Awaiting an image</div>'
                '<div class="muted">Upload a crop leaf photo to see the '
                "diagnosis and treatment here.</div>"
                "</div>"
            )
            st.markdown(empty_html, unsafe_allow_html=True)
            return

        with st.spinner("Analyzing..."):
            batch = _preprocess(image)
            probs = model.predict(batch, verbose=0)[0]

        top_idx = int(np.argmax(probs))
        top_label = class_names[top_idx]
        top_conf = float(probs[top_idx])
        uncertain = top_conf < UNCERTAIN_THRESHOLD

        _render_result_banner(top_label, top_conf)
        _render_treatment(top_label, uncertain)
        _render_probabilities(class_names, probs)

    st.caption(
        "General guidance for educational use. Always follow product labels and "
        "local regulations, and consult your agricultural extension service for a "
        "confirmed diagnosis."
    )


if __name__ == "__main__":
    main()
