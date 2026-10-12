"""Streamlit web app for interactive crop disease detection.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

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
# If the gap between the top-1 and top-2 probabilities is below this, the
# prediction is ambiguous (two classes look almost equally likely) and is
# treated as uncertain even when the top confidence clears the threshold.
MARGIN_THRESHOLD = 0.15
# Reject uploads larger than this to avoid memory issues from huge images.
MAX_UPLOAD_MB = 10

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


@st.cache_data(show_spinner=False)
def _find_example_images() -> dict[str, str]:
    """Find one example image per class from the validation set.

    Returns a mapping of pretty class name -> absolute image path. Cached so the
    filesystem scan runs only once. Returns an empty dict if no data is present.
    """
    examples: dict[str, str] = {}
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for base in (config.VAL_DIR, config.TRAIN_DIR):
        if not base.exists():
            continue
        for class_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            pretty = _pretty(class_dir.name)
            if pretty in examples:
                continue
            for img in sorted(class_dir.iterdir()):
                if img.suffix.lower() in exts:
                    examples[pretty] = str(img)
                    break
    return examples


def _preprocess(pil_image: Image.Image) -> np.ndarray:
    img = pil_image.convert("RGB").resize(config.IMG_SIZE)
    arr = np.asarray(img, dtype=np.float32)  # [0, 255]
    return np.expand_dims(arr, axis=0)


def _model_view(pil_image: Image.Image) -> Image.Image:
    """Return exactly what the model sees: an RGB image resized to IMG_SIZE."""
    return pil_image.convert("RGB").resize(config.IMG_SIZE)


def _pretty(label: str) -> str:
    """Turn 'Tomato___Early_blight' into 'Tomato · Early blight'."""
    return label.replace("___", " · ").replace("_", " ")


def _load_image_safely(source) -> Image.Image | None:
    """Open an image defensively, surfacing a friendly error on failure.

    `source` may be a Streamlit UploadedFile or a filesystem path. Returns the
    loaded PIL image, or None if the file is too large, corrupt, or unreadable
    (in which case an error has already been shown to the user).
    """
    # Size guard for uploads (UploadedFile exposes .size and .name).
    size = getattr(source, "size", None)
    if size is not None and size > MAX_UPLOAD_MB * 1024 * 1024:
        st.error(
            f"That image is {size / (1024 * 1024):.1f} MB, which is larger than "
            f"the {MAX_UPLOAD_MB} MB limit. Please upload a smaller photo."
        )
        return None

    try:
        # Read raw bytes so we can both verify integrity and then reload.
        if hasattr(source, "read"):
            raw = source.read()
        else:
            raw = Path(source).read_bytes()

        # verify() detects truncated/corrupt files but consumes the image,
        # so we reopen from the same bytes afterward for actual use.
        Image.open(io.BytesIO(raw)).verify()
        image = Image.open(io.BytesIO(raw))
        image.load()
        return image
    except (UnidentifiedImageError, OSError, ValueError):
        st.error(
            "That file could not be read as an image. It may be corrupt or in an "
            "unsupported format. Please try a different photo."
        )
        return None


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


def _classify_status(top_conf: float, margin: float) -> tuple[bool, str]:
    """Decide the result status from confidence and the top-1/top-2 margin.

    Returns (uncertain, reason). `reason` is empty when the result is confident.
    """
    if top_conf < UNCERTAIN_THRESHOLD:
        return True, "low confidence"
    if margin < MARGIN_THRESHOLD:
        return True, "two classes look almost equally likely"
    return False, ""


def _render_result_banner(top_label: str, top_conf: float, uncertain: bool) -> None:
    """Color-coded banner: healthy / disease / uncertain."""
    if uncertain:
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


def _render_treatment(top_label: str, uncertain: bool, reason: str) -> None:
    t = get_treatment(top_label)
    steps_html = "".join(
        f'<div class="treat-step"><span class="dot">✓</span><span>{step}</span></div>'
        for step in t["steps"]
    )
    note = ""
    if uncertain:
        note = (
            f'<div class="uncertain-note">⚠️ Uncertain result ({reason}) — treat '
            "this as a suggestion and verify with another photo or an expert.</div>"
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


def _render_probabilities(ranked: list[tuple[str, float]]) -> None:
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


def _build_report(
    top_label: str,
    top_conf: float,
    uncertain: bool,
    reason: str,
    ranked: list[tuple[str, float]],
    source_name: str,
) -> str:
    """Build a plain-text diagnosis report for download."""
    t = get_treatment(top_label)
    status = "UNCERTAIN" if uncertain else ("HEALTHY" if is_healthy(top_label) else "DISEASE DETECTED")
    lines = [
        "CROP DISEASE DETECTION — DIAGNOSIS REPORT",
        "=" * 44,
        f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Image     : {source_name}",
        "",
        f"Status    : {status}",
        f"Diagnosis : {_pretty(top_label)}",
        f"Confidence: {top_conf * 100:.1f}%",
    ]
    if uncertain:
        lines.append(f"Note      : Result flagged uncertain ({reason}). Please verify.")
    lines += ["", "TOP MATCHES", "-" * 44]
    lines += [f"  {_pretty(n):<42} {p * 100:6.2f}%" for n, p in ranked]
    lines += [
        "",
        "RECOMMENDED TREATMENT",
        "-" * 44,
        f"Summary: {t['summary']}",
        f"Cause  : {t['cause']}",
        "Steps  :",
    ]
    lines += [f"  - {step}" for step in t["steps"]]
    lines += [
        "",
        "=" * 44,
        "General guidance for educational use. Always follow product labels and",
        "local regulations, and consult your agricultural extension service for a",
        "confirmed diagnosis.",
    ]
    return "\n".join(lines)


def _analyze_and_render(image: Image.Image, model, class_names: list[str], source_name: str) -> None:
    """Run inference on a loaded image and render the full result panel."""
    with st.spinner("Analyzing..."):
        batch = _preprocess(image)
        probs = model.predict(batch, verbose=0)[0]

    order = np.argsort(probs)[::-1]
    ranked = [(class_names[i], float(probs[i])) for i in order[:5]]
    top_label = class_names[int(order[0])]
    top_conf = float(probs[int(order[0])])
    second_conf = float(probs[int(order[1])]) if len(order) > 1 else 0.0
    margin = top_conf - second_conf

    uncertain, reason = _classify_status(top_conf, margin)

    _render_result_banner(top_label, top_conf, uncertain)
    _render_treatment(top_label, uncertain, reason)
    _render_probabilities(ranked)

    report = _build_report(top_label, top_conf, uncertain, reason, ranked, source_name)
    st.download_button(
        "⬇️ Download diagnosis report",
        data=report,
        file_name=f"diagnosis_{top_label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
        width="stretch",
    )


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

    # Session state tracks which image to analyze and where it came from.
    st.session_state.setdefault("example_path", None)

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown("#### 📤 Upload a leaf image")
        uploaded = st.file_uploader(
            "Drag and drop or browse",
            type=["png", "jpg", "jpeg", "bmp", "webp"],
            label_visibility="collapsed",
        )

        # --- Example gallery -------------------------------------------------
        examples = _find_example_images()
        if examples:
            with st.expander("🌿 Or try an example image"):
                st.caption("Pick a sample leaf to run the model without uploading.")
                choice = st.selectbox(
                    "Example class",
                    options=["—"] + list(examples.keys()),
                    label_visibility="collapsed",
                )
                if choice != "—":
                    if st.button("Use this example", width="stretch"):
                        st.session_state["example_path"] = examples[choice]

        # Decide the active image source: a fresh upload wins over an example.
        image = None
        source_name = ""
        if uploaded is not None:
            st.session_state["example_path"] = None  # upload overrides example
            image = _load_image_safely(uploaded)
            source_name = uploaded.name
        elif st.session_state["example_path"]:
            path = st.session_state["example_path"]
            image = _load_image_safely(path)
            source_name = Path(path).name

        if image is not None:
            st.image(image, caption=f"Input: {source_name}", width="stretch")
            # Show exactly what the model sees (RGB, resized to IMG_SIZE).
            with st.expander("🔍 What the model sees"):
                st.image(
                    _model_view(image),
                    caption=f"Preprocessed to {config.IMG_WIDTH}×{config.IMG_HEIGHT} RGB",
                    width=config.IMG_WIDTH,
                )

    with right:
        if image is None:
            empty_html = (
                '<div class="card empty-state">'
                '<div class="empty-icon">🍃</div>'
                '<div class="empty-title">Awaiting an image</div>'
                '<div class="muted">Upload a crop leaf photo (or try an example) '
                "to see the diagnosis and treatment here.</div>"
                "</div>"
            )
            st.markdown(empty_html, unsafe_allow_html=True)
        else:
            _analyze_and_render(image, model, class_names, source_name)

    st.caption(
        "General guidance for educational use. Always follow product labels and "
        "local regulations, and consult your agricultural extension service for a "
        "confirmed diagnosis."
    )


if __name__ == "__main__":
    main()
