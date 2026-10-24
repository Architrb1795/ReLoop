"""Waste Classification AI -- Streamlit demo app.

Loads the final trained model (ResNet50, see reports/model_comparison.md for why it was
selected) and classifies an uploaded image into one of six waste categories. This app never
trains or fine-tunes anything -- it only loads a saved checkpoint and runs inference, using
the exact same preprocessing pipeline (src/data/transforms.py) used during evaluation.
"""
import json
import sys
import time
from pathlib import Path

import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.transforms import get_eval_transforms  # noqa: E402  (same pipeline as training/eval)
from src.models.transfer import build_transfer_model  # noqa: E402
from src.utils.config import MODELS_DIR  # noqa: E402
from src.utils.device import get_device  # noqa: E402

METADATA_PATH = MODELS_DIR / "model_metadata.json"

st.set_page_config(page_title="Waste Classification AI", page_icon=":recycle:", layout="centered")


@st.cache_resource(show_spinner=False)
def load_metadata() -> dict | None:
    if not METADATA_PATH.exists():
        return None
    with open(METADATA_PATH) as f:
        return json.load(f)


@st.cache_resource(show_spinner="Loading model...")
def load_model(metadata: dict):
    """Loads the model once per app session (cached across reruns/users)."""
    device = get_device()
    checkpoint_path = PROJECT_ROOT / metadata["checkpoint_path"]
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at {checkpoint_path}. "
            f"Train the model first (see notebooks/03b_transfer_learning_final.ipynb) "
            f"before running this app."
        )
    model = build_transfer_model(
        metadata["model_name"], num_classes=metadata["num_classes"], pretrained=False
    )
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device


def predict(model, device, image: Image.Image, metadata: dict) -> tuple[list[tuple[str, float]], float]:
    """Runs inference on a single PIL image. Returns (sorted class-probability pairs,
    inference latency in ms). Uses the exact same eval transform as Phase 3/4 training.
    """
    transform = get_eval_transforms(img_size=metadata["input_size"][0])
    input_tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0]
    if device.type == "cuda":
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - t0) * 1000

    class_names = metadata["class_names"]
    pairs = sorted(zip(class_names, probs.cpu().tolist()), key=lambda x: -x[1])
    return pairs, latency_ms


def main():
    st.title("Waste Classification AI")
    st.write("Upload an image of waste and the model will classify it into one of six categories.")

    metadata = load_metadata()
    if metadata is None:
        st.error(
            f"Model metadata not found at `{METADATA_PATH.relative_to(PROJECT_ROOT)}`. "
            f"This file is required to know which model/classes/preprocessing to use. "
            f"Run the Phase 10 metadata script before launching this app."
        )
        st.stop()

    try:
        model, device = load_model(metadata)
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:
        st.error(f"Failed to load the model: {e}")
        st.stop()

    st.caption(
        f"Model: **{metadata['model_name']}** &nbsp;|&nbsp; "
        f"Device: **{device.type.upper()}**"
        + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")
    )

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is None:
        return

    try:
        image = Image.open(uploaded_file)
        image.verify()
        uploaded_file.seek(0)
        image = Image.open(uploaded_file).convert("RGB")
    except UnidentifiedImageError:
        st.error("This file doesn't look like a valid image. Please upload a JPG, JPEG, or PNG file.")
        st.stop()
    except Exception as e:
        st.error(f"Could not read the uploaded file: {e}")
        st.stop()

    st.image(image, caption="Uploaded image", use_container_width=True)

    try:
        predictions, latency_ms = predict(model, device, image, metadata)
    except Exception as e:
        st.error(f"Inference failed unexpectedly: {e}")
        st.stop()

    top_class, top_confidence = predictions[0]

    st.subheader(f"Prediction: {top_class.capitalize()}")
    st.write(f"Confidence: **{top_confidence * 100:.1f}%**")

    st.markdown("**Top 3 Predictions**")
    for class_name, prob in predictions[:3]:
        st.write(f"{class_name.capitalize():<12s} {prob * 100:.1f}%")

    st.markdown("**All Class Probabilities**")
    for class_name, prob in predictions:
        st.progress(prob, text=f"{class_name.capitalize()}: {prob * 100:.1f}%")

    st.caption(f"Model: {metadata['model_name']} &nbsp;|&nbsp; Inference latency: {latency_ms:.1f} ms")


if __name__ == "__main__":
    main()
