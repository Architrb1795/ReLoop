"""
ReLoop — Production Application & Unified AI Inference Server
=============================================================
Serves the ReLoop web application (`index.html`, `styles.css`, `app.js`) and exposes
unified JSON API endpoints powered by `ReLoopVisionEngine` (PyTorch ResNet50):

  - GET  /api/health       -> Model & checkpoint readiness status
  - GET  /api/state        -> Student wallet, impact metrics, and recent disposal activity
  - GET  /api/test-sample  -> Loads a real held-out test image (or OOD stone / poor-quality sample)
                              and runs it through the exact unified inference pipeline
  - POST /api/predict      -> Unified inference for uploaded images & live camera frames
  - POST /api/deposit      -> Verified (simulated) RVM disposal & reward transaction creation
"""
from __future__ import annotations

import base64
import io
import json
import mimetypes
import os
import sys
import time
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

WORKSPACE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from hardware_and_ai.rvm_ai_engine import RELOOP_CLASS_CONFIG, ReLoopVisionEngine

# Global Vision Engine Singleton (loaded once at server startup)
VISION_ENGINE: ReLoopVisionEngine | None = None

# In-memory store of verified predictions eligible for single-use deposit
PENDING_PREDICTIONS: dict[str, dict] = {}

# Student Account / Demo State (Explicitly marked as Demo Baseline + Live Verified Session)
STUDENT_STATE = {
    "is_demo_account": True,
    "student_name": "Archit (Campus ID #RL-2048)",
    "points_balance": 1280,
    "next_reward_target": 1600,
    "waste_disposed_kg": 12.4,
    "material_recovered_kg": 8.7,
    "successful_disposals": 24,
    "recent_activity": [
        {
            "id": "tx_demo_01",
            "item_label": "Plastic Bottle",
            "category": "plastic",
            "points": 20,
            "weight_kg": 0.025,
            "timestamp": "Demo Baseline • Verified RVM-01",
            "status": "VERIFIED_DEMO",
        },
        {
            "id": "tx_demo_02",
            "item_label": "Paper Cup",
            "category": "paper",
            "points": 10,
            "weight_kg": 0.015,
            "timestamp": "Demo Baseline • Verified RVM-02",
            "status": "VERIFIED_DEMO",
        },
        {
            "id": "tx_demo_03",
            "item_label": "Aluminium Can",
            "category": "metal",
            "points": 30,
            "weight_kg": 0.018,
            "timestamp": "Demo Baseline • Verified RVM-01",
            "status": "VERIFIED_DEMO",
        },
    ],
}


def generate_stone_test_image() -> Image.Image:
    """Creates a realistic granular grey/slate stone image for OOD rejection testing."""
    rng = np.random.default_rng(42)
    arr = np.full((384, 512, 3), 218, dtype=np.uint8)
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    cx, cy = 256, 192
    rx, ry = 115, 92
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    pts = []
    for a in angles:
        r_j = rng.uniform(0.86, 1.12)
        pts.append((cx + rx * r_j * np.cos(a), cy + ry * r_j * np.sin(a)))
    draw.polygon(pts, fill=(108, 106, 102))

    np_im = np.array(img, dtype=np.int16)
    mask = np_im[:, :, 0] != 218
    speckle = rng.normal(0, 24, size=(384, 512, 3)).astype(np.int16)
    np_im[mask] = np.clip(np_im[mask] + speckle[mask], 35, 185)
    return Image.fromarray(np_im.astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=0.7))


def generate_dark_blurry_test_image() -> Image.Image:
    """Creates a severely underexposed/dark frame for Image Quality Gate testing."""
    rng = np.random.default_rng(99)
    arr = rng.integers(3, 15, size=(384, 512, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def image_to_data_url(img: Image.Image, fmt: str = "JPEG") -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format=fmt, quality=88)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "image/jpeg" if fmt.upper() == "JPEG" else "image/png"
    return f"data:{mime};base64,{b64}"


REPRESENTATIVE_HELDOUT_TEST_FILES: dict[str, str] = {
    "cardboard": "cardboard60.jpg",
    "glass": "glass130.jpg",
    "metal": "metal386.jpg",
    "paper": "paper157.jpg",
    "plastic": "plastic429.jpg",
    "trash": "trash74.jpg",
}


def find_heldout_test_sample(waste_class: str) -> tuple[Image.Image, str]:
    """Selects a real held-out test split image from `waste-classification-ai-main`."""
    raw_dir = WORKSPACE_ROOT / "waste-classification-ai-main" / "data" / "raw" / "dataset-resized"
    preferred_name = REPRESENTATIVE_HELDOUT_TEST_FILES.get(waste_class)
    if preferred_name:
        preferred_path = raw_dir / waste_class / preferred_name
        if preferred_path.exists():
            return Image.open(preferred_path).convert("RGB"), preferred_path.name

    manifest_path = (
        WORKSPACE_ROOT / "waste-classification-ai-main" / "data" / "processed" / "splits.csv"
    )
    if manifest_path.exists():
        import pandas as pd

        df = pd.read_csv(manifest_path)
        sub = df[(df["split"] == "test") & (df["class"] == waste_class)]
        for _, row in sub.iterrows():
            cand = raw_dir / waste_class / Path(str(row["path"]).replace("\\", "/")).name
            if cand.exists():
                return Image.open(cand).convert("RGB"), cand.name

    # Fallback: scan directory directly
    cls_dir = raw_dir / waste_class
    if cls_dir.exists():
        files = sorted(cls_dir.glob("*.jpg"))
        if files:
            chosen = files[min(10, len(files) - 1)]
            return Image.open(chosen).convert("RGB"), chosen.name

    raise FileNotFoundError(f"No sample image found for class '{waste_class}'")


class ReLoopRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WORKSPACE_ROOT), **kwargs)

    def log_message(self, format: str, *args) -> None:
        # Keep console logs clean
        pass

    def _send_json(self, status_code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            self._send_json(
                200,
                {
                    "status": "ok",
                    "engine_ready": VISION_ENGINE is not None,
                    "checkpoint": VISION_ENGINE.active_checkpoint_name if VISION_ENGINE else None,
                    "device": str(VISION_ENGINE.device) if VISION_ENGINE else None,
                    "supported_classes": list(RELOOP_CLASS_CONFIG.keys()),
                },
            )
            return

        if parsed.path == "/api/state":
            self._send_json(200, STUDENT_STATE)
            return

        if parsed.path == "/api/test-sample":
            qs = parse_qs(parsed.query)
            sample_type = (qs.get("type", ["plastic"])[0] or "plastic").lower()
            try:
                if sample_type == "stone":
                    img = generate_stone_test_image()
                    fname = "ood_stone_sample.jpg"
                elif sample_type == "poor_quality":
                    img = generate_dark_blurry_test_image()
                    fname = "dark_unusable_frame.jpg"
                elif sample_type in RELOOP_CLASS_CONFIG:
                    img, fname = find_heldout_test_sample(sample_type)
                else:
                    self._send_json(400, {"error": f"Unsupported sample type: {sample_type}"})
                    return

                result = VISION_ENGINE.predict(img, source="upload", filename_hint=fname)
                result["image_data_url"] = image_to_data_url(img)
                result["sample_filename"] = fname
                PENDING_PREDICTIONS[result["prediction_id"]] = result
                self._send_json(200, result)
            except Exception as exc:
                self._send_json(500, {"status": "ERROR", "reason": str(exc)})
            return

        return super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        content_len = int(self.headers.get("Content-Length", "0"))
        if content_len > 12 * 1024 * 1024:
            self._send_json(
                413,
                {
                    "status": "ERROR",
                    "reason": "Uploaded file exceeds the 10 MB size limit.",
                },
            )
            return

        raw_body = self.rfile.read(content_len) if content_len > 0 else b"{}"
        try:
            data = json.loads(raw_body.decode("utf-8"))
        except Exception:
            self._send_json(400, {"status": "ERROR", "reason": "Invalid JSON request body."})
            return

        if parsed.path == "/api/predict":
            img_b64 = data.get("image_base64", "")
            source = data.get("source", "upload")
            filename = data.get("filename", "")

            if not img_b64 or not isinstance(img_b64, str):
                self._send_json(
                    400, {"status": "ERROR", "reason": "No image payload provided."}
                )
                return

            try:
                if "," in img_b64:
                    header, b64_part = img_b64.split(",", 1)
                else:
                    b64_part = img_b64
                img_bytes = base64.b64decode(b64_part)
                img = Image.open(io.BytesIO(img_bytes))
                img.verify()
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception:
                self._send_json(
                    400,
                    {
                        "status": "ERROR",
                        "reason": "Invalid or corrupted image file. Please upload a valid JPG, PNG, or WEBP image.",
                    },
                )
                return

            result = VISION_ENGINE.predict(img, source=source, filename_hint=filename)
            PENDING_PREDICTIONS[result["prediction_id"]] = result
            # Keep memory bounded
            if len(PENDING_PREDICTIONS) > 200:
                oldest_key = next(iter(PENDING_PREDICTIONS))
                PENDING_PREDICTIONS.pop(oldest_key, None)

            self._send_json(200, result)
            return

        if parsed.path == "/api/deposit":
            pred_id = data.get("prediction_id", "")
            pred = PENDING_PREDICTIONS.get(pred_id)
            if not pred:
                self._send_json(
                    400,
                    {
                        "status": "ERROR",
                        "reason": "No active verified prediction found for this deposit request.",
                    },
                )
                return

            if pred.get("deposited"):
                self._send_json(
                    400,
                    {
                        "status": "ERROR",
                        "reason": "This item has already been deposited and credited.",
                    },
                )
                return

            if not pred.get("acceptable_for_segregation") or pred.get("status") != "RECOGNIZED":
                self._send_json(
                    400,
                    {
                        "status": "REJECTED",
                        "reason": (
                            f"Cannot issue points or actuate sorting gate for {pred.get('status')} item. "
                            "Item routed to Safe Reject Compartment."
                        ),
                    },
                )
                return

            # Mark prediction as consumed so it cannot be re-deposited
            pred["deposited"] = True

            pts = int(pred.get("reward_points_eligible", 0))
            w_kg = float(pred.get("estimated_weight_kg", 0.02))
            rec_kg = float(pred.get("estimated_recovered_kg", 0.015))

            STUDENT_STATE["points_balance"] += pts
            STUDENT_STATE["waste_disposed_kg"] = round(
                STUDENT_STATE["waste_disposed_kg"] + w_kg, 3
            )
            STUDENT_STATE["material_recovered_kg"] = round(
                STUDENT_STATE["material_recovered_kg"] + rec_kg, 3
            )
            STUDENT_STATE["successful_disposals"] += 1

            tx = {
                "id": f"tx_{int(time.time())}",
                "item_label": pred["short_label"],
                "display_label": pred["display_label"],
                "category": pred["predicted_class"],
                "points": pts,
                "weight_kg": w_kg,
                "confidence": pred["confidence"],
                "timestamp": datetime.now().strftime("Today, %I:%M:%S %p") + " • Simulated RVM Deposit",
                "status": "SIMULATED_VERIFIED",
            }
            STUDENT_STATE["recent_activity"].insert(0, tx)

            self._send_json(
                200,
                {
                    "status": "DEPOSITED",
                    "transaction": tx,
                    "student_state": STUDENT_STATE,
                },
            )
            return

        self._send_json(404, {"status": "ERROR", "reason": "Endpoint not found."})


def run_server(port: int = 8080) -> None:
    global VISION_ENGINE
    print("Initializing ReLoop Vision Engine (PyTorch ResNet50)...")
    VISION_ENGINE = ReLoopVisionEngine()
    print(
        f"ReLoop Vision Engine loaded ({VISION_ENGINE.active_checkpoint_name}) on {VISION_ENGINE.device}"
    )
    server = ThreadingHTTPServer(("0.0.0.0", port), ReLoopRequestHandler)
    print(f"ReLoop Server listening at http://localhost:{port}")
    server.serve_forever()


if __name__ == "__main__":
    port_num = int(os.environ.get("PORT", "8080"))
    run_server(port=port_num)
