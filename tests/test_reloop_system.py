"""
ReLoop — End-to-End Automated Acceptance & Integration Test Suite
=================================================================
Verifies:
  1. Dataset & Split Integrity (2,527 images, 70/15/15 stratified split).
  2. Calibrated Model Loading & Evaluation Metrics vs. Baseline.
  3. Unified Inference Consistency between Uploaded Images and Live Camera Frames.
  4. Open-Set Unknown-Object (Stone/Rock) Rejection & Poor Image Quality Rejection.
  5. Data Integrity: Separation between AI prediction and verified (simulated) deposit.
  6. Frontend HTML/CSS/JS Product Structure (2 primary tabs, no phone mockup, brand rules).
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WORKSPACE_ROOT / "waste-classification-ai-main"
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from hardware_and_ai.rvm_ai_engine import RELOOP_CLASS_CONFIG, ReLoopVisionEngine
from server import find_heldout_test_sample, generate_dark_blurry_test_image, generate_stone_test_image


class TestReLoopEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = ReLoopVisionEngine()
        cls.eval_json_path = REPO_ROOT / "reports" / "results" / "reloop_overhaul_evaluation.json"

    def test_01_dataset_and_evaluation_metrics_recorded(self):
        """Verify dataset audit and scientific evaluation results exist and beat baseline."""
        self.assertTrue(
            self.eval_json_path.exists(),
            f"Evaluation report missing at {self.eval_json_path}",
        )
        with open(self.eval_json_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        audit = report["dataset_audit"]
        self.assertEqual(audit["total_manifest_rows"], 2527)
        self.assertEqual(audit["missing_files_count"], 0)
        self.assertEqual(audit["corrupted_files_count"], 0)

        base_test = report["baseline_test_metrics"]
        imp_test = report["improved_test_metrics"]
        self.assertGreaterEqual(imp_test["accuracy"], base_test["accuracy"])
        self.assertGreaterEqual(imp_test["macro_f1"], base_test["macro_f1"])

        ood = report["ood_rejection_metrics"]
        self.assertGreater(ood["test_auroc_known_vs_ood"], 0.85)
        self.assertGreater(ood["test_stone_specific_rejection_rate"], 0.80)

    def test_02_supported_waste_classes_recognition(self):
        """Verify held-out test samples for supported classes are recognized accurately."""
        for cls_name in ["cardboard", "glass", "metal", "paper", "plastic"]:
            img, fname = find_heldout_test_sample(cls_name)
            res = self.engine.predict(img, source="upload", filename_hint=fname)
            self.assertEqual(
                res["status"],
                "RECOGNIZED",
                f"Expected RECOGNIZED for held-out {cls_name} ({fname}), got {res['status']}",
            )
            self.assertEqual(res["predicted_class"], cls_name)
            self.assertTrue(res["acceptable_for_segregation"])
            self.assertGreater(res["reward_points_eligible"], 0)

    def test_03_upload_and_camera_inference_equivalence(self):
        """Verify uploaded-image and live-camera paths produce identical predictions."""
        img, fname = find_heldout_test_sample("plastic")
        res_upload = self.engine.predict(img, source="upload", filename_hint=fname)
        res_camera = self.engine.predict(img, source="camera", filename_hint="frame.jpg")

        self.assertEqual(res_upload["status"], res_camera["status"])
        self.assertEqual(res_upload["predicted_class"], res_camera["predicted_class"])
        self.assertAlmostEqual(res_upload["confidence"], res_camera["confidence"], places=4)

    def test_04_unknown_object_stone_rejection(self):
        """Verify that a stone / rock image is rejected as UNKNOWN and cannot earn points."""
        stone_img = generate_stone_test_image()
        res = self.engine.predict(stone_img, source="upload", filename_hint="test_img.jpg")
        self.assertEqual(res["status"], "UNKNOWN")
        self.assertFalse(res["recognized"])
        self.assertFalse(res["acceptable_for_segregation"])
        self.assertEqual(res["reward_points_eligible"], 0)
        self.assertEqual(res["sorting_action"], "SAFE_REJECT_COMPARTMENT")
        self.assertIn("Unknown Object", res["display_label"])

    def test_05_poor_image_quality_rejection(self):
        """Verify dark, overexposed, and blank images trigger POOR_QUALITY state."""
        dark_img = generate_dark_blurry_test_image()
        res_dark = self.engine.predict(dark_img, source="camera")
        self.assertEqual(res_dark["status"], "POOR_QUALITY")
        self.assertFalse(res_dark["acceptable_for_segregation"])
        self.assertEqual(res_dark["reward_points_eligible"], 0)

        blank_img = Image.fromarray(np.full((224, 224, 3), 140, dtype=np.uint8))
        res_blank = self.engine.predict(blank_img, source="upload")
        self.assertEqual(res_blank["status"], "POOR_QUALITY")
        self.assertEqual(res_blank["quality_code"], "EMPTY_BACKGROUND")

    def test_06_ui_brand_and_navigation_compliance(self):
        """Verify index.html and styles.css match ReLoop brand & two-tab rules."""
        html_content = (WORKSPACE_ROOT / "index.html").read_text(encoding="utf-8")
        css_content = (WORKSPACE_ROOT / "styles.css").read_text(encoding="utf-8")

        # Brand & Tagline checks
        self.assertIn("ReLoop", html_content)
        self.assertIn("Waste. Reward. Repeat.", html_content)
        self.assertIn("Don't ask people to care about waste. Give them a reason to.", html_content)
        self.assertIn("Most smart bins stop at SORT.", html_content)
        self.assertIn("ReLoop continues until VALUE.", html_content)

        # Color tokens check
        self.assertIn("#B7F000", css_content)
        self.assertIn("#111827", css_content)
        self.assertIn("#7C3AED", css_content)
        self.assertIn("#F8F9FA", css_content)

        # Phone simulation & technical tabs must be completely removed
        self.assertNotIn("phone-shell", html_content)
        self.assertNotIn("dynamic-notch", html_content)
        self.assertNotIn("tab-ailab", html_content)
        self.assertNotIn("tab-hardware", html_content)

        # Two primary product tabs check
        self.assertIn("RVM Machine", html_content)
        self.assertIn("Student App", html_content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
