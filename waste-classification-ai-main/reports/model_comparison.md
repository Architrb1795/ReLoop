# Model Comparison & Final Selection

All metrics below are the **official Phase 3/4 results** (unchanged) plus **newly measured
inference benchmarks** (Phase 7 — measured on the saved checkpoints, no retraining, RTX 4050,
10 warm-up runs excluded from timing, 100 timed single-image runs, `torch.cuda.synchronize()`
around every timed call). Full raw numbers: `reports/results/phase7_inference_benchmark.json`.

## Full Comparison Table

| Model | Accuracy | Macro F1 | Weighted F1 | Parameters | Model Size | Single-Image Latency (mean / median) | Batch-32 (per-image) | Training Time | Deployment Considerations |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| CNN | 63.68% | 0.621 | 0.650 | 390,918 | 1.51 MB | 1.093 / 0.762 ms | 0.938 ms | ~796s | Lightweight but lower accuracy |
| MobileNetV2 | 82.11% | 0.805 | 0.823 | 2,231,558 | 8.75 MB | 6.314 / 6.140 ms | 1.027 ms | 572s | Lightweight, strong accuracy/size trade-off |
| EfficientNet-B0 | 80.53% | 0.788 | 0.808 | 4,015,234 | 15.61 MB | 12.591 / 11.307 ms | 1.301 ms | 560s | Moderate size, slowest single-image latency of the three TL models |
| **ResNet50** | **91.05%** | **0.894** | **0.913** | 23,520,326 | 90.03 MB | 7.267 / 6.897 ms | 3.056 ms | 679s | Highest accuracy, heaviest model |

## Reading the Benchmarks — a Non-Obvious Finding

**Parameter count does not predict single-image latency here.** EfficientNet-B0 (4.0M
params) is the *slowest* model for single-image inference (12.6ms mean) — slower than
ResNet50 (23.5M params, 7.3ms) despite having roughly 1/6th the parameters. This is because
EfficientNet-B0's architecture (many sequential depthwise-separable + squeeze-excitation
blocks) issues far more, smaller GPU kernel launches per forward pass; at batch size 1, that
per-kernel launch overhead dominates over raw compute, and the RTX 4050 sits under-utilized
between launches. ResNet50's fewer, larger, more GPU-friendly convolutions pipeline more
efficiently even though there's more raw compute per image.

This gap narrows sharply at batch size 32 (EfficientNet-B0: 1.301ms/image, ResNet50:
3.056ms/image — ResNet50 becomes clearly the most expensive per-image, as parameter count
would predict) because launch overhead gets amortized across the batch. **The practical
takeaway: for a single-image-at-a-time Streamlit upload workflow (this project's actual
deployment shape), EfficientNet-B0's parameter-count advantage over ResNet50 does not
translate into a latency advantage — if anything it's slightly worse.** This is exactly why
the brief warns not to assume a winner before measuring.

All four models' single-image latency is well under 15ms on this hardware — for a
human-facing "upload one image, see a result" UI, the difference between 1ms and 13ms is
imperceptible; none of these models is the bottleneck for that use case.

## Weighing the Trade-offs

- **CNN**: fastest and smallest by a wide margin, but 27+ accuracy points and 0.27+ macro F1
  behind every transfer-learning model. Valuable as the from-scratch baseline and teaching
  artifact, not viable as the deployed classifier.
- **MobileNetV2**: the best *small-model* trade-off — smallest of the three transfer-learning
  models (8.75MB), second-fastest single-image latency, and actually the *best* accuracy/F1
  among MobileNetV2 vs. EfficientNet-B0 (82.11% vs 80.53%). If model size/footprint were the
  binding constraint, EfficientNet-B0 would be hard to justify over it on this dataset.
- **EfficientNet-B0**: does not lead on any axis measured here — not the most accurate, not
  the smallest, not the fastest. Included for completeness per the project brief, but the
  benchmarks don't support choosing it over MobileNetV2 for a lightweight deployment.
- **ResNet50**: clearly the most accurate (+8.9 points over MobileNetV2, +0.089 macro F1),
  and — per the latency finding above — not meaningfully slower than the other
  transfer-learning models for single-image inference on this GPU. Its cost is model
  size (90MB vs. 8.75MB) and higher batch-throughput cost, not single-request latency.

## Final Decision

### Scenario A — Maximum Accuracy
**ResNet50.** Confirmed by the benchmarks, not assumed: it wins on every accuracy metric,
and — contrary to the naive assumption that a 23.5M-parameter model must be slower —
its single-image latency (7.3ms) is actually *faster* than EfficientNet-B0's (12.6ms) and
not meaningfully different from MobileNetV2's (6.3ms) for the single-upload use case this
project targets. The only real cost is the 90MB checkpoint vs. 8.75MB.

### Scenario B — Lightweight / Constrained Deployment
**MobileNetV2.** If the deployment target has tight storage/bandwidth constraints (e.g., an
edge device, or repeated redistribution of the model file matters), MobileNetV2 is the
correct choice — smallest file size, competitive latency, and the best accuracy of the two
"lightweight" candidates (beats EfficientNet-B0 on every metric while also being smaller).

### Selected for This Project's Streamlit Deployment: **ResNet50**

This project's Streamlit app (Phase 9) runs locally on the same RTX 4050 used for training —
there is no edge-device or bandwidth constraint driving Scenario B here, and single-image
latency (the metric that actually matters for a one-image-at-a-time upload UI) is not a
meaningful differentiator between ResNet50 and the smaller models on this hardware. Given
that, **accuracy and F1 are the deciding factors, and ResNet50 leads decisively (91.05%
accuracy, 0.894 macro F1) without a latency penalty for this use case.** MobileNetV2 remains
documented as the recommended alternative if this project were redeployed under storage or
edge-device constraints (Scenario B).
