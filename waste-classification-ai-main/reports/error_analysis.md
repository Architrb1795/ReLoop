# Error Analysis — ResNet50 (Final Model)

Evaluated the **saved ResNet50 checkpoint** (`models/resnet50_best.pt`) on the existing test
split (380 images, same seed=42, same preprocessing, same evaluation protocol as the
official Phase 4 run). Re-verified test accuracy: **91.05%**, exactly matching the official
result — confirms no drift or retraining occurred.

Total misclassified: **34 / 380 images (8.9%)**.

## 1. Main Findings

- Errors are **spread thinly across many different confusion pairs** rather than
  concentrated in one dominant failure mode — the largest single pair (`glass→metal` and
  `paper→trash`, tied at 4 errors each) accounts for less than a quarter of all errors.
- A meaningful fraction of errors are **high-confidence mistakes on genuinely ambiguous
  source images** (see §5) — composite materials (a paper-exterior bubble mailer, a
  plastic-wrapped cardboard box) where the *photographed object* itself belongs to more
  than one category, not clear model failures.
- **`Trash` is the clear weak point** — not because the model doesn't recognize it (recall
  is 95%, tied for the best of any class) but because it has by far the lowest **precision**
  (63.3%) — many non-trash images get pulled into the `trash` bucket.

## 2. Most Confused Class Pairs

| Actual → Predicted | Count |
|---|---:|
| glass → metal | 4 |
| paper → trash | 4 |
| plastic → metal | 3 |
| plastic → glass | 3 |
| cardboard → paper | 3 |
| plastic → trash | 3 |
| metal → trash | 2 |
| cardboard → trash | 2 |
| glass → plastic | 2 |
| cardboard → glass | 1 |

## 3. Strongest Classes

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| **Cardboard** | 0.964 | 0.900 | **0.931** |
| **Paper** | 0.953 | 0.910 | **0.931** |
| Metal | 0.881 | 0.952 | 0.915 |
| Glass | 0.921 | 0.921 | 0.921 |

Cardboard and Paper tie for the best F1. Cardboard's strength matches the Phase 1 EDA
prediction (a distinctive, consistent brown fibrous texture). Metal has the *best recall*
of any class (95.2%) — the model rarely misses an actual metal item, though a modest
number of non-metal images get misclassified as metal (dragging precision down to 88.1%).

## 4. Weakest Class

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| **Trash** | **0.633** | 0.950 | **0.760** |

`Trash` is weakest by a wide margin — its F1 (0.760) is 0.15+ below every other class. This
is a **precision problem, not a recall problem**: the model almost never *misses* a real
trash image (95% recall, matching the class-weighted loss doing its job — see
[reports/model_comparison.md](model_comparison.md) and `src/training/class_weights.py`),
but it over-predicts `trash` on ambiguous images from other classes, most often `paper`
(4 of the model's false-trash calls) and `plastic` (3). This is consistent across every
model trained in this project (CNN, MobileNetV2, EfficientNet-B0, ResNet50) — it is a
property of the *class*, not a specific architecture's weakness.

## 5. Representative Examples

From `reports/figures/error_analysis/misclassified_examples.png`:

| Actual | Predicted | Confidence | Likely explanation |
|---|---|---:|---|
| Plastic | Paper | 97.8% (high, wrong) | A padded/bubble mailer — paper exterior wrapping a plastic bubble-wrap interior. The *photographed object* is genuinely a paper+plastic composite; TrashNet's single-label scheme can't capture that. |
| Paper | Metal | 95.9% (high, wrong) | A glossy, reflective blue paper tube/packaging — the glossy foil-like finish visually resembles a metal can under studio lighting. |
| Plastic | Trash | 93.7% (high, wrong) | A crumpled black plastic bag — colloquially "black plastic bag" reads as generic trash even to a human, and TrashNet's `trash` catch-all category has no dedicated visual identity to distinguish it from a bag that happens to be plastic. |
| Cardboard | Trash | 84.3% (high, wrong) | A plain, texture-less brown cardboard surface with no visible box edges/structure — without a recognizable box shape, the model has little signal beyond "brown," which overlaps with worn/dirty `trash` images. |
| Plastic | Metal | 78.2% (medium, wrong) | A round object with a shiny, silvery, slightly warped surface — visually closer to a dented metal lid than typical clear plastic. |
| Paper | Trash | 44.5% (low, wrong) | A tightly crumpled ball of newsprint — genuinely ambiguous even by eye; the low confidence shows the model itself was uncertain, not confidently wrong. |

## 6. Possible Causes

Based on the actual errors observed (not assumed):

- **Composite/ambiguous source materials.** Several high-confidence errors are objects that
  are legitimately made of two materials (paper-wrapped plastic mailers, plastic-wrapped
  cardboard). TrashNet's single-label-per-image scheme cannot represent this, and no model
  can be "correct" on a genuinely ambiguous label.
- **Reflective/glossy surfaces cross material boundaries visually.** Glossy paper or
  scuffed plastic can visually mimic metal's reflectivity under the dataset's uniform
  studio lighting — this shows up directly in `paper→metal` and `plastic→metal` errors.
- **`Trash` has no consistent visual identity.** As a catch-all category (Phase 1 EDA
  already flagged this), it has no unifying texture/color/shape, so the model instead
  falls back on rougher heuristics (crumpled, dark, textureless) that overlap with other
  classes' outliers — explaining why `trash` pulls in stray `paper` and `plastic` images.
- **Featureless close-ups lose class-defining structure.** The cardboard→trash error shows
  a plain, texture-less brown surface with no visible box edges — cropped close enough that
  the shape cue that normally identifies cardboard (a rectangular box) is absent.
- **Class imbalance's residual effect.** `Trash` has the fewest training images (96) of any
  class even after class-weighting; despite the weighting successfully preventing the model
  from *ignoring* trash (95% recall), it has proportionally less exposure to what
  "non-trash" looks like at the margins, contributing to lower precision.

## 7. Potential Improvements

- **Multi-label or soft-labeling** for genuinely composite objects, rather than forcing a
  single hard label — even just flagging known-ambiguous training images could reduce noisy
  gradient signal.
- **More `trash` training examples**, ideally with more visual diversity — the current 96
  images are unlikely to cover the category's inherent breadth.
- **Targeted augmentation for reflective/glossy confusions** — e.g., more aggressive color
  jitter specifically to reduce the model's reliance on glossiness as a metal cue.
- **A second-stage "high-uncertainty" review path**: since several of ResNet50's errors are
  low-confidence (e.g., the 44.5% paper→trash case), a confidence threshold could flag
  uncertain predictions for manual review in a real deployment rather than forcing a
  guess.
- **Real-world background diversity** (Phase 1 EDA limitation, confirmed relevant here):
  all training/test images share plain studio backgrounds; several errors involve subtle
  material cues (glossiness, texture) that may not transfer to cluttered real-world scenes.
