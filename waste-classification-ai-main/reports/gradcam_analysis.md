# Grad-CAM Analysis — ResNet50 (Final Model)

Grad-CAM was computed on the target layer `model.layer4[-1]` (ResNet50's final residual
block, right before global average pooling) using the **saved ResNet50 checkpoint** — no
retraining. 10 examples were generated: one high-confidence correct prediction per class
(6 images) plus 4 incorrect predictions drawn from the Phase 5 misclassification list.
Full set in `reports/figures/gradcam/`.

**What Grad-CAM shows, and what it doesn't:** these heatmaps visualize which spatial
regions contributed most to the model's score for its predicted class — they are not proof
that those regions are the *correct* or *causally necessary* reason for the prediction, and
they reflect whatever the model learned, including any spurious correlations it may have
picked up. Treat this as "where the model looked," not "why the model is right."

## Findings

**The model consistently focuses on the photographed object itself, not the background.**
Across all 10 examples — correct and incorrect — the heatmap is centered on the object
(the cardboard box's textured/printed region, the crumpled trash bag's body, the protein
tub's label and cap) rather than the plain studio background or shadow regions. This is a
meaningfully good sign: it means ResNet50 is not "cheating" by exploiting some background
artifact (e.g., a shadow shape or lighting gradient) as a shortcut — the object's own visual
properties are driving the prediction, which supports the model's ability to generalize
better than if it were keying off incidental scene properties.

**Within the object, attention tends toward high-texture or label/print regions rather than
uniform surfaces.** The cardboard example (`01_...cardboard_to_cardboard.png`) shows the
heatmap concentrated on the printed text and tape seam, not the plain brown cardboard area
beside it — the model is drawn to the most information-dense part of the object, which
makes intuitive sense for a texture/pattern-based classifier.

**On errors, the model is looking at the right place but reading the wrong material cue.**
This is the most informative finding from combining Grad-CAM with Phase 5's misclassified
examples:
- `07_incorrect_plastic_to_metal.png`: a round plastic lid with a swirled, shiny surface.
  Grad-CAM centers tightly on the lid's reflective swirl pattern — exactly the region a
  human would also look at — but that swirled shininess reads as "metal" to the model.
  **This is a genuine visual-similarity error, not a localization failure.**
- `09_incorrect_plastic_to_glass.png`: a black plastic tub with a glossy label and a
  reflective highlight streak. Grad-CAM attends to the label body and the bottle-like neck/
  cap — both are real evidence for "glass bottle" shape and reflectivity. The model isn't
  distracted; it's making a defensible (if wrong) call based on the object's genuinely
  glass-bottle-like shape and gloss.
- The `paper→trash` errors show heatmaps spread more diffusely across the crumpled paper's
  folds rather than one sharp focal point — consistent with Phase 5's observation that these
  are lower-confidence, genuinely ambiguous cases (a crumpled, texture-less mass looks similar
  regardless of whether it started as paper or generic trash).

**No evidence of background, shadow, or surface artifacts driving predictions.** None of
the 10 examples showed hot regions on the plain background, drop shadow, or work surface —
had that appeared, it would be a red flag that the model learned a dataset-specific shortcut
(e.g., "this exact background shade means class X") rather than genuine object features,
which would be a serious generalization risk for real-world deployment. That risk hasn't
disappeared (TrashNet's studio backgrounds are still very uniform — see Phase 1 EDA), but at
least within this dataset, the model isn't relying on the background itself.

## Conclusion

ResNet50's attention is consistently object-centered and texture/pattern-driven, both when
correct and when wrong. Its errors are best explained not by poor localization but by
**genuine material ambiguity** — reflective plastic that looks metallic, glossy black
plastic that looks like glass, crumpled paper that looks like generic trash — the same
conclusion Phase 5's error analysis reached independently from the confusion matrix and
misclassified examples alone. The two analyses corroborate each other.
