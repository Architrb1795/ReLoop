"""Stratified train/val/test splitting for the TrashNet dataset.

We split by *file path*, not by copying images into separate folders. The
result is a single manifest CSV (`data/processed/splits.csv`) with columns
`path, class, split`. This keeps the ~48MB of source images in one place
(no duplication) while still giving every downstream step (Dataset classes,
notebooks, reproducibility checks) a single source of truth for which image
belongs to which split.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.dataset_info import scan_dataset
from src.utils.config import (
    RANDOM_SEED,
    SPLITS_MANIFEST_PATH,
    TEST_RATIO,
    TRAIN_RATIO,
    TRASHNET_DIR,
    VAL_RATIO,
)


def create_stratified_split(
    root_dir: str | Path = TRASHNET_DIR,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Scan `root_dir` and assign every image to train/val/test.

    Stratified sampling (via `stratify=`) ensures each split preserves the
    original per-class proportions — important here because `trash` already
    has only 137 images; a non-stratified random split could easily leave it
    under- or over-represented in one split by chance.

    Splitting happens in two stages because `train_test_split` only splits
    one set into two: first carve off `train_ratio` for training, then split
    the remainder evenly between val and test according to their relative
    ratios.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9, "Ratios must sum to 1.0"

    scan = scan_dataset(root_dir)
    metadata = scan.metadata[["path", "class"]].copy()

    train_df, remainder_df = train_test_split(
        metadata,
        train_size=train_ratio,
        stratify=metadata["class"],
        random_state=seed,
    )

    # remainder_df holds (val_ratio + test_ratio) of the data; split it so the
    # two halves come out proportional to their share of that remainder.
    relative_val_size = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        remainder_df,
        train_size=relative_val_size,
        stratify=remainder_df["class"],
        random_state=seed,
    )

    train_df = train_df.assign(split="train")
    val_df = val_df.assign(split="val")
    test_df = test_df.assign(split="test")

    manifest = pd.concat([train_df, val_df, test_df], ignore_index=True)
    return manifest


def save_split_manifest(manifest: pd.DataFrame, out_path: str | Path = SPLITS_MANIFEST_PATH) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_path, index=False)
    return out_path


def load_split_manifest(
    path: str | Path = SPLITS_MANIFEST_PATH,
    root_dir: str | Path = TRASHNET_DIR,
) -> pd.DataFrame:
    """Load the split manifest CSV and resolve image paths against the local
    `TRASHNET_DIR` if the recorded absolute paths came from another machine.
    This preserves the exact stratified train/val/test assignment across machines.
    """
    df = pd.read_csv(path)
    root_path = Path(root_dir)

    def _resolve_path(row) -> str:
        raw_p = Path(str(row["path"]).replace("\\", "/"))
        if raw_p.exists():
            return str(raw_p.resolve())
        candidate = root_path / str(row["class"]) / raw_p.name
        if candidate.exists():
            return str(candidate.resolve())
        return str(raw_p)

    if len(df) > 0:
        first_p = Path(str(df.iloc[0]["path"]))
        if not first_p.exists():
            df["path"] = df.apply(_resolve_path, axis=1)
    return df


def split_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    """Cross-tab of image counts per (split, class) plus per-split proportions."""
    counts = manifest.groupby(["split", "class"]).size().unstack(fill_value=0)
    counts = counts.reindex(["train", "val", "test"])
    proportions = counts.div(counts.sum(axis=1), axis=0).round(3)
    return counts, proportions


if __name__ == "__main__":
    manifest = create_stratified_split()
    path = save_split_manifest(manifest)
    counts, proportions = split_summary(manifest)
    print(f"Manifest saved to {path}")
    print("\nCounts:\n", counts)
    print("\nClass proportions per split:\n", proportions)
