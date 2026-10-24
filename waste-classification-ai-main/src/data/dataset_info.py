"""Utilities for scanning and verifying the TrashNet image dataset.

These functions are read-only: they inspect the dataset on disk and return
structured results (a pandas DataFrame of per-image metadata, plus summary
dicts). Visualization and interpretation happen in the notebooks.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass
class DatasetScanResult:
    """Container for the outcome of scanning a class-folder image dataset."""

    metadata: pd.DataFrame
    corrupted_files: list[str] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)


def scan_dataset(root_dir: str | Path) -> DatasetScanResult:
    """Walk a directory of `<root_dir>/<class_name>/<image>` and collect metadata.

    For every image found, records: file path, class label, file extension,
    file size in bytes, width, height, and color mode (e.g. "RGB", "L").
    Files that fail to open as images are recorded separately as corrupted
    rather than raising, so one bad file doesn't stop the scan.
    """
    root_dir = Path(root_dir)
    classes = sorted([d.name for d in root_dir.iterdir() if d.is_dir()])

    records = []
    corrupted_files = []

    for class_name in classes:
        class_dir = root_dir / class_name
        for file_path in sorted(class_dir.iterdir()):
            if file_path.suffix.lower() not in VALID_EXTENSIONS:
                continue
            try:
                with Image.open(file_path) as img:
                    img.verify()
                # Re-open after verify() (which leaves the file unusable for further ops)
                with Image.open(file_path) as img:
                    width, height = img.size
                    mode = img.mode
            except (UnidentifiedImageError, OSError):
                corrupted_files.append(str(file_path))
                continue

            records.append(
                {
                    "path": str(file_path),
                    "class": class_name,
                    "extension": file_path.suffix.lower(),
                    "file_size_kb": file_path.stat().st_size / 1024,
                    "width": width,
                    "height": height,
                    "mode": mode,
                }
            )

    metadata = pd.DataFrame.from_records(records)
    return DatasetScanResult(metadata=metadata, corrupted_files=corrupted_files, classes=classes)


def class_distribution(metadata: pd.DataFrame) -> pd.Series:
    """Return image count per class, sorted descending."""
    return metadata["class"].value_counts().sort_values(ascending=False)


def imbalance_ratio(metadata: pd.DataFrame) -> float:
    """Ratio of largest class count to smallest class count.

    A value close to 1.0 means classes are balanced; values above ~2-3
    typically warrant mitigation (class weights, oversampling, or augmentation
    focused on minority classes).
    """
    counts = class_distribution(metadata)
    return float(counts.max() / counts.min())


def summarize(scan: DatasetScanResult) -> dict:
    """Build a plain-dict summary of a DatasetScanResult, suitable for printing/logging."""
    df = scan.metadata
    return {
        "total_images": len(df),
        "num_classes": len(scan.classes),
        "classes": scan.classes,
        "images_per_class": class_distribution(df).to_dict(),
        "imbalance_ratio": imbalance_ratio(df) if len(df) else None,
        "num_corrupted": len(scan.corrupted_files),
        "corrupted_files": scan.corrupted_files,
        "extensions": df["extension"].value_counts().to_dict() if len(df) else {},
        "color_modes": df["mode"].value_counts().to_dict() if len(df) else {},
        "unique_dimensions": df[["width", "height"]].drop_duplicates().shape[0] if len(df) else 0,
        "width_range": (int(df["width"].min()), int(df["width"].max())) if len(df) else None,
        "height_range": (int(df["height"].min()), int(df["height"].max())) if len(df) else None,
        "file_size_kb_mean": float(df["file_size_kb"].mean()) if len(df) else None,
    }
