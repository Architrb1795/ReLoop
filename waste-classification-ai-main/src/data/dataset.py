"""PyTorch Dataset for TrashNet, backed by the split manifest CSV."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from src.data.splitter import load_split_manifest
from src.utils.config import CLASS_NAMES, SPLITS_MANIFEST_PATH


class WasteDataset(Dataset):
    """Loads TrashNet images for one split ("train", "val", or "test").

    Reads image paths and labels from the manifest produced by
    `src.data.splitter`, rather than relying on `torchvision.datasets.ImageFolder`
    over physically-separated directories. This avoids duplicating the
    source images on disk while still giving each split a fixed, versioned
    set of files (the manifest CSV).
    """

    def __init__(
        self,
        split: str,
        transform=None,
        manifest_path: str | Path = SPLITS_MANIFEST_PATH,
        manifest: pd.DataFrame | None = None,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be one of 'train'/'val'/'test', got {split!r}")

        full_manifest = manifest if manifest is not None else load_split_manifest(manifest_path)
        self.data = full_manifest[full_manifest["split"] == split].reset_index(drop=True)
        self.transform = transform
        self.class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        row = self.data.iloc[index]
        image = Image.open(row["path"]).convert("RGB")
        label = self.class_to_idx[row["class"]]

        if self.transform is not None:
            image = self.transform(image)

        return image, label
