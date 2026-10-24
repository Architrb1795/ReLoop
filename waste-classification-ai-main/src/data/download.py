"""Download and extract the TrashNet dataset (resized variant).

TrashNet's GitHub repo (https://github.com/garythung/trashnet) ships a
pre-resized copy of the dataset as `data/dataset-resized.zip` (~41 MB), which
avoids downloading the much larger original full-resolution image set. This
script shallow-clones the repo into a temporary location, extracts just that
zip into `data/raw/dataset-resized/`, and discards everything else (git
history, Lua training scripts) that isn't needed for this project.

Usage:
    python -m src.data.download
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import DATA_RAW_DIR, TRASHNET_DIR  # noqa: E402

TRASHNET_REPO_URL = "https://github.com/garythung/trashnet.git"


def download_trashnet(force: bool = False) -> Path:
    """Ensure the resized TrashNet dataset exists under `data/raw/dataset-resized/`.

    Returns the path to the extracted dataset. If it already exists and
    `force` is False, skips downloading entirely (idempotent — safe to call
    every time a pipeline runs).
    """
    if TRASHNET_DIR.exists() and not force:
        print(f"Dataset already present at {TRASHNET_DIR}, skipping download.")
        return TRASHNET_DIR

    if TRASHNET_DIR.exists() and force:
        shutil.rmtree(TRASHNET_DIR)

    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        clone_dir = Path(tmp_dir) / "trashnet"
        print(f"Cloning {TRASHNET_REPO_URL} (shallow)...")
        subprocess.run(
            ["git", "clone", "--depth", "1", TRASHNET_REPO_URL, str(clone_dir)],
            check=True,
        )

        zip_path = clone_dir / "data" / "dataset-resized.zip"
        if not zip_path.exists():
            raise FileNotFoundError(f"Expected zip not found at {zip_path}")

        print(f"Extracting {zip_path.name}...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(DATA_RAW_DIR)

    macosx_dir = DATA_RAW_DIR / "__MACOSX"
    if macosx_dir.exists():
        shutil.rmtree(macosx_dir)
    for ds_store in DATA_RAW_DIR.rglob(".DS_Store"):
        ds_store.unlink()

    print(f"Dataset ready at {TRASHNET_DIR}")
    return TRASHNET_DIR


if __name__ == "__main__":
    download_trashnet()
