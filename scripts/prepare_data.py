"""Download VisDrone2019-DET from the HF mirror and build the Roboflow-layout
dataset directory that rfdetr auto-detects:

    dataset/
      train/  _annotations.coco.json + *.jpg (symlinks into data/)
      valid/  _annotations.coco.json + *.jpg
      test/   _annotations.coco.json + *.jpg (mirror of valid; unused by default)

Raw VisDrone .txt annotations stay under data/ — the taxonomy reads the
truncation/occlusion fields the converter copies into the COCO JSONs.
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

from huggingface_hub import hf_hub_download  # noqa: E402

from scripts.visdrone_to_coco import convert_split  # noqa: E402

SPLITS = {
    # roboflow split dir -> VisDrone split name
    "train": "VisDrone2019-DET-train",
    "valid": "VisDrone2019-DET-val",
}


def download_and_extract(repo_id: str, data_root: Path) -> None:
    data_root.mkdir(parents=True, exist_ok=True)
    for vd_split in SPLITS.values():
        marker = data_root / vd_split / "images"
        if marker.is_dir() and any(marker.iterdir()):
            print(f"[data] {vd_split} already extracted, skipping")
            continue
        zip_path = hf_hub_download(
            repo_id=repo_id, filename=f"{vd_split}.zip", repo_type="dataset"
        )
        print(f"[data] extracting {zip_path}")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(data_root)
        if not marker.is_dir():
            raise RuntimeError(
                f"unexpected zip layout for {vd_split}: no {marker} after extract"
            )


def build_split(data_root: Path, dataset_dir: Path, rf_split: str, vd_split: str) -> None:
    src = data_root / vd_split
    dst = dataset_dir / rf_split
    dst.mkdir(parents=True, exist_ok=True)

    images = sorted((src / "images").glob("*.jpg"))
    if not images:
        raise RuntimeError(f"no images found in {src / 'images'}")
    for img in images:
        link = dst / img.name
        if not link.exists():
            link.symlink_to(img.resolve())

    convert_split(
        images_dir=src / "images",
        annot_dir=src / "annotations",
        out_json=dst / "_annotations.coco.json",
    )


def prepare(repo_id: str, data_root: str, dataset_dir: str) -> Path:
    data_root_p = Path(data_root)
    dataset_dir_p = Path(dataset_dir)
    download_and_extract(repo_id, data_root_p)
    for rf_split, vd_split in SPLITS.items():
        build_split(data_root_p, dataset_dir_p, rf_split, vd_split)
    # rfdetr's roboflow layout expects a test/ dir to exist; mirror valid.
    test_dir = dataset_dir_p / "test"
    if not test_dir.exists():
        build_split(data_root_p, dataset_dir_p, "test", SPLITS["valid"])
    return dataset_dir_p
