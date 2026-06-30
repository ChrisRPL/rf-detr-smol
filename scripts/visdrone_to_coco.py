"""VisDrone annotations -> COCO instances format converter for RF-DETR training.

VisDrone .txt line format:
    <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<category>,<truncation>,<occlusion>

    score:     0 = ignored region (skip), 1 = valid
    category:  0 = ignored, 1..10 = classes (see VISDRONE_CATEGORIES),
               11 = "others" (skip)

Output is standard MS-COCO `instances` JSON:
    { "images": [...], "annotations": [...], "categories": [...] }
which RF-DETR consumes via the supervision/COCO data loader.
"""

from __future__ import annotations

import json
import os
from glob import glob
from pathlib import Path

from PIL import Image

# VisDrone category id -> name. IDs 0 and 11 are ignored / "others".
VISDRONE_CATEGORIES: dict[int, str] = {
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
}

# Remap VisDrone ids to contiguous 1-based COCO ids (deterministic order).
CATEGORY_LIST = [
    {"id": new_id, "name": name}
    for new_id, name in enumerate(VISDRONE_CATEGORIES.values(), start=1)
]
# visdrone_id -> coco_id lookup
ID_REMAP = {
    vd_id: new_id
    for new_id, vd_id in enumerate(VISDRONE_CATEGORIES.keys(), start=1)
}


def convert_split(
    images_dir: str | os.PathLike,
    annot_dir: str | os.PathLike,
    out_json: str | os.PathLike,
    skip_ignored: bool = True,
) -> dict:
    """Convert one VisDrone split to a COCO dict and write it to `out_json`.

    Args:
        images_dir: Directory containing *.jpg images.
        annot_dir: Directory containing matching *.txt VisDrone annotations.
        out_json: Path to write the resulting COCO JSON.
        skip_ignored: Drop rows with score=0, category=0, or category=11.

    Returns:
        The COCO dict that was written.
    """
    images_dir = Path(images_dir)
    annot_dir = Path(annot_dir)
    out_json = Path(out_json)

    image_paths = sorted(images_dir.glob("*.jpg"))
    coco: dict = {"images": [], "annotations": [], "categories": CATEGORY_LIST}

    ann_id = 1
    for img_id, img_path in enumerate(image_paths, start=1):
        with Image.open(img_path) as im:
            w, h = im.size
        coco["images"].append(
            {
                "id": img_id,
                "file_name": img_path.name,
                "width": w,
                "height": h,
            }
        )

        txt_path = annot_dir / f"{img_path.stem}.txt"
        if not txt_path.exists():
            continue

        with open(txt_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                x, y, bw, bh = map(float, parts[:4])
                score = int(parts[4])
                cat = int(parts[5])

                if skip_ignored and (score == 0 or cat in (0, 11)):
                    continue
                if cat not in ID_REMAP:
                    continue
                if bw <= 0 or bh <= 0:
                    continue

                # Clamp negative top-left into the image frame.
                x = max(0.0, x)
                y = max(0.0, y)

                coco["annotations"].append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": ID_REMAP[cat],
                        "bbox": [x, y, bw, bh],  # COCO style: [x_min, y_min, w, h]
                        "area": bw * bh,
                        "iscrowd": 0,
                    }
                )
                ann_id += 1

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(coco, f)
    print(
        f"[{out_json.name}] images={len(coco['images'])} "
        f"annotations={len(coco['annotations'])}"
    )
    return coco


def convert_all(data_root: str | os.PathLike = "data") -> None:
    """Convert train / val / test-dev splits under `data_root`."""
    data_root = Path(data_root)
    splits = [
        ("VisDrone2019-DET-train", "train_coco.json"),
        ("VisDrone2019-DET-val", "val_coco.json"),
        ("VisDrone2019-DET-test-dev", "test_coco.json"),
    ]
    for split_name, out_name in splits:
        split_dir = data_root / split_name
        if not split_dir.exists():
            print(f"skip {split_name}: missing")
            continue
        convert_split(
            images_dir=split_dir / "images",
            annot_dir=split_dir / "annotations",
            out_json=split_dir / out_name,
        )


if __name__ == "__main__":
    convert_all()