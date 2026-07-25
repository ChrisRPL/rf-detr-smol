"""Cross-architecture runner (Ultralytics: YOLO11 / RT-DETR) on VisDrone.

Trains an ultralytics detector on VisDrone, then evaluates it through the SAME
taxonomy harness as RF-DETR (scripts/evaluate.py) against the same val COCO GT.
This is what makes the cross-architecture comparison fair: identical ground
truth, identical size/occlusion/border/density cross-tabs, identical matcher.

Class contract: YOLO class index i corresponds to COCO category_id i+1 (our
converter numbers VisDrone classes 1..10 in a fixed order; data.yaml lists them
in that same order, so the mapping is a +1 shift).
"""

from __future__ import annotations

import json
from pathlib import Path

from pycocotools.coco import COCO

import experiment_config as cfg
from scripts.evaluate import coco_metrics, failure_taxonomy, threshold_sweep


def _coco_to_yolo(coco_json: Path, images_dir: Path, out_labels: Path) -> None:
    """Write one YOLO .txt per image (class cx cy w h, normalized)."""
    coco = COCO(str(coco_json))
    out_labels.mkdir(parents=True, exist_ok=True)
    for img in coco.dataset["images"]:
        w, h = img["width"], img["height"]
        lines = []
        for a in coco.loadAnns(coco.getAnnIds(imgIds=[img["id"]])):
            x, y, bw, bh = a["bbox"]
            cls = a["category_id"] - 1  # COCO id (1-based) -> YOLO index (0-based)
            cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw / w:.6f} {bh / h:.6f}")
        (out_labels / f"{Path(img['file_name']).stem}.txt").write_text("\n".join(lines))


def build_yolo_dataset(dataset_dir: Path, yolo_dir: Path) -> Path:
    """Build a YOLO-format view (images symlinked, labels converted) + data.yaml."""
    coco_train = COCO(str(dataset_dir / "train" / "_annotations.coco.json"))
    names = [c["name"] for c in sorted(coco_train.dataset["categories"],
                                       key=lambda c: c["id"])]
    for split_src, split_dst in [("train", "train"), ("valid", "val")]:
        img_out = yolo_dir / "images" / split_dst
        img_out.mkdir(parents=True, exist_ok=True)
        src_imgs = dataset_dir / split_src
        for jpg in src_imgs.glob("*.jpg"):
            link = img_out / jpg.name
            if not link.exists():
                link.symlink_to(jpg.resolve())
        _coco_to_yolo(src_imgs / "_annotations.coco.json", src_imgs,
                      yolo_dir / "labels" / split_dst)
    data_yaml = yolo_dir / "visdrone.yaml"
    data_yaml.write_text(
        f"path: {yolo_dir.resolve()}\ntrain: images/train\nval: images/val\n"
        f"nc: {len(names)}\nnames: {names}\n"
    )
    print(f"[ultra] YOLO dataset ready: {data_yaml} ({len(names)} classes)")
    return data_yaml


def predict_to_coco(model, dataset_dir: Path) -> list[dict]:
    """Run the trained model over val images -> COCO-format detection dicts
    keyed to OUR val image ids (so the shared taxonomy lines up)."""
    valid_dir = dataset_dir / "valid"
    coco_gt = COCO(str(valid_dir / "_annotations.coco.json"))
    name_to_id = {Path(im["file_name"]).name: im["id"]
                  for im in coco_gt.dataset["images"]}
    results: list[dict] = []
    images = sorted(valid_dir.glob("*.jpg"))
    for start in range(0, len(images), 32):
        batch = [str(p) for p in images[start:start + 32]]
        preds = model.predict(batch, imgsz=cfg.RESOLUTION,
                              conf=cfg.EVAL_SCORE_THRESHOLD, verbose=False,
                              device=0)  # CUDA_VISIBLE_DEVICES already masks to one card
        for path, r in zip(batch, preds):
            img_id = name_to_id.get(Path(path).name)
            if img_id is None:
                continue
            b = r.boxes
            if b is None:
                continue
            xyxy = b.xyxy.cpu().numpy()
            conf = b.conf.cpu().numpy()
            cls = b.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), sc, c in zip(xyxy, conf, cls):
                results.append({
                    "image_id": img_id, "category_id": int(c) + 1,
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(sc),
                })
        done = min(start + 32, len(images))
        if done % 160 == 0 or done == len(images):
            print(f"[ultra] inference {done}/{len(images)} images", flush=True)
    print(f"[ultra] detections kept: {len(results)}")
    return results


def main(dataset_dir: Path) -> None:
    from ultralytics import YOLO, RTDETR

    data_yaml = build_yolo_dataset(dataset_dir, Path(cfg.YOLO_DIR))

    ctor = RTDETR if "rtdetr" in cfg.ULTRA_WEIGHTS.lower() else YOLO
    print(f"[ultra] {ctor.__name__}({cfg.ULTRA_WEIGHTS}) imgsz={cfg.RESOLUTION}")
    model = ctor(cfg.ULTRA_WEIGHTS)
    model.train(
        data=str(data_yaml), epochs=cfg.EPOCHS, imgsz=cfg.RESOLUTION,
        batch=cfg.BATCH_SIZE, workers=cfg.NUM_WORKERS, device=0,
        project=cfg.OUTPUT_DIR, name="train", exist_ok=True, verbose=True,
    )

    print("\n== EVALUATE (shared taxonomy) ==", flush=True)
    coco_gt = COCO(str(dataset_dir / "valid" / "_annotations.coco.json"))
    results = predict_to_coco(model, dataset_dir)
    Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    with open(Path(cfg.OUTPUT_DIR) / "val_detections.json", "w") as f:
        json.dump(results, f)
    coco_metrics(coco_gt, results, "STANDARD")
    failure_taxonomy(coco_gt, results, "STANDARD")
    threshold_sweep(coco_gt, results, "STANDARD")
