"""Evaluation + failure taxonomy for the VisDrone small-object study.

Everything prints to stdout — the run log is the only evidence channel.

Sections emitted:
  == COCO METRICS ==        pycocotools summary (maxDets raised to 500)
  == PER-CLASS AP ==        AP@[.5:.95] per category
  == TAXONOMY: * ==         GT recall cross-tabs over size / occlusion /
                            truncation / class / image density
  == FP BREAKDOWN ==        confident false positives by cause
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import experiment_config as cfg

OCCLUSION_LABELS = {0: "occ=none", 1: "occ=partial", 2: "occ=heavy"}
TRUNCATION_LABELS = {0: "trunc=none", 1: "trunc=partial"}


def _bin_label(edges: list[float], i: int) -> str:
    lo, hi = edges[i], edges[i + 1]
    return f"{lo:g}-{hi:g}" if hi != float("inf") else f">{lo:g}"


def _bin_index(edges: list[float], value: float) -> int:
    for i in range(len(edges) - 1):
        if edges[i] <= value < edges[i + 1]:
            return i
    return len(edges) - 2


def _iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """IoU between two sets of xywh boxes."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)))
    ax1, ay1 = boxes_a[:, 0], boxes_a[:, 1]
    ax2, ay2 = ax1 + boxes_a[:, 2], ay1 + boxes_a[:, 3]
    bx1, by1 = boxes_b[:, 0], boxes_b[:, 1]
    bx2, by2 = bx1 + boxes_b[:, 2], by1 + boxes_b[:, 3]
    ix1 = np.maximum(ax1[:, None], bx1[None, :])
    iy1 = np.maximum(ay1[:, None], by1[None, :])
    ix2 = np.minimum(ax2[:, None], bx2[None, :])
    iy2 = np.minimum(ay2[:, None], by2[None, :])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = boxes_a[:, 2] * boxes_a[:, 3]
    area_b = boxes_b[:, 2] * boxes_b[:, 3]
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def run_inference(model, valid_dir: Path, coco_gt: COCO, batch_size: int = 16) -> list[dict]:
    """Predict every val image; return COCO-format detection dicts."""
    name_to_cat = {c["name"]: c["id"] for c in coco_gt.dataset["categories"]}
    class_names = list(model.class_names)
    label_to_cat = {}
    unmapped = set()
    for label, name in enumerate(class_names):
        if name in name_to_cat:
            label_to_cat[label] = name_to_cat[name]
        else:
            unmapped.add(name)
    if unmapped:
        print(f"[eval] WARNING: model classes not in GT (dropped): {sorted(unmapped)}")
    print(f"[eval] model class slots: {class_names}")

    images = coco_gt.dataset["images"]
    results: list[dict] = []
    dropped = 0
    for start in range(0, len(images), batch_size):
        chunk = images[start : start + batch_size]
        paths = [str(valid_dir / im["file_name"]) for im in chunk]
        detections = model.predict(paths, threshold=cfg.EVAL_SCORE_THRESHOLD)
        if not isinstance(detections, list):
            detections = [detections]
        for im, det in zip(chunk, detections):
            for (x1, y1, x2, y2), score, class_id in zip(
                det.xyxy, det.confidence, det.class_id
            ):
                cat = label_to_cat.get(int(class_id))
                if cat is None:
                    dropped += 1
                    continue
                results.append(
                    {
                        "image_id": im["id"],
                        "category_id": cat,
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": float(score),
                    }
                )
        done = min(start + batch_size, len(images))
        if done % 160 == 0 or done == len(images):
            print(f"[eval] inference {done}/{len(images)} images", flush=True)
    if dropped:
        print(f"[eval] dropped {dropped} detections with unmapped class ids")
    print(f"[eval] total detections kept: {len(results)}")
    return results


def coco_metrics(coco_gt: COCO, results: list[dict]) -> None:
    print("\n== COCO METRICS ==")
    if not results:
        print("no detections — skipping COCOeval")
        return
    coco_dt = coco_gt.loadRes([dict(r) for r in results])
    ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
    ev.params.maxDets = [1, 10, 500]
    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    print("\n== PER-CLASS AP ==")
    # precision: [iou_thrs, recall, cat, area, maxdets]; area 0 = all, last maxDets
    precision = ev.eval["precision"]
    for k, cat_id in enumerate(ev.params.catIds):
        p = precision[:, :, k, 0, -1]
        p = p[p > -1]
        ap = p.mean() if p.size else float("nan")
        name = coco_gt.cats[cat_id]["name"]
        print(f"  {name:<16} AP={ap:.4f}")


class Tally:
    def __init__(self) -> None:
        self.hit: dict = defaultdict(int)
        self.total: dict = defaultdict(int)

    def add(self, key, matched: bool) -> None:
        self.total[key] += 1
        if matched:
            self.hit[key] += 1

    def print_table(self, title: str) -> None:
        print(f"\n== TAXONOMY: {title} ==")
        keys = sorted(self.total.keys(), key=str)
        for key in keys:
            n, h = self.total[key], self.hit[key]
            label = " | ".join(key) if isinstance(key, tuple) else str(key)
            print(f"  {label:<34} recall={h / n:6.3f}  matched={h:>6}/{n:<6}")


def failure_taxonomy(coco_gt: COCO, results: list[dict]) -> None:
    dets_by_img: dict[int, list[dict]] = defaultdict(list)
    for r in results:
        dets_by_img[r["image_id"]].append(r)

    by_size, by_size_occ, by_size_trunc = Tally(), Tally(), Tally()
    by_class, by_class_small, by_density = Tally(), Tally(), Tally()
    fp_counts: dict[str, int] = defaultdict(int)
    n_gt_matched = 0
    n_gt_total = 0

    for img in coco_gt.dataset["images"]:
        img_id = img["id"]
        gts = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=[img_id]))
        dets = sorted(dets_by_img.get(img_id, []), key=lambda d: -d["score"])

        gt_boxes = np.array([g["bbox"] for g in gts], dtype=float).reshape(-1, 4)
        det_boxes = np.array([d["bbox"] for d in dets], dtype=float).reshape(-1, 4)
        iou = _iou_matrix(det_boxes, gt_boxes)

        gt_matched = np.zeros(len(gts), dtype=bool)
        det_status: list[str] = []
        for di, det in enumerate(dets):
            same_class = np.array(
                [g["category_id"] == det["category_id"] for g in gts], dtype=bool
            )
            candidates = iou[di] * same_class
            best = int(candidates.argmax()) if len(gts) else -1
            best_iou = candidates[best] if best >= 0 else 0.0
            if best >= 0 and best_iou >= cfg.MATCH_IOU and not gt_matched[best]:
                gt_matched[best] = True
                det_status.append("tp")
                continue
            # false positive — classify the cause
            if best >= 0 and best_iou >= cfg.MATCH_IOU and gt_matched[best]:
                det_status.append("duplicate")
            elif len(gts) and iou[di].max() >= cfg.MATCH_IOU:
                det_status.append("classification")
            elif best >= 0 and best_iou >= 0.1:
                det_status.append("localization")
            else:
                det_status.append("background")

        for det, status in zip(dets, det_status):
            if status != "tp" and det["score"] >= cfg.FP_SCORE_THRESHOLD:
                fp_counts[status] += 1

        density_bin = _bin_label(
            cfg.DENSITY_BIN_EDGES, _bin_index(cfg.DENSITY_BIN_EDGES, len(gts))
        )
        for g, matched in zip(gts, gt_matched):
            n_gt_total += 1
            n_gt_matched += bool(matched)
            size = float(np.sqrt(g["bbox"][2] * g["bbox"][3]))
            sbin = _bin_label(cfg.SIZE_BIN_EDGES, _bin_index(cfg.SIZE_BIN_EDGES, size))
            occ = OCCLUSION_LABELS.get(g.get("occlusion", 0), "occ=?")
            trunc = TRUNCATION_LABELS.get(g.get("truncation", 0), "trunc=?")
            cls = coco_gt.cats[g["category_id"]]["name"]
            by_size.add((f"size {sbin}",), matched)
            by_size_occ.add((f"size {sbin}", occ), matched)
            by_size_trunc.add((f"size {sbin}", trunc), matched)
            by_class.add((cls,), matched)
            if size < 32:
                by_class_small.add((cls,), matched)
            by_density.add((f"gt/img {density_bin}",), matched)

    print(
        f"\n== TAXONOMY: OVERALL == \n  recall@IoU{cfg.MATCH_IOU:g}"
        f"(score>={cfg.EVAL_SCORE_THRESHOLD:g}) = {n_gt_matched / max(n_gt_total, 1):.3f}"
        f"  ({n_gt_matched}/{n_gt_total} GT boxes)"
    )
    by_size.print_table("RECALL BY SIZE (sqrt-area px)")
    by_size_occ.print_table("RECALL BY SIZE x OCCLUSION")
    by_size_trunc.print_table("RECALL BY SIZE x TRUNCATION")
    by_class.print_table("RECALL BY CLASS")
    by_class_small.print_table("RECALL BY CLASS (small GT only, <32px)")
    by_density.print_table("RECALL BY IMAGE DENSITY")

    print(f"\n== FP BREAKDOWN (score>={cfg.FP_SCORE_THRESHOLD:g}) ==")
    total_fp = sum(fp_counts.values())
    for cause in ("background", "localization", "classification", "duplicate"):
        n = fp_counts.get(cause, 0)
        share = n / total_fp if total_fp else 0.0
        print(f"  {cause:<16} {n:>7}  ({share:5.1%})")
    print(f"  {'total':<16} {total_fp:>7}")


def evaluate(model, dataset_dir: str) -> None:
    valid_dir = Path(dataset_dir) / "valid"
    coco_gt = COCO(str(valid_dir / "_annotations.coco.json"))
    results = run_inference(model, valid_dir, coco_gt)
    Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    with open(Path(cfg.OUTPUT_DIR) / "val_detections.json", "w") as f:
        json.dump(results, f)
    coco_metrics(coco_gt, results)
    failure_taxonomy(coco_gt, results)
