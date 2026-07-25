"""Evaluation + failure taxonomy for the VisDrone small-object study.

Everything prints to stdout — the run log is the only evidence channel.

Two eval modes share every metric section, labeled (STANDARD) / (TILED):
  == COCO METRICS (…) ==      pycocotools summary (maxDets raised to 500)
  == PER-CLASS AP (…) ==      AP@[.5:.95] per category
  == TAXONOMY: * (…) ==       GT recall cross-tabs over size / occlusion /
                              truncation / border distance / class / density
  == THRESHOLD SWEEP (…) ==   recall & precision vs confidence threshold
  == FP BREAKDOWN (…) ==      confident false positives by cause

TILED mode is SAHI-style sliced inference: overlapping square tiles (plus an
optional full-image pass) predicted independently, boxes shifted back to image
coordinates and merged with class-aware NMS. The detector is unchanged — this
isolates pixels-per-object optics from architecture.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import experiment_config as cfg

OCCLUSION_LABELS = {0: "occ=none", 1: "occ=partial", 2: "occ=heavy"}
TRUNCATION_LABELS = {0: "trunc=none", 1: "trunc=partial"}
BORDER_BIN_EDGES = [0, 4, 16, 64, float("inf")]  # px from nearest image border
SWEEP_THRESHOLDS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]


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


def _nms_xywh(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    """Greedy NMS on xywh boxes; returns kept indices."""
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        ious = _iou_matrix(boxes[i : i + 1], boxes[order[1:]])[0]
        order = order[1:][ious < iou_thr]
    return keep


def _label_map(model, coco_gt: COCO) -> dict[int, int]:
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
    return label_to_cat


def _detections_to_results(det, label_to_cat, img_id, dropped, dx=0.0, dy=0.0):
    out = []
    for (x1, y1, x2, y2), score, class_id in zip(det.xyxy, det.confidence, det.class_id):
        cat = label_to_cat.get(int(class_id))
        if cat is None:
            dropped[0] += 1
            continue
        out.append(
            {
                "image_id": img_id,
                "category_id": cat,
                "bbox": [float(x1 + dx), float(y1 + dy), float(x2 - x1), float(y2 - y1)],
                "score": float(score),
            }
        )
    return out


def run_inference(model, valid_dir: Path, coco_gt: COCO, batch_size: int = 16) -> list[dict]:
    """Predict every val image once at model resolution (STANDARD mode)."""
    label_to_cat = _label_map(model, coco_gt)
    print(f"[eval] model class slots: {list(model.class_names)}")
    images = coco_gt.dataset["images"]
    results: list[dict] = []
    dropped = [0]
    for start in range(0, len(images), batch_size):
        chunk = images[start : start + batch_size]
        paths = [str(valid_dir / im["file_name"]) for im in chunk]
        detections = model.predict(paths, threshold=cfg.EVAL_SCORE_THRESHOLD)
        if not isinstance(detections, list):
            detections = [detections]
        for im, det in zip(chunk, detections):
            results.extend(_detections_to_results(det, label_to_cat, im["id"], dropped))
        done = min(start + batch_size, len(images))
        if done % 160 == 0 or done == len(images):
            print(f"[eval] standard inference {done}/{len(images)} images", flush=True)
    if dropped[0]:
        print(f"[eval] dropped {dropped[0]} detections with unmapped class ids")
    print(f"[eval] standard detections kept: {len(results)}")
    return results


def _tile_origins(size: int, tile: int, stride: int) -> list[int]:
    """Left/top origins covering `size` with the final tile flush to the edge."""
    if size <= tile:
        return [0]
    origins = list(range(0, size - tile, stride))
    origins.append(size - tile)
    return origins


def run_tiled_inference(model, valid_dir: Path, coco_gt: COCO) -> list[dict]:
    """SAHI-style sliced inference (TILED mode)."""
    tile = int(cfg.TILE_SIZE)
    stride = max(1, int(tile * (1 - cfg.TILE_OVERLAP)))
    label_to_cat = _label_map(model, coco_gt)
    images = coco_gt.dataset["images"]
    results: list[dict] = []
    dropped = [0]
    n_tiles_total = 0
    for idx, im in enumerate(images, start=1):
        path = valid_dir / im["file_name"]
        with Image.open(path) as pil:
            pil = pil.convert("RGB")
            w, h = pil.size
            tiles, offsets = [], []
            for oy in _tile_origins(h, tile, stride):
                for ox in _tile_origins(w, tile, stride):
                    tiles.append(pil.crop((ox, oy, min(ox + tile, w), min(oy + tile, h))))
                    offsets.append((ox, oy))
            n_tiles_total += len(tiles)
            raw: list[dict] = []
            for start in range(0, len(tiles), 16):
                batch = tiles[start : start + 16]
                dets = model.predict(batch, threshold=cfg.EVAL_SCORE_THRESHOLD)
                if not isinstance(dets, list):
                    dets = [dets]
                for det, (ox, oy) in zip(dets, offsets[start : start + 16]):
                    raw.extend(
                        _detections_to_results(det, label_to_cat, im["id"], dropped, ox, oy)
                    )
            if cfg.TILE_INCLUDE_FULL:
                det = model.predict(str(path), threshold=cfg.EVAL_SCORE_THRESHOLD)
                raw.extend(_detections_to_results(det, label_to_cat, im["id"], dropped))
        # class-aware NMS merge across tiles + full pass
        merged: list[dict] = []
        by_cat: dict[int, list[dict]] = defaultdict(list)
        for r in raw:
            by_cat[r["category_id"]].append(r)
        for rs in by_cat.values():
            boxes = np.array([r["bbox"] for r in rs], dtype=float).reshape(-1, 4)
            scores = np.array([r["score"] for r in rs], dtype=float)
            merged.extend(rs[k] for k in _nms_xywh(boxes, scores, cfg.TILE_MERGE_IOU))
        results.extend(merged)
        if idx % 50 == 0 or idx == len(images):
            print(f"[eval] tiled inference {idx}/{len(images)} images "
                  f"({n_tiles_total} tiles so far)", flush=True)
    if dropped[0]:
        print(f"[eval] dropped {dropped[0]} detections with unmapped class ids")
    print(f"[eval] tiled detections kept: {len(results)} "
          f"(tile={tile}px overlap={cfg.TILE_OVERLAP} full={cfg.TILE_INCLUDE_FULL})")
    return results


def coco_metrics(coco_gt: COCO, results: list[dict], label: str) -> None:
    print(f"\n== COCO METRICS ({label}) ==")
    if not results:
        print("no detections — skipping COCOeval")
        return
    coco_dt = coco_gt.loadRes([dict(r) for r in results])
    ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
    ev.params.maxDets = [1, 10, 500]
    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    print(f"\n== PER-CLASS AP ({label}) ==")
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


def _match_image(gts, dets, iou_thr):
    """Greedy same-class matching; returns (gt_matched bools, det_status list)."""
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
        if best >= 0 and best_iou >= iou_thr and not gt_matched[best]:
            gt_matched[best] = True
            det_status.append("tp")
        elif best >= 0 and best_iou >= iou_thr:
            det_status.append("duplicate")
        elif len(gts) and iou[di].max() >= iou_thr:
            det_status.append("classification")
        elif best >= 0 and best_iou >= 0.1:
            det_status.append("localization")
        else:
            det_status.append("background")
    return gt_matched, det_status


def failure_taxonomy(coco_gt: COCO, results: list[dict], label: str) -> None:
    dets_by_img: dict[int, list[dict]] = defaultdict(list)
    for r in results:
        dets_by_img[r["image_id"]].append(r)

    by_size, by_size_occ, by_size_trunc = Tally(), Tally(), Tally()
    by_border, by_size_border = Tally(), Tally()
    by_class, by_class_small, by_density = Tally(), Tally(), Tally()
    fp_counts: dict[str, int] = defaultdict(int)
    n_gt_matched = 0
    n_gt_total = 0

    for img in coco_gt.dataset["images"]:
        img_id = img["id"]
        gts = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=[img_id]))
        dets = sorted(dets_by_img.get(img_id, []), key=lambda d: -d["score"])
        gt_matched, det_status = _match_image(gts, dets, cfg.MATCH_IOU)

        for det, status in zip(dets, det_status):
            if status != "tp" and det["score"] >= cfg.FP_SCORE_THRESHOLD:
                fp_counts[status] += 1

        density_bin = _bin_label(
            cfg.DENSITY_BIN_EDGES, _bin_index(cfg.DENSITY_BIN_EDGES, len(gts))
        )
        img_w, img_h = img["width"], img["height"]
        for g, matched in zip(gts, gt_matched):
            n_gt_total += 1
            n_gt_matched += bool(matched)
            x, y, bw, bh = g["bbox"]
            size = float(np.sqrt(bw * bh))
            sbin = _bin_label(cfg.SIZE_BIN_EDGES, _bin_index(cfg.SIZE_BIN_EDGES, size))
            border_dist = max(0.0, min(x, y, img_w - (x + bw), img_h - (y + bh)))
            bbin = _bin_label(BORDER_BIN_EDGES, _bin_index(BORDER_BIN_EDGES, border_dist))
            occ = OCCLUSION_LABELS.get(g.get("occlusion", 0), "occ=?")
            trunc = TRUNCATION_LABELS.get(g.get("truncation", 0), "trunc=?")
            cls = coco_gt.cats[g["category_id"]]["name"]
            by_size.add((f"size {sbin}",), matched)
            by_size_occ.add((f"size {sbin}", occ), matched)
            by_size_trunc.add((f"size {sbin}", trunc), matched)
            by_border.add((f"border {bbin}px",), matched)
            if size < 32:
                by_size_border.add((f"border {bbin}px",), matched)
                by_class_small.add((cls,), matched)
            by_class.add((cls,), matched)
            by_density.add((f"gt/img {density_bin}",), matched)

    print(
        f"\n== TAXONOMY: OVERALL ({label}) ==\n  recall@IoU{cfg.MATCH_IOU:g}"
        f"(score>={cfg.EVAL_SCORE_THRESHOLD:g}) = {n_gt_matched / max(n_gt_total, 1):.3f}"
        f"  ({n_gt_matched}/{n_gt_total} GT boxes)"
    )
    by_size.print_table(f"RECALL BY SIZE (sqrt-area px) ({label})")
    by_size_occ.print_table(f"RECALL BY SIZE x OCCLUSION ({label})")
    by_size_trunc.print_table(f"RECALL BY SIZE x TRUNCATION ({label})")
    by_border.print_table(f"RECALL BY BORDER DISTANCE ({label})")
    by_size_border.print_table(f"RECALL BY BORDER DISTANCE (small <32px only) ({label})")
    by_class.print_table(f"RECALL BY CLASS ({label})")
    by_class_small.print_table(f"RECALL BY CLASS (small GT only, <32px) ({label})")
    by_density.print_table(f"RECALL BY IMAGE DENSITY ({label})")

    print(f"\n== FP BREAKDOWN (score>={cfg.FP_SCORE_THRESHOLD:g}) ({label}) ==")
    total_fp = sum(fp_counts.values())
    for cause in ("background", "localization", "classification", "duplicate"):
        n = fp_counts.get(cause, 0)
        share = n / total_fp if total_fp else 0.0
        print(f"  {cause:<16} {n:>7}  ({share:5.1%})")
    print(f"  {'total':<16} {total_fp:>7}")


def threshold_sweep(coco_gt: COCO, results: list[dict], label: str) -> None:
    """Recall (overall + small) and det-level precision vs confidence cut."""
    print(f"\n== THRESHOLD SWEEP ({label}) ==")
    print(f"  {'thr':>5} {'recall':>8} {'recall<32px':>12} {'precision':>10} {'dets':>8}")
    for thr in SWEEP_THRESHOLDS:
        kept = [r for r in results if r["score"] >= thr]
        dets_by_img: dict[int, list[dict]] = defaultdict(list)
        for r in kept:
            dets_by_img[r["image_id"]].append(r)
        gt_tot = gt_hit = small_tot = small_hit = tp = 0
        for img in coco_gt.dataset["images"]:
            gts = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=[img["id"]]))
            dets = sorted(dets_by_img.get(img["id"], []), key=lambda d: -d["score"])
            gt_matched, det_status = _match_image(gts, dets, cfg.MATCH_IOU)
            tp += sum(1 for s in det_status if s == "tp")
            for g, m in zip(gts, gt_matched):
                gt_tot += 1
                gt_hit += bool(m)
                if np.sqrt(g["bbox"][2] * g["bbox"][3]) < 32:
                    small_tot += 1
                    small_hit += bool(m)
        prec = tp / len(kept) if kept else 0.0
        print(f"  {thr:>5.2f} {gt_hit / max(gt_tot, 1):>8.3f} "
              f"{small_hit / max(small_tot, 1):>12.3f} {prec:>10.3f} {len(kept):>8}")


def evaluate(model, dataset_dir: str) -> None:
    valid_dir = Path(dataset_dir) / "valid"
    coco_gt = COCO(str(valid_dir / "_annotations.coco.json"))
    Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    results = run_inference(model, valid_dir, coco_gt)
    with open(Path(cfg.OUTPUT_DIR) / "val_detections.json", "w") as f:
        json.dump(results, f)
    coco_metrics(coco_gt, results, "STANDARD")
    failure_taxonomy(coco_gt, results, "STANDARD")
    threshold_sweep(coco_gt, results, "STANDARD")

    if getattr(cfg, "EVAL_TILED", False):
        tiled = run_tiled_inference(model, valid_dir, coco_gt)
        with open(Path(cfg.OUTPUT_DIR) / "val_detections_tiled.json", "w") as f:
            json.dump(tiled, f)
        coco_metrics(coco_gt, tiled, "TILED")
        failure_taxonomy(coco_gt, tiled, "TILED")
        threshold_sweep(coco_gt, tiled, "TILED")
