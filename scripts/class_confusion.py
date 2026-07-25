"""Round 6: quantify the class-confusion residual on the best model.

Two analyses over the standard val detections (COCO-format `results`):

  == CONFUSION MATRIX ==   for every det matched to a GT box at IoU>=0.5
                           (regardless of class agreement), tally predicted vs
                           true class. Off-diagonal mass = inter-class swaps.
  == CLASS-MERGE UPPER BOUND ==  remap confusable groups into super-classes and
                           recompute recall/precision. The gain is an upper
                           bound on what perfect disambiguation could recover.

Everything is post-hoc on saved detections — no GPU needed beyond the eval that
produced them. Figures + JSON go to output/confusion/ for the paper.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO

import experiment_config as cfg

# VisDrone super-classes for the merge probe (by category NAME, order-independent).
MERGE_GROUPS = {
    "person": ["pedestrian", "people"],
    "tricycle-like": ["tricycle", "awning-tricycle"],
    "large-vehicle": ["truck", "bus"],
    # car/van kept separate — they are a real, frequent confusion but also a
    # meaningful distinction; reported in the matrix, not merged, to avoid
    # over-claiming the upper bound.
}


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    ix1 = np.maximum(a[:, 0][:, None], b[:, 0][None, :])
    iy1 = np.maximum(a[:, 1][:, None], b[:, 1][None, :])
    ix2 = np.minimum(ax2[:, None], bx2[None, :])
    iy2 = np.minimum(ay2[:, None], by2[None, :])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    ua = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None, :] - inter
    return np.where(ua > 0, inter / ua, 0.0)


def _confusion_matrix(coco_gt: COCO, results: list[dict], cat_ids, names):
    """Predicted vs true class over IoU>=0.5 localization-correct detections."""
    idx = {c: i for i, c in enumerate(cat_ids)}
    n = len(cat_ids)
    mat = np.zeros((n, n), dtype=np.int64)  # rows = true, cols = predicted
    dets_by_img = defaultdict(list)
    for r in results:
        if r["score"] >= cfg.FP_SCORE_THRESHOLD:
            dets_by_img[r["image_id"]].append(r)
    for img in coco_gt.dataset["images"]:
        gts = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=[img["id"]]))
        dets = sorted(dets_by_img.get(img["id"], []), key=lambda d: -d["score"])
        if not gts or not dets:
            continue
        gb = np.array([g["bbox"] for g in gts], float).reshape(-1, 4)
        db = np.array([d["bbox"] for d in dets], float).reshape(-1, 4)
        iou = _iou(db, gb)
        used = set()
        for di, det in enumerate(dets):
            j = int(iou[di].argmax())
            if iou[di, j] >= cfg.MATCH_IOU and j not in used:
                used.add(j)
                mat[idx[gts[j]["category_id"]], idx[det["category_id"]]] += 1
    return mat


def _merge_metrics(coco_gt: COCO, results: list[dict], names_by_id):
    """Recall/precision before vs after merging confusable classes."""
    name_to_super = {}
    for sup, members in MERGE_GROUPS.items():
        for m in members:
            name_to_super[m] = sup

    def remap(cat_id):
        nm = names_by_id[cat_id]
        return name_to_super.get(nm, nm)

    def score(use_merge):
        dets_by_img = defaultdict(list)
        for r in results:
            dets_by_img[r["image_id"]].append(r)
        gt_tot = gt_hit = tp = det_tot = 0
        for img in coco_gt.dataset["images"]:
            gts = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=[img["id"]]))
            dets = sorted(dets_by_img.get(img["id"], []), key=lambda d: -d["score"])
            dets = [d for d in dets if d["score"] >= cfg.FP_SCORE_THRESHOLD]
            gb = np.array([g["bbox"] for g in gts], float).reshape(-1, 4)
            db = np.array([d["bbox"] for d in dets], float).reshape(-1, 4)
            iou = _iou(db, gb) if len(gts) and len(dets) else np.zeros((len(dets), len(gts)))
            gcls = [remap(g["category_id"]) if use_merge else g["category_id"] for g in gts]
            dcls = [remap(d["category_id"]) if use_merge else d["category_id"] for d in dets]
            matched = set()
            det_tot += len(dets)
            for di in range(len(dets)):
                same = np.array([gcls[gj] == dcls[di] for gj in range(len(gts))], bool)
                cand = iou[di] * same
                j = int(cand.argmax()) if len(gts) else -1
                if j >= 0 and cand[j] >= cfg.MATCH_IOU and j not in matched:
                    matched.add(j)
                    tp += 1
            gt_tot += len(gts)
            gt_hit += len(matched)
        return gt_hit / max(gt_tot, 1), tp / max(det_tot, 1)

    base_r, base_p = score(False)
    merged_r, merged_p = score(True)
    return {"baseline_recall": base_r, "baseline_precision": base_p,
            "merged_recall": merged_r, "merged_precision": merged_p}


def _plot_confusion(mat, names, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    row_sum = mat.sum(1, keepdims=True)
    norm = mat / np.clip(row_sum, 1, None)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(norm, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    ax.set_xlabel("predicted class"); ax.set_ylabel("true class (GT)")
    ax.set_title("Class confusion on localization-correct detections (row-normalized)")
    for i in range(len(names)):
        for j in range(len(names)):
            if norm[i, j] >= 0.03:
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                        color="white" if norm[i, j] < 0.6 else "black", fontsize=7)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)


def analyze(dataset_dir: str) -> None:
    valid_dir = Path(dataset_dir) / "valid"
    coco_gt = COCO(str(valid_dir / "_annotations.coco.json"))
    results = json.load(open(Path(cfg.OUTPUT_DIR) / "val_detections.json"))
    cats = sorted(coco_gt.dataset["categories"], key=lambda c: c["id"])
    cat_ids = [c["id"] for c in cats]
    names = [c["name"] for c in cats]
    names_by_id = {c["id"]: c["name"] for c in cats}

    mat = _confusion_matrix(coco_gt, results, cat_ids, names)
    print("\n== CONFUSION MATRIX (true rows -> predicted cols, IoU>=0.5) ==")
    print("  " + "".join(f"{n[:6]:>8}" for n in names))
    for i, n in enumerate(names):
        print(f"  {n:<16}" + "".join(f"{mat[i, j]:>8}" for j in range(len(names))))
    diag = int(np.trace(mat)); tot = int(mat.sum())
    print(f"  correct-class share of localized dets: {diag / max(tot, 1):.3f} "
          f"({diag}/{tot})")
    # top off-diagonal confusions
    off = [(names[i], names[j], int(mat[i, j])) for i in range(len(names))
           for j in range(len(names)) if i != j]
    off.sort(key=lambda t: -t[2])
    print("  top confusions (true -> predicted):")
    for a, b, c in off[:6]:
        print(f"    {a:<16} -> {b:<16} {c}")

    merge = _merge_metrics(coco_gt, results, names_by_id)
    print("\n== CLASS-MERGE UPPER BOUND ==")
    print(f"  merge groups: {MERGE_GROUPS}")
    print(f"  recall    : baseline {merge['baseline_recall']:.3f} -> "
          f"merged {merge['merged_recall']:.3f} "
          f"(+{merge['merged_recall'] - merge['baseline_recall']:.3f})")
    print(f"  precision : baseline {merge['baseline_precision']:.3f} -> "
          f"merged {merge['merged_precision']:.3f} "
          f"(+{merge['merged_precision'] - merge['baseline_precision']:.3f})")

    out = Path(cfg.OUTPUT_DIR) / "confusion"
    out.mkdir(parents=True, exist_ok=True)
    _plot_confusion(mat, names, out / "confusion_matrix.png")
    with open(out / "confusion_data.json", "w") as f:
        json.dump({"matrix": mat.tolist(), "names": names, "merge": merge,
                   "top_confusions": off[:10]}, f, indent=1)
    print(f"[confusion] artifacts -> {out.resolve()}")
