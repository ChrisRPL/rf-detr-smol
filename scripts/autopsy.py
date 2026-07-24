"""RF-DETR autopsy: trace where small-object information dies inside the model.

Replicates LWDETR.forward stage by stage (backbone -> projected pyramid ->
two-stage proposal selection -> decoder layers -> heads) on val images and
tallies, per GT size bin:

  == AUTOPSY: STAGE MAP ==          tensor shapes / strides / token counts
  == AUTOPSY: TOKEN COVERAGE ==     feature cells per GT box at each scale
  == AUTOPSY: FEATURE ENERGY ==     activation-norm SNR inside GT footprint
  == AUTOPSY: PROPOSAL COVERAGE ==  two-stage encoder proposals hitting GT
  == AUTOPSY: DECODER EVOLUTION ==  best-query IoU/score per decoder layer
  == AUTOPSY: SURVIVAL FUNNEL ==    the thesis table - stage-by-stage survival

All geometry runs in normalized [0,1] coordinates: rfdetr's square resize maps
normalized original-image coords 1:1 to model space, so no rescaling
bookkeeping is needed. Size bins use ORIGINAL-pixel sqrt-area (same as the
taxonomy). Every section is fault-isolated: a shape surprise prints its
traceback and the run continues with the remaining sections.
"""

from __future__ import annotations

import json
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw
from pycocotools.coco import COCO

import experiment_config as cfg

BIN_COLORS = {  # size bin -> RGB, used consistently across all figures
    "0-8": (228, 26, 28), "8-16": (255, 127, 0), "16-32": (255, 215, 0),
    "32-96": (77, 175, 74), ">96": (55, 126, 184),
}

MEANS = [0.485, 0.456, 0.406]
STDS = [0.229, 0.224, 0.225]


def _size_bin(size_px: float) -> str:
    edges = cfg.SIZE_BIN_EDGES
    for i in range(len(edges) - 1):
        if edges[i] <= size_px < edges[i + 1]:
            lo, hi = edges[i], edges[i + 1]
            return f"{lo:g}-{hi:g}" if hi != float("inf") else f">{lo:g}"
    return "?"


class MeanTally:
    def __init__(self) -> None:
        self.sum: dict = defaultdict(float)
        self.n: dict = defaultdict(int)

    def add(self, key, value: float) -> None:
        self.sum[key] += float(value)
        self.n[key] += 1

    def rows(self):
        for key in sorted(self.n, key=str):
            yield key, self.sum[key] / self.n[key], self.n[key]


def _print_mean_table(title: str, tally: MeanTally, fmt: str = "{:.3f}") -> None:
    print(f"\n== AUTOPSY: {title} ==")
    for key, mean, n in tally.rows():
        label = " | ".join(key) if isinstance(key, tuple) else str(key)
        print(f"  {label:<40} {fmt.format(mean):>9}  (n={n})")


def _viz_dir() -> Path:
    d = Path(cfg.OUTPUT_DIR) / "autopsy"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _draw_boxes(img: Image.Image, boxes_xyxy_norm, colors, width: int = 2) -> Image.Image:
    out = img.copy()
    dr = ImageDraw.Draw(out)
    w, h = out.size
    for (x1, y1, x2, y2), color in zip(boxes_xyxy_norm, colors):
        dr.rectangle([x1 * w, y1 * h, x2 * w, y2 * h], outline=color, width=width)
    return out


def _save_image_viz(stem, pil, res, gt_xyxy, bins, srcs, prop_centers, final_boxes,
                    final_scores) -> None:
    """Per-image qualitative artifacts (paper figures): GT overlay, per-scale
    feature-energy heatmaps, proposal scatter, final decoder boxes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = _viz_dir()
    base = pil.resize((res, res))
    colors = [BIN_COLORS.get(b, (200, 200, 200)) for b in bins]
    _draw_boxes(base, gt_xyxy, colors).save(out / f"{stem}_1gt.png")

    for s_i, src in enumerate(srcs):
        norm_map = src[0].norm(dim=0).cpu().numpy()
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(base)
        ax.imshow(norm_map, cmap="inferno", alpha=0.55,
                  extent=(0, res, res, 0), interpolation="bilinear")
        for (x1, y1, x2, y2), b in zip(gt_xyxy, bins):
            if b in ("0-8", "8-16"):
                ax.add_patch(plt.Rectangle((x1 * res, y1 * res), (x2 - x1) * res,
                                           (y2 - y1) * res, fill=False,
                                           edgecolor="cyan", linewidth=0.8))
        ax.set_axis_off()
        ax.set_title(f"feature-energy heatmap, scale{s_i} (small GT in cyan)")
        fig.savefig(out / f"{stem}_2energy_scale{s_i}.png", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)

    if prop_centers is not None:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(base)
        ax.scatter(prop_centers[:, 0] * res, prop_centers[:, 1] * res,
                   s=4, c="lime", alpha=0.6, label="two-stage proposal centers")
        for (x1, y1, x2, y2), b in zip(gt_xyxy, bins):
            if b in ("0-8", "8-16"):
                ax.add_patch(plt.Rectangle((x1 * res, y1 * res), (x2 - x1) * res,
                                           (y2 - y1) * res, fill=False,
                                           edgecolor="red", linewidth=0.8))
        ax.set_axis_off()
        ax.legend(loc="lower right", fontsize=7)
        ax.set_title("proposal allocation vs small GT (red)")
        fig.savefig(out / f"{stem}_3proposals.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    if final_boxes is not None:
        keep = final_scores >= 0.3
        _draw_boxes(base, final_boxes[keep],
                    [(0, 255, 0)] * int(keep.sum())).save(out / f"{stem}_4final.png")


@torch.no_grad()
def run_autopsy(rf_model, dataset_dir: str) -> None:
    ctx = rf_model.model            # ModelContext
    net = ctx.model                 # LWDETR nn.Module
    device = ctx.device
    res = ctx.resolution
    net.eval()
    print(f"[autopsy] model resolution={res} num_queries={net.num_queries} "
          f"group_detr={net.group_detr} two_stage={net.two_stage} device={device}")

    valid_dir = Path(dataset_dir) / "valid"
    coco_gt = COCO(str(valid_dir / "_annotations.coco.json"))
    images = coco_gt.dataset["images"]
    max_images = int(getattr(cfg, "AUTOPSY_MAX_IMAGES", 0) or len(images))
    images = images[:max_images]
    print(f"[autopsy] tracing {len(images)} val images, "
          f"{sum(len(coco_gt.getAnnIds(imgIds=[im['id']])) for im in images)} GT boxes")

    # visualize the densest images — most illustrative for the paper
    n_viz = int(getattr(cfg, "AUTOPSY_VIZ_IMAGES", 6))
    viz_ids = {
        im["id"]
        for im in sorted(images, key=lambda i: -len(coco_gt.getAnnIds(imgIds=[i["id"]])))[:n_viz]
    }
    stage_info: list[dict] = []
    stage_map_printed = False
    coverage = MeanTally()      # (bin, scale) -> overlapped cells
    zero_center = MeanTally()   # (bin, scale) -> 1.0 if NO cell center inside box
    energy = MeanTally()        # (bin, scale) -> in-box norm / global norm
    prop_hit = MeanTally()      # (bin,) -> 1.0 if any proposal center in box
    prop_count = MeanTally()    # (bin,) -> proposals inside box
    dec_iou = MeanTally()       # (bin, layer) -> best IoU
    dec_found = MeanTally()     # (bin, layer) -> best IoU >= 0.5
    dec_score = MeanTally()     # (bin, layer) -> score of best-IoU query
    funnel = MeanTally()        # (bin, stage) -> survival fraction contributors

    for idx, im in enumerate(images, start=1):
        anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=[im["id"]]))
        if not anns:
            continue
        w0, h0 = im["width"], im["height"]
        # GT normalized xyxy + size bins (original px)
        gt_xyxy = np.array(
            [[a["bbox"][0] / w0, a["bbox"][1] / h0,
              (a["bbox"][0] + a["bbox"][2]) / w0,
              (a["bbox"][1] + a["bbox"][3]) / h0] for a in anns], dtype=np.float64
        ).clip(0, 1)
        bins = [_size_bin(float(np.sqrt(a["bbox"][2] * a["bbox"][3]))) for a in anns]

        pil = Image.open(valid_dir / im["file_name"]).convert("RGB")
        t = TF.to_tensor(TF.resize(pil, [res, res]))
        t = TF.normalize(t, MEANS, STDS).unsqueeze(0).to(device)

        from rfdetr.utilities.tensors import nested_tensor_from_tensor_list

        samples = nested_tensor_from_tensor_list(t)
        features, poss, cross = net.backbone(samples)
        srcs, masks = [], []
        for feat in features:
            src, mask = feat.decompose()
            srcs.append(src)
            masks.append(mask)

        if not stage_map_printed:
            print("\n== AUTOPSY: STAGE MAP ==")
            print(f"  input: {tuple(t.shape)} ({res}x{res} px)")
            for s_i, src in enumerate(srcs):
                _, c, hs_, ws_ = src.shape
                print(f"  scale{s_i}: shape={tuple(src.shape)} stride={res // hs_} "
                      f"tokens={hs_ * ws_} channels={c}")
                stage_info.append({"scale": s_i, "shape": list(src.shape),
                                   "stride": res // hs_, "tokens": hs_ * ws_})
            stage_map_printed = True

        # --- token coverage + feature energy per scale ---
        try:
            for s_i, src in enumerate(srcs):
                _, c, hs_, ws_ = src.shape
                norm_map = src[0].norm(dim=0).cpu().numpy()  # [hs, ws]
                global_mean = float(norm_map.mean()) or 1.0
                for g, (x1, y1, x2, y2) in enumerate(gt_xyxy):
                    c0, c1 = int(x1 * ws_), min(int(np.ceil(x2 * ws_)), ws_)
                    r0, r1 = int(y1 * hs_), min(int(np.ceil(y2 * hs_)), hs_)
                    n_cells = max(0, c1 - c0) * max(0, r1 - r0)
                    coverage.add((bins[g], f"scale{s_i}"), n_cells)
                    centers_x = (np.arange(ws_) + 0.5) / ws_
                    centers_y = (np.arange(hs_) + 0.5) / hs_
                    inside = ((centers_x >= x1) & (centers_x < x2)).sum() * \
                             ((centers_y >= y1) & (centers_y < y2)).sum()
                    zero_center.add((bins[g], f"scale{s_i}"), 1.0 if inside == 0 else 0.0)
                    if n_cells > 0:
                        box_norm = norm_map[r0:r1, c0:c1].mean()
                        energy.add((bins[g], f"scale{s_i}"), box_norm / global_mean)
        except Exception:
            print("[autopsy] WARNING: coverage/energy section failed:")
            traceback.print_exc()

        # --- transformer: proposals + decoder evolution ---
        pcn = None
        final_q = None
        final_s = None
        try:
            ref_w = net.refpoint_embed.weight[: net.num_queries]
            qf_w = net.query_feat.weight[: net.num_queries]
            cross_srcs = None
            if cross is not None:
                cross_srcs = [f.decompose()[0] for f in cross]
            hs, ref_unsig, hs_enc, ref_enc = net.transformer(
                srcs, masks, poss, ref_w, qf_w, cross_attn_srcs=cross_srcs
            )[:4]

            # two-stage proposal centers (apply sigmoid iff values look unbounded)
            if ref_enc is not None:
                pc = ref_enc[-1] if ref_enc.dim() == 4 else ref_enc  # [B,K,4]
                pc = pc[0, :, :2]
                if float(pc.abs().max()) > 1.5:
                    pc = pc.sigmoid()
                pcn = pc.cpu().numpy()
                for g, (x1, y1, x2, y2) in enumerate(gt_xyxy):
                    inside = ((pcn[:, 0] >= x1) & (pcn[:, 0] < x2) &
                              (pcn[:, 1] >= y1) & (pcn[:, 1] < y2)).sum()
                    prop_hit.add((bins[g],), 1.0 if inside > 0 else 0.0)
                    prop_count.add((bins[g],), float(inside))

            # per-layer boxes via the model's own heads (bbox_reparam math)
            delta = net.bbox_embed(hs)
            cxcy = delta[..., :2] * ref_unsig[..., 2:] + ref_unsig[..., :2]
            wh = delta[..., 2:].exp() * ref_unsig[..., 2:]
            boxes = torch.cat([cxcy, wh], dim=-1)          # [L,B,Q,4] cxcywh
            scores = net.class_embed(hs).sigmoid().max(-1).values  # [L,B,Q]
            n_layers = boxes.shape[0]
            bx = boxes[:, 0].cpu().numpy()
            sc = scores[:, 0].cpu().numpy()
            q_xyxy = np.stack(
                [bx[..., 0] - bx[..., 2] / 2, bx[..., 1] - bx[..., 3] / 2,
                 bx[..., 0] + bx[..., 2] / 2, bx[..., 1] + bx[..., 3] / 2], axis=-1
            )
            final_q, final_s = q_xyxy[-1], sc[-1]
            for g, gt in enumerate(gt_xyxy):
                ga = max((gt[2] - gt[0]) * (gt[3] - gt[1]), 1e-12)
                for layer in range(n_layers):
                    q = q_xyxy[layer]
                    ix1 = np.maximum(q[:, 0], gt[0]); iy1 = np.maximum(q[:, 1], gt[1])
                    ix2 = np.minimum(q[:, 2], gt[2]); iy2 = np.minimum(q[:, 3], gt[3])
                    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
                    qa = np.clip(q[:, 2] - q[:, 0], 0, None) * np.clip(q[:, 3] - q[:, 1], 0, None)
                    iou = inter / np.maximum(qa + ga - inter, 1e-12)
                    best = int(iou.argmax())
                    key_l = f"layer{layer}"
                    dec_iou.add((bins[g], key_l), float(iou[best]))
                    dec_found.add((bins[g], key_l), 1.0 if iou[best] >= 0.5 else 0.0)
                    dec_score.add((bins[g], key_l), float(sc[layer, best]))
                    if layer == n_layers - 1:
                        funnel.add((bins[g], "3:decoder-final-iou50"),
                                   1.0 if iou[best] >= 0.5 else 0.0)
        except Exception:
            print("[autopsy] WARNING: transformer section failed:")
            traceback.print_exc()

        # funnel stages 1-2 (token center at finest scale, proposal hit)
        try:
            hs0, ws0 = srcs[0].shape[2], srcs[0].shape[3]
            for g, (x1, y1, x2, y2) in enumerate(gt_xyxy):
                cx_in = ((np.arange(ws0) + 0.5) / ws0 >= x1) & ((np.arange(ws0) + 0.5) / ws0 < x2)
                cy_in = ((np.arange(hs0) + 0.5) / hs0 >= y1) & ((np.arange(hs0) + 0.5) / hs0 < y2)
                funnel.add((bins[g], "1:token-center@finest"),
                           1.0 if (cx_in.sum() * cy_in.sum()) > 0 else 0.0)
        except Exception:
            traceback.print_exc()

        if im["id"] in viz_ids:
            try:
                _save_image_viz(Path(im["file_name"]).stem, pil, res, gt_xyxy,
                                bins, srcs, pcn, final_q, final_s)
                print(f"[autopsy] viz saved for {im['file_name']}")
            except Exception:
                print("[autopsy] WARNING: viz failed:")
                traceback.print_exc()

        if idx % 100 == 0 or idx == len(images):
            print(f"[autopsy] {idx}/{len(images)} images traced", flush=True)

    # proposal funnel stage recorded via prop_hit
    for key, mean, n in prop_hit.rows():
        funnel.sum[(key[0], "2:proposal-hit")] = mean * n
        funnel.n[(key[0], "2:proposal-hit")] = n

    _print_mean_table("TOKEN COVERAGE (mean cells overlapping GT)", coverage, "{:.1f}")
    _print_mean_table("TOKEN COVERAGE (share of GT with NO cell center)", zero_center)
    _print_mean_table("FEATURE ENERGY (in-box norm / global norm)", energy)
    _print_mean_table("PROPOSAL COVERAGE (share of GT with proposal center inside)", prop_hit)
    _print_mean_table("PROPOSAL COVERAGE (mean proposals inside GT)", prop_count, "{:.1f}")
    _print_mean_table("DECODER EVOLUTION (mean best IoU)", dec_iou)
    _print_mean_table("DECODER EVOLUTION (found@IoU0.5 rate)", dec_found)
    _print_mean_table("DECODER EVOLUTION (score of best-IoU query)", dec_score)
    _print_mean_table("SURVIVAL FUNNEL (per size bin, stage survival rate)", funnel)

    # machine-readable dump: regenerate/restyle paper figures without a GPU
    tallies = {
        "coverage_cells": coverage, "zero_center_share": zero_center,
        "energy_snr": energy, "proposal_hit": prop_hit,
        "proposal_count": prop_count, "decoder_best_iou": dec_iou,
        "decoder_found50": dec_found, "decoder_best_score": dec_score,
        "survival_funnel": funnel,
    }
    data = {
        "resolution": res,
        "num_queries": int(net.num_queries),
        "stage_map": stage_info,
        "size_bin_edges": [e for e in cfg.SIZE_BIN_EDGES if e != float("inf")],
        "tallies": {
            name: {" | ".join(k) if isinstance(k, tuple) else str(k): [mean, n]
                   for k, mean, n in t.rows()}
            for name, t in tallies.items()
        },
    }
    json_path = _viz_dir() / "autopsy_data.json"
    with open(json_path, "w") as f:
        json.dump(data, f, indent=1)
    print(f"\n[autopsy] artifacts written to {_viz_dir().resolve()} "
          f"(qualitative PNGs + autopsy_data.json)")
    print("[autopsy] complete")
