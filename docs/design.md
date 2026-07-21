# Design: RF-DETR small-object failure study on VisDrone

Date: 2026-07-21 · Status: approved (chat session)

## Research question & hypothesis

Where and why does RF-DETR fail on small / distant / visually weak objects?
Hypothesized mechanisms: (1) resolution bottleneck, (2) feature compression,
(3) query-matching limits, (4) weak local detail representation.

## Approach (decided)

- **Baseline = fine-tune first** (option B): COCO-pretrained RF-DETR-Base
  fine-tuned on VisDrone2019-DET train, analyzed on val. Zero-shot was
  rejected: VisDrone's 10 classes don't map cleanly onto COCO's 80, which
  would contaminate the failure taxonomy with label-space artifacts.
- **Model**: RF-DETR-Base (~29M params), default 560 px input, default 300
  queries — the baseline must *exhibit* the hypothesized failures, not
  pre-fix them.
- **Compute**: HF Jobs (`--backend hf`), A100, ~8 h timeout.
- **Data**: HF Hub mirror `jeyanthangj2004/Visdrone-raw` (original VisDrone
  zips; layout verified). Downloaded fresh each run for reproducibility.

## Fixed run contract

Run command (identical on every node, never edited):

    python -m pip install -q -r requirements.txt && python scripts/run_experiment.py

All knobs live in `experiment_config.py`; experiments are branches that edit
code/config only. Deps are pinned in `requirements.txt` (also code).

## Evidence printed to the run log

- Per-epoch training progress (rfdetr/PTL logging).
- Final: COCO AP / AP50 / AP_small/medium/large (maxDets=500), per-class AP.
- Failure taxonomy: GT recall by size bin (<8, 8–16, 16–32, 32–96, >96 px
  sqrt-area) × occlusion × truncation × class × image density; FP breakdown
  (background / localization / classification / duplicate).
- The converter carries VisDrone's occlusion/truncation flags into the COCO
  JSON (validated on real val split: 548 imgs, 38 759 boxes, 68.6% small).

## Tree plan

1. Baseline node (frozen after first successful run).
2. Taxonomy report → Files tab.
3. Round 1: 2–3 siblings off the baseline attacking the mechanism the
   taxonomy implicates most (e.g. resolution 560/728/1008, or query count
   300/500/900). Winner becomes round 2's parent.
