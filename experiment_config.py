"""Single source of truth for every experiment knob.

Experiments branch off the baseline and edit THIS FILE (or model code) —
never the run command. Keep each change minimal and isolated so a branch
diff reads as the experiment's definition.
"""

# ---- model -----------------------------------------------------------------
MODEL_VARIANT = "base"  # rfdetr variant: nano | small | medium | base | large
RESOLUTION = 1008       # square input size; must be divisible by 56 for base
NUM_QUERIES = 900       # matches the parent checkpoint (queries-900 winner)
GPU_DEVICE = 2          # None -> auto; int -> pin train+eval to cuda:<idx>

# ---- checkpoint reuse (inference-time experiments) --------------------------
# Path to a trained checkpoint on the run host. When set, training is SKIPPED
# and evaluation runs on this exact model — the correct design for pure
# inference-time interventions (no seed variance). Missing file = hard error,
# never a silent 6h retrain.
REUSE_CHECKPOINT = "/home/kromanowski/.orx/runs/1f4d3fca-6f4a-410d-87b2-4520d07ef982/repo/output/checkpoint_best_ema.pth"

# ---- tiled (SAHI-style) evaluation ------------------------------------------
EVAL_TILED = False      # not used on this node
EVAL_STANDARD = False   # parent numbers already recorded; autopsy only
TILE_SIZE = 1008        # square tile side, in ORIGINAL image pixels
TILE_OVERLAP = 0.2      # fractional overlap between adjacent tiles
TILE_MERGE_IOU = 0.6    # class-aware NMS threshold when merging tile boxes
TILE_INCLUDE_FULL = True  # also merge a full-image pass (SAHI standard)

# ---- training --------------------------------------------------------------
EPOCHS = 24
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4    # effective batch = BATCH_SIZE * GRAD_ACCUM_STEPS
LR = 1e-4               # rfdetr defaults; listed here so sweeps are one-line diffs
LR_ENCODER = 1.5e-4
NUM_WORKERS = 8
CHECKPOINT_INTERVAL = 8
EARLY_STOPPING = False

# ---- data ------------------------------------------------------------------
HF_DATASET_REPO = "jeyanthangj2004/Visdrone-raw"  # original VisDrone2019-DET zips
DATA_ROOT = "data"       # raw zips + extracted splits
DATASET_DIR = "dataset"  # roboflow-layout view (train/valid/test) rfdetr consumes

# ---- evaluation / taxonomy -------------------------------------------------
EVAL_SCORE_THRESHOLD = 0.05   # keep low: recall analysis needs weak detections
FP_SCORE_THRESHOLD = 0.30     # false-positive taxonomy only counts confident FPs
MATCH_IOU = 0.5
# bins over sqrt(bbox area) in ORIGINAL image pixels; COCO "small" is < 32
SIZE_BIN_EDGES = [0, 8, 16, 32, 96, float("inf")]
DENSITY_BIN_EDGES = [0, 25, 50, 100, 200, float("inf")]  # GT objects per image

OUTPUT_DIR = "output"

# ---- autopsy (round 4) ------------------------------------------------------
RUN_AUTOPSY = True        # stage-by-stage information tracing
AUTOPSY_MAX_IMAGES = 548  # full val split
AUTOPSY_VIZ_IMAGES = 6    # densest images get qualitative PNG artifacts
