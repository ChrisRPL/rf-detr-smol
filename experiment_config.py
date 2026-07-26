"""Single source of truth for every experiment knob.

Gate 3 — NVESD ATR MWIR baseline. Fine-tune RF-DETR-Base on the pre-built
thermal range-study dataset (already on the run host, no download), then run
the SAME failure taxonomy (now including recall-by-range) to test whether the
tokenization size-cliff found on VisDrone reproduces on real long-range
thermal ATR targets.
"""

# ---- model -----------------------------------------------------------------
MODEL_VARIANT = "base"
RESOLUTION = 560         # baseline, for comparability with the VisDrone baseline
NUM_QUERIES = None       # rfdetr default (300); ATR is sparse (1-2 targets/frame)
GPU_DEVICE = 1

# ---- data (pre-built local COCO/roboflow dataset on the run host) -----------
DATA_LOCAL_DIR = "/home/kromanowski/datasets/nvesd_atr/nvesd_atr_cegr_heldout"
HF_DATASET_REPO = ""     # unused when DATA_LOCAL_DIR is set
DATA_ROOT = "data"
DATASET_DIR = "dataset"

# ---- training --------------------------------------------------------------
EPOCHS = 40              # small dataset (1790 train); more epochs than VisDrone
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 2
LR = 1e-4
LR_ENCODER = 1.5e-4
NUM_WORKERS = 8
CHECKPOINT_INTERVAL = 10
EARLY_STOPPING = False

# ---- evaluation / taxonomy -------------------------------------------------
EVAL_STANDARD = True
EVAL_TILED = False
EVAL_SCORE_THRESHOLD = 0.05
FP_SCORE_THRESHOLD = 0.30
MATCH_IOU = 0.5
SIZE_BIN_EDGES = [0, 8, 16, 32, 96, float("inf")]
DENSITY_BIN_EDGES = [0, 25, 50, 100, 200, float("inf")]

OUTPUT_DIR = "output"

# ---- analysis toggles ------------------------------------------------------
RUN_AUTOPSY = False       # enable on the descendant node once the baseline lands
RUN_CONFUSION = False
AUTOPSY_MAX_IMAGES = 898
AUTOPSY_VIZ_IMAGES = 6
