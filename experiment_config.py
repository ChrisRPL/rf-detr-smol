"""Single source of truth for every experiment knob.

Round 5 cross-architecture node (Ultralytics family). The taxonomy/eval harness
(evaluate.py) is model-agnostic — it consumes COCO-format detections — so YOLO
and RT-DETR feed the IDENTICAL failure taxonomy as RF-DETR, against the same
val GT (with occlusion/truncation), keeping the comparison clean.
"""

# ---- model family -----------------------------------------------------------
MODEL_FAMILY = "ultralytics"   # "rfdetr" | "ultralytics"
ULTRA_WEIGHTS = "rtdetr-l.pt"   # pretrained checkpoint ultralytics auto-downloads
RESOLUTION = 1024              # imgsz; matches RF-DETR's high-res operating point
GPU_DEVICE = 1

# ---- training ---------------------------------------------------------------
EPOCHS = 60                    # YOLO/RT-DETR converge slower per-epoch than DETR ft
BATCH_SIZE = 16
LR = 0.01                      # ultralytics default (SGD); listed for one-line sweeps
NUM_WORKERS = 8

# ---- data -------------------------------------------------------------------
HF_DATASET_REPO = "jeyanthangj2004/Visdrone-raw"
DATA_ROOT = "data"
DATASET_DIR = "dataset"        # roboflow-layout COCO view (shared with rfdetr path)
YOLO_DIR = "dataset_yolo"      # YOLO-format view built from the COCO view

# ---- evaluation / taxonomy (shared with the rfdetr harness) -----------------
EVAL_SCORE_THRESHOLD = 0.05
FP_SCORE_THRESHOLD = 0.30
MATCH_IOU = 0.5
SIZE_BIN_EDGES = [0, 8, 16, 32, 96, float("inf")]
DENSITY_BIN_EDGES = [0, 25, 50, 100, 200, float("inf")]
EVAL_TILED = False
EVAL_STANDARD = True

OUTPUT_DIR = "output"
