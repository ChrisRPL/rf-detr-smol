"""Fixed entrypoint for every experiment node: prepare data, train, evaluate.

The run command (`python scripts/run_experiment.py`) never changes across the
experiment tree. Behavior varies only through committed code — here, the model
family is selected by experiment_config.MODEL_FAMILY, so RF-DETR and the
Ultralytics cross-architecture controls share this one entrypoint and the same
downstream taxonomy.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import experiment_config as cfg  # noqa: E402

# Pin the GPU before anything initializes CUDA (see the rfdetr branches' notes).
if getattr(cfg, "GPU_DEVICE", None) is not None:
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.GPU_DEVICE)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
)

from scripts.prepare_data import prepare  # noqa: E402


def ensure_headless_cv2() -> None:
    """Replace broken GUI opencv builds with the headless one (libGL missing)."""
    import importlib
    import subprocess

    try:
        importlib.import_module("cv2")
        print("[env] cv2 imports cleanly, no repair needed")
        return
    except ImportError as e:
        print(f"[env] cv2 broken ({e}); reinstalling opencv-python-headless")
    pip = [sys.executable, "-m", "pip"]
    subprocess.run([*pip, "uninstall", "-q", "-y", "opencv-python",
                    "opencv-contrib-python", "opencv-python-headless"], check=False)
    subprocess.run([*pip, "install", "-q", "--force-reinstall", "--no-deps",
                    "opencv-python-headless>=4.9"], check=True)


def main() -> None:
    t0 = time.time()
    knobs = {k: v for k, v in vars(cfg).items() if k.isupper()}
    print(f"== CONFIG ==\n{knobs}")

    print("\n== ENV CHECK ==", flush=True)
    ensure_headless_cv2()

    print("\n== DATA PREP ==", flush=True)
    dataset_dir = prepare(cfg.HF_DATASET_REPO, cfg.DATA_ROOT, cfg.DATASET_DIR)
    print(f"[data] prepared in {time.time() - t0:.0f}s")

    family = getattr(cfg, "MODEL_FAMILY", "rfdetr")
    print(f"\n== RUN ({family}) ==", flush=True)
    if family == "ultralytics":
        from scripts.run_ultralytics import main as ultra_main

        ultra_main(dataset_dir)
    else:
        raise SystemExit(
            f"MODEL_FAMILY={family!r} not supported on this branch; "
            "the rfdetr entrypoint lives on the rfdetr branches."
        )
    print(f"\n== RUN COMPLETE == total {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
