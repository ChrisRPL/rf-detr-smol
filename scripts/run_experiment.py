"""Fixed entrypoint for every experiment node: prepare data, fine-tune, evaluate.

The run command (`python scripts/run_experiment.py`) never changes across the
experiment tree — behavior varies only through committed code, chiefly
experiment_config.py.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
)

import experiment_config as cfg  # noqa: E402
from scripts.prepare_data import prepare  # noqa: E402


def ensure_headless_cv2() -> None:
    """Replace broken GUI opencv builds with the headless one.

    The HF CUDA image ships opencv-python compiled against libGL, which the
    container lacks (`ImportError: libGL.so.1`), and it shadows any headless
    install. Uninstalling every variant and reinstalling headless is the only
    reliable in-place fix (run #2, d8f6ec8f, proved coexistence stays broken).
    """
    import importlib
    import subprocess

    try:
        importlib.import_module("cv2")
        print("[env] cv2 imports cleanly, no repair needed")
        return
    except ImportError as e:
        print(f"[env] cv2 broken ({e}); reinstalling opencv-python-headless")
    pip = [sys.executable, "-m", "pip"]
    subprocess.run(
        [*pip, "uninstall", "-q", "-y", "opencv-python", "opencv-contrib-python",
         "opencv-python-headless"],
        check=False,
    )
    subprocess.run(
        [*pip, "install", "-q", "--force-reinstall", "--no-deps",
         "opencv-python-headless>=4.9"],
        check=True,
    )


def check_env() -> None:
    """Fail fast — with the REAL traceback — if a runtime import is broken.

    rfdetr raises a generic "install albumentations" ImportError mid-training
    that hides the underlying cause (e.g. cv2/libGL or numpy ABI conflicts).
    """
    import importlib
    import traceback

    broken = []
    for mod in ("numpy", "cv2", "albumentations", "torch", "pycocotools", "rfdetr"):
        try:
            m = importlib.import_module(mod)
            print(f"[env] {mod} {getattr(m, '__version__', '?')}")
        except Exception:
            print(f"[env] IMPORT FAILURE: {mod}")
            traceback.print_exc()
            broken.append(mod)
    if broken:
        raise SystemExit(f"broken environment, imports failed: {broken}")


def build_model():
    import rfdetr

    variants = {
        "nano": "RFDETRNano",
        "small": "RFDETRSmall",
        "medium": "RFDETRMedium",
        "base": "RFDETRBase",
        "large": "RFDETRLarge",
    }
    cls = getattr(rfdetr, variants[cfg.MODEL_VARIANT])
    kwargs: dict = {"resolution": cfg.RESOLUTION}
    if cfg.NUM_QUERIES is not None:
        kwargs["num_queries"] = cfg.NUM_QUERIES
        kwargs["num_select"] = cfg.NUM_QUERIES
    if getattr(cfg, "GPU_DEVICE", None) is not None:
        kwargs["device"] = f"cuda:{cfg.GPU_DEVICE}"
    print(f"[model] {cls.__name__}({kwargs})")
    return cls(**kwargs)


def pick_eval_model(trained_model, output_dir: Path):
    """Prefer the best checkpoint on disk; fall back to the in-memory model."""
    checkpoints = sorted(output_dir.rglob("*.pth"), key=lambda p: p.stat().st_mtime)
    ranked = (
        [p for p in checkpoints if "best_ema" in p.name]
        or [p for p in checkpoints if "best" in p.name]
        or checkpoints
    )
    if ranked:
        path = ranked[-1]
        try:
            from rfdetr.detr import RFDETR

            ckpt_kwargs = {}
            if getattr(cfg, "GPU_DEVICE", None) is not None:
                ckpt_kwargs["device"] = f"cuda:{cfg.GPU_DEVICE}"
            model = RFDETR.from_checkpoint(path, **ckpt_kwargs)
            print(f"[eval] evaluating checkpoint: {path}")
            return model
        except Exception as e:  # fall back, but SAY so — never evaluate silently
            print(f"[eval] WARNING: from_checkpoint({path}) failed: {e!r}")
    print("[eval] evaluating in-memory model from the training run")
    return trained_model


def main() -> None:
    t0 = time.time()
    knobs = {k: v for k, v in vars(cfg).items() if k.isupper()}
    print(f"== CONFIG ==\n{knobs}")

    print("\n== ENV CHECK ==", flush=True)
    ensure_headless_cv2()
    check_env()

    print("\n== DATA PREP ==", flush=True)
    dataset_dir = prepare(cfg.HF_DATASET_REPO, cfg.DATA_ROOT, cfg.DATASET_DIR)
    print(f"[data] prepared in {time.time() - t0:.0f}s")

    print("\n== TRAIN ==", flush=True)
    model = build_model()
    t1 = time.time()
    model.train(
        dataset_dir=str(dataset_dir),
        epochs=cfg.EPOCHS,
        batch_size=cfg.BATCH_SIZE,
        grad_accum_steps=cfg.GRAD_ACCUM_STEPS,
        lr=cfg.LR,
        lr_encoder=cfg.LR_ENCODER,
        num_workers=cfg.NUM_WORKERS,
        checkpoint_interval=cfg.CHECKPOINT_INTERVAL,
        early_stopping=cfg.EARLY_STOPPING,
        output_dir=cfg.OUTPUT_DIR,
        tensorboard=False,
    )
    print(f"[train] finished in {(time.time() - t1) / 60:.1f} min")

    print("\n== EVALUATE ==", flush=True)
    from scripts.evaluate import evaluate

    eval_model = pick_eval_model(model, Path(cfg.OUTPUT_DIR))
    evaluate(eval_model, str(dataset_dir))

    print(f"\n== RUN COMPLETE == total {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
