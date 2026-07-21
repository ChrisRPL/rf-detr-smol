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

            model = RFDETR.from_checkpoint(path)
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
