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


def expand_query_checkpoint(cls, num_queries: int) -> str:
    """Tile the pretrained query embeddings to support num_queries > 300.

    COCO-pretrained RF-DETR ships 300 queries; refpoint_embed/query_feat are
    Embedding(group_detr * 300, d) laid out group-major (inference slices
    weight[:num_queries] as group 0). Loading with num_queries=600/900 fails on
    shape mismatch (run c1626f69), so we expand the checkpoint: reshape to
    (groups, 300, d), tile along the query dim, add small noise to the copies
    so duplicated queries diverge during training, and keep copy 0 exact.
    """
    import torch

    throwaway = cls(device="cpu")  # default 300-query build; downloads weights
    src = Path(str(throwaway.model_config.pretrain_weights or "")).expanduser()
    group_detr = throwaway.model_config.group_detr
    base_queries = throwaway.model_config.num_queries
    del throwaway
    if not src.is_file():
        src = Path.home() / ".roboflow" / "models" / "rf-detr-base.pth"
    factor, rem = divmod(num_queries, base_queries)
    if rem:
        raise SystemExit(
            f"NUM_QUERIES={num_queries} must be a multiple of {base_queries}"
        )
    dst = src.with_name(f"{src.stem}-q{num_queries}{src.suffix}")
    if dst.is_file():
        print(f"[model] reusing expanded checkpoint {dst}")
        return str(dst)

    try:
        ckpt = torch.load(src, map_location="cpu", weights_only=True)
    except Exception:
        # The checkpoint stores non-tensor objects (e.g. train args). It is
        # Roboflow's official artifact, MD5-validated by rfdetr just above in
        # the throwaway build, so unpickling it is as trusted as rfdetr itself.
        ckpt = torch.load(src, map_location="cpu", weights_only=False)
    sd = ckpt["model"]
    for key in ("refpoint_embed.weight", "query_feat.weight"):
        w = sd[key]
        n, d = w.shape
        per_group = n // group_detr
        tiled = w.reshape(group_detr, per_group, d).repeat(1, factor, 1)
        noise = torch.randn_like(tiled) * (0.01 * w.std())
        noise[:, :per_group, :] = 0  # first copy stays exactly pretrained
        sd[key] = (tiled + noise).reshape(-1, d)
        print(f"[model] expanded {key}: {(n, d)} -> {tuple(sd[key].shape)}")
    torch.save(ckpt, dst)
    print(f"[model] wrote expanded checkpoint {dst}")
    return str(dst)


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
        if cfg.NUM_QUERIES != 300:
            kwargs["pretrain_weights"] = expand_query_checkpoint(
                cls, cfg.NUM_QUERIES
            )
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
    train_kwargs = {}
    if getattr(cfg, "GPU_DEVICE", None) is not None:
        # ModelConfig.device only places the initial weights; the PTL trainer
        # picks its own device and defaults to cuda:0 (run 4a42d7fa OOM'd when
        # two siblings collided there). train()'s device kwarg maps to
        # accelerator="gpu", devices=[N] — the pin that actually sticks.
        train_kwargs["device"] = f"cuda:{cfg.GPU_DEVICE}"
    model.train(
        dataset_dir=str(dataset_dir),
        **train_kwargs,
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
