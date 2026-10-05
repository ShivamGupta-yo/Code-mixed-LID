"""
src/utils.py - shared helpers for the lid-codemixed pipeline.

IMPORTANT: torch is NOT imported at module level. CUDA_VISIBLE_DEVICES must be
set before torch initialises CUDA, so setup_device() sets the env var first
and only then imports torch.
"""
import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import yaml

# Repo root = parent of src/. All paths in configs are relative to this,
# so the code does not depend on /raid/home/... and works on any machine.
REPO_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# GPU
# --------------------------------------------------------------------------- #
def setup_device(gpu_id=None):
    """Select the GPU and refuse to run on CPU.

    Priority:
      1. Inside a Slurm job, Slurm has already set CUDA_VISIBLE_DEVICES to the
         GPU it allocated to us. Overriding it could point at a GPU that
         belongs to someone else's job, so we keep Slurm's value.
      2. Outside Slurm, use gpu_id (from the config) if given.
      3. Otherwise, fall back to whatever the shell has set.
    """
    in_slurm = "SLURM_JOB_ID" in os.environ
    slurm_gpus = os.environ.get("CUDA_VISIBLE_DEVICES")

    if in_slurm and slurm_gpus is not None:
        source = f"Slurm allocation (job {os.environ['SLURM_JOB_ID']})"
    elif gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        source = "config gpu_id"
    else:
        source = "shell environment"

    import torch  # imported only AFTER CUDA_VISIBLE_DEVICES is final

    if not torch.cuda.is_available():
        # Never silently fall back to CPU: a 20-epoch mBERT run on CPU would
        # take days and could be mistaken for a working job.
        raise RuntimeError(
            "No CUDA GPU available. Are you inside an srun/sbatch job with "
            "--gres=gpu:1? CUDA_VISIBLE_DEVICES="
            f"{os.environ.get('CUDA_VISIBLE_DEVICES')!r}"
        )

    device = torch.device("cuda:0")  # index 0 among the *visible* GPUs
    print(f"[setup_device] using {torch.cuda.get_device_name(0)} "
          f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}, "
          f"source: {source})")
    return device


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed, deterministic=False):
    """Seed every RNG that affects training.

    deterministic=True forces deterministic cuDNN kernels: closer run-to-run
    results but slower. Even then, results are not guaranteed bit-identical
    across GPUs or library versions, hence we report mean +/- std over seeds.
    """
    import torch

    random.seed(seed)                 # Python: e.g. shuffling in splits
    np.random.seed(seed)              # NumPy
    torch.manual_seed(seed)           # CPU tensors, weight initialisation
    torch.cuda.manual_seed_all(seed)  # all visible GPUs
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------- #
# Config, paths, I/O
# --------------------------------------------------------------------------- #
def load_config(path):
    """Load a YAML config. Relative paths inside it are resolved later with
    resolve_path()."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(p):
    """Turn a repo-relative path (e.g. 'data/raw/lince_hineng') into an
    absolute one. Absolute paths are returned unchanged."""
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def save_json(obj, path):
    """Save metrics/results. ensure_ascii=False keeps Devanagari readable
    (needed later for COMI-LINGUA)."""
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def get_logger(name, log_file=None):
    """Logger that prints to screen and, optionally, to outputs/logs/."""
    logger = logging.getLogger(name)
    if logger.handlers:  # avoid duplicate lines if called twice
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file:
        log_file = resolve_path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def count_parameters(model):
    """(total, trainable) parameter counts. With fine-tuning enabled, both
    should be ~178M for mBERT + the BiLSTM/head."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable