"""
scripts/00_smoke_test.py - quick end-to-end check of data + model.

Run from the repo root:
  python scripts/00_smoke_test.py
  python scripts/00_smoke_test.py --config configs/lince_hineng_mbert.yaml
"""
import argparse
import sys
from pathlib import Path

# Make `import src...` work when running a file inside scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoTokenizer, logging as hf_logging  # noqa: E402

from src.data import build_dataloaders  # noqa: E402
from src.model import build_model  # noqa: E402
from src.utils import (count_parameters, load_config, set_seed,  # noqa: E402
                       setup_device)

hf_logging.set_verbosity_error()  # hide the harmless ">512 tokens" warning


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/lince_hineng_mbert.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = setup_device()
    set_seed(42)

    tok = AutoTokenizer.from_pretrained(cfg["model"]["encoder_name"])
    loaders, l2i, _, _ = build_dataloaders(cfg, tok, train_seed=42)

    model = build_model(cfg, len(l2i)).to(device)
    total, trainable = count_parameters(model)
    print(f"params: total={total:,} trainable={trainable:,}")

    batch = next(iter(loaders["train"]))
    batch = {k: v.to(device) for k, v in batch.items() if k != "ids"}
    print({k: tuple(v.shape) for k, v in batch.items()})

    out = model(**batch)
    print(f"logits: {tuple(out['logits'].shape)} | "
          f"loss: {out['loss'].item():.4f} (expected ~2.08 = ln 8)")

    out["loss"].backward()
    n_grad = sum(p.grad is not None for p in model.parameters())
    n_all = sum(1 for _ in model.parameters())
    print(f"backward OK: {n_grad}/{n_all} parameter tensors have gradients")


if __name__ == "__main__":
    main()