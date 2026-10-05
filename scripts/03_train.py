"""
scripts/03_train.py - train BERT + BiLSTM + softmax LID tagger.

Protocol (per seed):
  1. train on train split (official train minus 10% val, minus >512 sentences)
  2. after every epoch: loss + weighted F1 on the held-out VAL split
  3. early stopping on train.early_stop_metric (val_loss by default,
     or val_weighted_f1 / val_macro_f1); keep the best epoch
  4. reload the best epoch and evaluate ONCE on the official DEV split
The dev set never influences training or model selection.

Run from the repo root:
  python scripts/03_train.py                                  # all seeds
  python scripts/03_train.py --seeds 42 --max-epochs 1        # quick test
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from transformers import get_linear_schedule_with_warmup  # noqa: E402
from transformers import logging as hf_logging  # noqa: E402

from src.data import build_dataloaders  # noqa: E402
from src.metrics import compute_metrics, format_report, run_evaluation  # noqa
from src.model import build_model, get_param_groups  # noqa: E402
from src.utils import (count_parameters, get_logger, load_config,  # noqa
                       resolve_path, save_json, set_seed, setup_device)

hf_logging.set_verbosity_error()  # hide the harmless ">512 tokens" warning


def train_one_seed(cfg, seed, device, tokenizer, logger, max_epochs):
    exp = cfg["experiment_name"]
    tc = cfg["train"]
    set_seed(seed)

    loaders, label2id, id2label, data_stats = build_dataloaders(
        cfg, tokenizer, train_seed=seed, logger=logger)

    model = build_model(cfg, len(label2id)).to(device)
    total, trainable = count_parameters(model)
    logger.info(f"[seed {seed}] params total={total:,} trainable={trainable:,}")

    optimizer = torch.optim.AdamW(get_param_groups(
        model, tc["lr_encoder"], tc["lr_head"], tc["weight_decay"]))

    # Linear warmup then linear decay to 0 over the maximum number of steps.
    # Warmup avoids large, destructive updates to pretrained BERT weights in
    # the first steps, when the new BiLSTM/head is still random.
    total_steps = len(loaders["train"]) * max_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(tc["warmup_ratio"] * total_steps), total_steps)

    # Which validation number decides the "best" epoch.
    #   val_loss        -> lower is better  (original setting)
    #   val_weighted_f1 -> higher is better (tracks the reported metric)
    #   val_macro_f1    -> higher is better (favours rare tags; noisy)
    es_metric = tc.get("early_stop_metric", "val_loss")
    if es_metric not in ("val_loss", "val_weighted_f1", "val_macro_f1"):
        raise ValueError(f"unknown early_stop_metric: {es_metric}")
    higher_is_better = es_metric != "val_loss"
    best_score = -float("inf") if higher_is_better else float("inf")

    best_epoch, best_state = -1, None
    bad_epochs, history = 0, []
    t0 = time.time()

    for epoch in range(1, max_epochs + 1):
        # ---------------- train ----------------
        model.train()
        run_loss, run_words = 0.0, 0
        for batch in loaders["train"]:
            batch = {k: v.to(device) for k, v in batch.items() if k != "ids"}
            out = model(**batch)

            optimizer.zero_grad(set_to_none=True)
            out["loss"].backward()
            # clip exploding gradients (LSTMs are prone to them)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            optimizer.step()
            scheduler.step()

            n = int(batch["word_mask"].sum())
            run_loss += out["loss"].item() * n
            run_words += n
        train_loss = run_loss / run_words

        # ---------------- validate (held-out part of TRAIN) ----------------
        val = run_evaluation(model, loaders["val"], device)
        val_m = compute_metrics(val["gold"], val["pred"], id2label)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val["loss"],
                        "val_weighted_f1": val_m["weighted_f1"],
                        "val_macro_f1": val_m["macro_f1"]})
        logger.info(f"[seed {seed}] epoch {epoch:>2} | train_loss "
                    f"{train_loss:.4f} | val_loss {val['loss']:.4f} | "
                    f"val_wF1 {100 * val_m['weighted_f1']:.2f} | "
                    f"val_mF1 {100 * val_m['macro_f1']:.2f}")

        # ---------------- early stopping ----------------
        score = history[-1][es_metric]  # same keys as the history record
        improved = (score > best_score) if higher_is_better \
            else (score < best_score)
        if improved:
            best_score, best_epoch, bad_epochs = score, epoch, 0
            # copy weights to CPU RAM (~720 MB); avoids a disk round-trip
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= tc["patience"]:
                logger.info(f"[seed {seed}] early stop: no {es_metric} "
                            f"improvement for {bad_epochs} epochs")
                break

    train_time = time.time() - t0

    # ---------------- final evaluation on official DEV ----------------
    model.load_state_dict(best_state)
    dev = run_evaluation(model, loaders["dev"], device)
    dev_m = compute_metrics(dev["gold"], dev["pred"], id2label)
    logger.info(f"[seed {seed}] best epoch {best_epoch} "
                f"({es_metric}={best_score:.4f}) | DEV:\n"
                f"{format_report(dev_m)}")

    # ---------------- save ----------------
    out_dir = resolve_path(cfg["output"]["dir"])
    ckpt = out_dir / "checkpoints" / exp / f"seed_{seed}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": best_state, "label2id": label2id,
                "config": cfg, "seed": seed, "best_epoch": best_epoch}, ckpt)

    save_json({
        "experiment": exp, "seed": seed, "best_epoch": best_epoch,
        "early_stop_metric": es_metric, "best_val_score": best_score,
        "epochs_run": len(history), "train_time_sec": round(train_time, 1),
        "gpu": torch.cuda.get_device_name(0), "config": cfg,
        "data_stats": data_stats, "history": history,
        "dev_metrics": dev_m,
    }, out_dir / "results" / exp / f"seed_{seed}.json")

    # per-sentence dev predictions, for error analysis later
    pred_file = out_dir / "results" / exp / f"dev_predictions_seed_{seed}.jsonl"
    with open(pred_file, "w", encoding="utf-8") as f:
        for sid, g, p in zip(dev["ids"], dev["gold"], dev["pred"]):
            f.write(json.dumps({"id": sid,
                                "gold": [id2label[i] for i in g],
                                "pred": [id2label[i] for i in p]},
                               ensure_ascii=False) + "\n")

    del model, optimizer, best_state
    torch.cuda.empty_cache()
    return dev_m, best_epoch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/lince_hineng_mbert.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="override config seeds, e.g. --seeds 42")
    ap.add_argument("--max-epochs", type=int, default=None,
                    help="override config max_epochs (for quick tests)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seeds = args.seeds or cfg["train"]["seeds"]
    max_epochs = args.max_epochs or cfg["train"]["max_epochs"]
    exp = cfg["experiment_name"]

    logger = get_logger(exp, f"{cfg['output']['dir']}/logs/{exp}.log")
    device = setup_device()
    logger.info(f"experiment={exp} seeds={seeds} max_epochs={max_epochs}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["encoder_name"])

    results = {}
    for seed in seeds:
        dev_m, best_epoch = train_one_seed(cfg, seed, device, tokenizer,
                                           logger, max_epochs)
        results[seed] = {"weighted_f1": dev_m["weighted_f1"],
                         "macro_f1": dev_m["macro_f1"],
                         "accuracy": dev_m["accuracy"],
                         "best_epoch": best_epoch}

    # ---------------- summary across seeds ----------------
    summary = {"experiment": exp, "seeds": seeds, "per_seed": results,
               "early_stop_metric": cfg["train"].get("early_stop_metric",
                                                     "val_loss")}
    for key in ("weighted_f1", "macro_f1", "accuracy"):
        vals = np.array([r[key] for r in results.values()]) * 100
        # ddof=1: sample std; with a single seed there is no spread
        std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        summary[key] = {"mean": float(vals.mean()), "std": std}
        logger.info(f"DEV {key}: {vals.mean():.2f} ± {std:.2f} "
                    f"(n={len(vals)} seeds)")
    save_json(summary, f"{cfg['output']['dir']}/results/{exp}/summary.json")
    logger.info("paper reference (Table 13, mBERT+BiLSTM, ALL): 96.72")


if __name__ == "__main__":
    main()