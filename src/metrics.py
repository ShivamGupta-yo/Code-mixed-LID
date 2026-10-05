"""
src/metrics.py - evaluation loop and token-level LID metrics.

Everything is computed at the WORD level over all real words in the split
(padding removed), which is how token-classification LID is scored.

Reported numbers:
  weighted F1 : per-class F1 averaged with weights = gold support.
                Most likely what the paper's tables report (the paper never
                says which average it used). Dominated by lang1/lang2/other.
  macro F1    : unweighted mean over classes that occur in gold. Exposes
                failures on rare tags (fw, mixed, unk, ambiguous).
  micro F1    : for single-label multi-class tagging this equals accuracy.
"""
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)


@torch.no_grad()
def run_evaluation(model, loader, device):
    """Run the model over a loader. Returns the word-weighted mean loss and
    per-sentence gold/pred label ids (padding removed), keyed by sentence id
    so errors can be traced back to the data later.
    The caller must switch back to model.train() afterwards."""
    model.eval()
    total_loss, total_words = 0.0, 0
    ids, gold, pred = [], [], []

    for batch in loader:
        batch_ids = batch["ids"]
        batch = {k: v.to(device) for k, v in batch.items() if k != "ids"}
        out = model(**batch)

        # CrossEntropyLoss returns the mean over the batch's real words.
        # Multiplying back by the word count gives a loss that is averaged
        # over WORDS for the whole split, not over batches of unequal size.
        n_words = int(batch["word_mask"].sum())
        total_loss += out["loss"].item() * n_words
        total_words += n_words

        p = out["logits"].argmax(dim=-1)
        for i, sid in enumerate(batch_ids):
            m = batch["word_mask"][i]
            ids.append(sid)
            gold.append(batch["labels"][i][m].tolist())
            pred.append(p[i][m].tolist())

    return {"loss": total_loss / total_words, "ids": ids,
            "gold": gold, "pred": pred}


def compute_metrics(gold_seqs, pred_seqs, id2label):
    """Token-level metrics from per-sentence label-id lists."""
    y_true = np.array([t for s in gold_seqs for t in s])
    y_pred = np.array([t for s in pred_seqs for t in s])
    all_ids = sorted(id2label)                       # every known tag
    gold_ids = sorted(set(y_true.tolist()))          # tags present in gold

    # zero_division=0: a tag that is never predicted gets precision 0
    # instead of a warning (e.g. 'ambiguous' with 1 dev token)
    w_p, w_r, w_f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=all_ids, average="weighted", zero_division=0)
    _, _, macro_f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=gold_ids, average="macro", zero_division=0)
    _, _, micro_f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=all_ids, average="micro", zero_division=0)

    p, r, f, support = precision_recall_fscore_support(
        y_true, y_pred, labels=all_ids, average=None, zero_division=0)
    n_pred = [int((y_pred == i).sum()) for i in all_ids]

    per_class = {
        id2label[i]: {"precision": float(p[k]), "recall": float(r[k]),
                      "f1": float(f[k]), "support": int(support[k]),
                      "n_predicted": n_pred[k]}
        for k, i in enumerate(all_ids)
    }

    return {
        "n_words": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_precision": float(w_p),
        "weighted_recall": float(w_r),
        "weighted_f1": float(w_f),
        "macro_f1": float(macro_f),
        "macro_f1_classes": [id2label[i] for i in gold_ids],
        "micro_f1": float(micro_f),
        "per_class": per_class,
        "confusion_matrix": {
            "labels": [id2label[i] for i in all_ids],   # rows = gold
            "matrix": confusion_matrix(y_true, y_pred,  # cols = predicted
                                       labels=all_ids).tolist(),
        },
    }


def format_report(m):
    """Readable summary for logs."""
    lines = [
        f"words={m['n_words']}  acc={100 * m['accuracy']:.2f}  "
        f"weighted_F1={100 * m['weighted_f1']:.2f}  "
        f"macro_F1={100 * m['macro_f1']:.2f}  "
        f"micro_F1={100 * m['micro_f1']:.2f}",
        f"{'tag':<10}{'P':>8}{'R':>8}{'F1':>8}{'support':>9}{'n_pred':>8}",
    ]
    for tag, c in sorted(m["per_class"].items(),
                         key=lambda kv: -kv[1]["support"]):
        lines.append(f"{tag:<10}{100 * c['precision']:>8.2f}"
                     f"{100 * c['recall']:>8.2f}{100 * c['f1']:>8.2f}"
                     f"{c['support']:>9}{c['n_predicted']:>8}")
    return "\n".join(lines)