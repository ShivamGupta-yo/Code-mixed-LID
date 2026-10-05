"""
scripts/04_evaluate.py - error analysis of finished runs (no GPU).

For each experiment in outputs/results/ (or those given with --experiments):
  1. per-tag precision / recall / F1, mean +/- std over seeds, with support
  2. confusion matrix averaged over seeds (rows = gold, cols = predicted)
  3. most frequent word-level errors (word, gold -> predicted), over all seeds,
     using dev words joined from the dev parquet by sentence id
Writes outputs/results/<exp>/error_analysis.json and errors_top.tsv.

Subset mode (--subset cm | mono): instead of the full report, re-score every
run on only the code-mixed (or only the monolingual) dev sentences, using the
saved dev predictions. Code-mixed = gold tags contain both lang1 and lang2, or
a 'mixed' tag (same rule as cm_only training in src/data.py).

Run from the repo root (login node is fine, no GPU needed):
  python scripts/04_evaluate.py
  python scripts/04_evaluate.py --experiments lince_hineng_mbert_bilstm_esf1
  python scripts/04_evaluate.py --subset cm
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from sklearn.metrics import accuracy_score, precision_recall_fscore_support  # noqa: E402

from src.data import is_code_mixed, load_split  # noqa: E402
from src.utils import resolve_path, save_json  # noqa: E402


def load_experiment(res_dir):
    summary = json.loads((res_dir / "summary.json").read_text(encoding="utf-8"))
    seeds = {}
    for s in summary["seeds"]:
        f = res_dir / f"seed_{s}.json"
        if f.exists():
            seeds[s] = json.loads(f.read_text(encoding="utf-8"))
    return summary, seeds


def per_tag_table(seeds):
    """mean/std of P, R, F1 per tag across seeds; support from the first seed
    (identical for all seeds: same dev set)."""
    first = next(iter(seeds.values()))["dev_metrics"]["per_class"]
    rows = {}
    for tag in first:
        vals = {k: np.array([100 * r["dev_metrics"]["per_class"][tag][k]
                             for r in seeds.values()])
                for k in ("precision", "recall", "f1")}
        n_pred = np.array([r["dev_metrics"]["per_class"][tag]["n_predicted"]
                           for r in seeds.values()])
        rows[tag] = {
            **{f"{k}_mean": float(v.mean()) for k, v in vals.items()},
            **{f"{k}_std": float(v.std(ddof=1)) if len(v) > 1 else 0.0
               for k, v in vals.items()},
            "support": first[tag]["support"],
            "n_predicted_mean": float(n_pred.mean()),
        }
    return rows


def mean_confusion(seeds):
    labels = next(iter(seeds.values()))["dev_metrics"]["confusion_matrix"]["labels"]
    mats = [np.array(r["dev_metrics"]["confusion_matrix"]["matrix"])
            for r in seeds.values()]
    return labels, np.mean(mats, axis=0)


def word_errors(res_dir, seeds, dev_words):
    """Count (word, gold, pred) errors over all seeds."""
    errors, n_tokens = Counter(), 0
    for s in seeds:
        f = res_dir / f"dev_predictions_seed_{s}.jsonl"
        if not f.exists():
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            words = dev_words[rec["id"]]
            if len(words) != len(rec["gold"]):  # alignment safety check
                raise ValueError(f"length mismatch for {rec['id']}")
            for w, g, p in zip(words, rec["gold"], rec["pred"]):
                n_tokens += 1
                if g != p:
                    errors[(w.lower(), g, p)] += 1
    return errors, n_tokens


def print_report(exp, summary, seeds, tag_rows, labels, cm, errors, n_tok, top):
    stop = summary.get("early_stop_metric", "val_loss")
    print(f"\n{'=' * 78}\n{exp}  (stop on {stop}, {len(seeds)} seeds)\n{'=' * 78}")

    print("\nPer-tag (dev, mean ± std over seeds):")
    print(f"{'tag':<10}{'P':>14}{'R':>14}{'F1':>14}{'support':>9}{'n_pred':>9}")
    for tag, r in sorted(tag_rows.items(), key=lambda kv: -kv[1]["support"]):
        print(f"{tag:<10}"
              f"{r['precision_mean']:>8.2f}±{r['precision_std']:<5.2f}"
              f"{r['recall_mean']:>8.2f}±{r['recall_std']:<5.2f}"
              f"{r['f1_mean']:>8.2f}±{r['f1_std']:<5.2f}"
              f"{r['support']:>9}{r['n_predicted_mean']:>9.1f}")

    print("\nConfusion matrix (mean over seeds; rows = gold, cols = predicted):")
    order = sorted(range(len(labels)), key=lambda i: -cm[i].sum())
    print("gold\\pred".ljust(11) + "".join(labels[j][:7].rjust(8) for j in order))
    for i in order:
        print(labels[i].ljust(11) + "".join(f"{cm[i, j]:8.1f}" for j in order))

    n_err = sum(errors.values())
    print(f"\nWord-level errors: {n_err} over {n_tok} predicted tokens "
          f"({100 * n_err / max(n_tok, 1):.2f}% across all seeds)")
    pairs = Counter()
    for (_, g, p), c in errors.items():
        pairs[(g, p)] += c
    print("Most frequent error types (gold -> pred):")
    for (g, p), c in pairs.most_common(8):
        print(f"  {g:>9} -> {p:<9} {c:>6}  ({100 * c / max(n_err, 1):.1f}% of errors)")
    print(f"Top {top} most frequent wrongly tagged words (summed over seeds):")
    for (w, g, p), c in errors.most_common(top):
        print(f"  {w:<20} {g:>9} -> {p:<9} {c:>4}")


def subset_scores(res_dir, seeds, subset):
    """Re-score each seed's saved dev predictions on a subset of sentences."""
    per_seed, n_sent, n_tok = [], 0, 0
    for s in seeds:
        f = res_dir / f"dev_predictions_seed_{s}.jsonl"
        if not f.exists():
            continue
        gold, pred, n_sent = [], [], 0
        for line in f.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            cm = is_code_mixed(rec["gold"])
            if (subset == "cm") != cm:
                continue
            n_sent += 1
            gold.extend(rec["gold"]); pred.extend(rec["pred"])
        n_tok = len(gold)
        if not gold:
            continue
        tags = sorted(set(gold))
        _, _, wf, _ = precision_recall_fscore_support(
            gold, pred, labels=tags, average="weighted", zero_division=0)
        _, _, mf, _ = precision_recall_fscore_support(
            gold, pred, labels=tags, average="macro", zero_division=0)
        per_seed.append((100 * wf, 100 * mf, 100 * accuracy_score(gold, pred)))
    return np.array(per_seed), n_sent, n_tok


def fmt(vals):
    sd = vals.std(ddof=1) if len(vals) > 1 else 0.0
    return f"{vals.mean():6.2f} ± {sd:4.2f}"


def print_subset_table(root, exps, subset):
    name = "code-mixed" if subset == "cm" else "monolingual"
    print(f"\nScores on {name} dev sentences only (mean ± std over seeds)\n")
    print(f"{'experiment':<38}{'stop on':<17}{'sent':>5}{'tokens':>8}"
          f"{'weighted F1':>16}{'macro F1':>16}{'accuracy':>16}")
    for exp in exps:
        res_dir = root / exp
        summary, seeds = load_experiment(res_dir)
        if not seeds:
            continue
        sc, n_sent, n_tok = subset_scores(res_dir, seeds, subset)
        if len(sc) == 0:
            continue
        stop = summary.get("early_stop_metric", "val_loss")
        print(f"{exp:<38}{stop:<17}{n_sent:>5}{n_tok:>8}"
              f"{fmt(sc[:, 0]):>16}{fmt(sc[:, 1]):>16}{fmt(sc[:, 2]):>16}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="outputs/results")
    ap.add_argument("--experiments", nargs="+", default=None)
    ap.add_argument("--dev-path",
                    default="data/raw/lince_hineng/dev-00000-of-00001.parquet")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--subset", choices=["cm", "mono"], default=None,
                    help="score only code-mixed (cm) or monolingual (mono) dev sentences")
    args = ap.parse_args()

    root = resolve_path(args.results_dir)
    exps = args.experiments or sorted(
        d.name for d in root.iterdir() if (d / "summary.json").exists())
    if args.subset:
        print_subset_table(root, exps, args.subset)
        return

    dev_words = {e["id"]: e["words"] for e in load_split(args.dev_path)}

    for exp in exps:
        res_dir = root / exp
        summary, seeds = load_experiment(res_dir)
        if not seeds:
            print(f"\n{exp}: no seed files, skipped")
            continue
        tag_rows = per_tag_table(seeds)
        labels, cm = mean_confusion(seeds)
        errors, n_tok = word_errors(res_dir, seeds, dev_words)
        print_report(exp, summary, seeds, tag_rows, labels, cm, errors,
                     n_tok, args.top)

        save_json({"experiment": exp, "per_tag": tag_rows,
                   "confusion_mean": {"labels": labels,
                                      "matrix": cm.round(2).tolist()},
                   "n_errors": sum(errors.values()), "n_tokens": n_tok},
                  res_dir / "error_analysis.json")
        with open(res_dir / "errors_top.tsv", "w", encoding="utf-8") as f:
            f.write("word\tgold\tpred\tcount_over_seeds\n")
            for (w, g, p), c in errors.most_common():
                f.write(f"{w}\t{g}\t{p}\t{c}\n")


if __name__ == "__main__":
    main()