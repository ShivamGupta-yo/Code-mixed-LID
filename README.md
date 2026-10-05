# lid-codemixed

Word-level language identification (LID) for code-mixed text: a replication of
**Chanda & Pal (2025)**, *Word Level Language Identification from Social Media
Code-Mixed Data Leveraging Transformer-Based Models*, SN Computer Science 6:844
(doi: 10.1007/s42979-025-04377-4), on the **LinCE Hindi-English** benchmark,
with additional ablations.

Model: mBERT (`bert-base-multilingual-cased`) → one vector per word → Bi-LSTM →
linear → one language tag per word (`lang1`, `lang2`, `other`, `ne`, `fw`,
`mixed`, `unk`, `ambiguous`).

## Results (LinCE HI-EN dev set, weighted F1, mean ± std over 3 seeds)

| Run | Config | Weighted F1 | Macro F1 |
|---|---|---|---|
| Paper (single run) | — | 96.72 | — |
| mBERT + Bi-LSTM, early stopping on val loss | `lince_hineng_mbert.yaml` | 96.45 ± 0.02 | 49.27 ± 2.25 |
| **mBERT + Bi-LSTM, early stopping on val F1 (main)** | `lince_hineng_mbert_esf1.yaml` | **96.74 ± 0.11** | 52.71 ± 0.41 |
| + mean pooling of sub-tokens | `lince_hineng_mbert_meanpool_esf1.yaml` | 96.64 ± 0.15 | 52.27 ± 0.78 |
| without Bi-LSTM | `lince_hineng_mbert_nobilstm_esf1.yaml` | 96.76 ± 0.07 | 59.37 ± 2.91 |
| trained on code-mixed sentences only (paper: 94.78) | `lince_hineng_mbert_esf1_cm.yaml` | 96.11 ± 0.03 | 48.81 ± 2.01 |

Code-mixed vs. monolingual dev sentences (weighted F1):

| Trained on | Code-mixed (326 sent.) | Monolingual (418 sent.) |
|---|---|---|
| All sentences | 95.92 ± 0.18 | 97.56 ± 0.03 |
| Code-mixed only | 95.95 ± 0.05 | 96.36 ± 0.03 |

Main findings: the paper's result reproduces (96.74 vs. 96.72); stopping on
validation F1 instead of loss explains the initial gap; neither the Bi-LSTM nor
mean pooling improves weighted F1; code-mixed-only training loses accuracy only
on monolingual sentences.

## Protocol

- Data: LinCE HI-EN (HuggingFace mirror `HuggMachas/Lince_benchmark_lid_hineng_train_valid`).
  Test labels are not public, so the **dev set is the evaluation set** (as in the paper).
- Verified against the paper: 4,823 / 744 sentences and 95,224 / 15,446 tokens (Table 1);
  dev tag counts match Table 7 except one token (`other` 2,231 / `ambiguous` 1 vs. 2,230 / 2).
- 2 training sentences exceeding mBERT's 512-subtoken limit (ids 217, 1902) are removed; dev is unchanged.
- Split: train 4,339 · validation 482 (10% held out from train, `split_seed` 13, used **only** for
  early stopping) · dev 744 (used **once**, for final evaluation).
- Training: batch 32, ≤ 20 epochs, patience 3, AdamW, LR 2e-5 (mBERT) / 1e-3 (Bi-LSTM + head),
  10% warmup, dropout 0.5, gradient clipping 1.0, seeds 42 / 43 / 44, one NVIDIA A100 (40 GB).
- Sub-word → word: first sub-token by default (`pooling: first`), optionally the mean (`pooling: mean`).

## Repository structure

```
configs/            one YAML file per experiment
data/raw/           datasets (not in Git; see "Data")
src/
  utils.py          GPU setup (respects Slurm, refuses CPU), seeding, config, logging
  data.py           loading, tokenization, word alignment, filtering, splits, padding
  model.py          mBERT (+ Bi-LSTM) tagger, first/mean pooling
  metrics.py        evaluation loop, weighted/macro F1, per-tag scores, confusion matrix
scripts/
  00_smoke_test.py  one forward/backward pass, sanity check
  01_verify_data.py checks the data against the paper's Tables 1 and 7
  03_train.py       training (all seeds) + final dev evaluation, saves results
  04_evaluate.py    error analysis; --subset cm|mono for code-mixed / monolingual scores
outputs/
  results/          metrics per run and seed (in Git)
  checkpoints/      model weights (not in Git)
  logs/             training logs (not in Git)
```

## Setup

```bash
conda activate <env>          # Python 3.11
pip install -r requirements.txt
```

## Data

Download `train-00000-of-00001.parquet` and `dev-00000-of-00001.parquet` from
`HuggMachas/Lince_benchmark_lid_hineng_train_valid` on HuggingFace into
`data/raw/lince_hineng/`, then verify:

```bash
python scripts/01_verify_data.py \
    --train data/raw/lince_hineng/train-00000-of-00001.parquet \
    --dev   data/raw/lince_hineng/dev-00000-of-00001.parquet
```

## Running

On a Slurm cluster, inside a GPU allocation:

```bash
python scripts/00_smoke_test.py --config configs/lince_hineng_mbert_esf1.yaml
python scripts/03_train.py      --config configs/lince_hineng_mbert_esf1.yaml
python scripts/03_train.py      --config configs/lince_hineng_mbert_esf1.yaml --seeds 42 --max-epochs 1   # quick test
```

Evaluation (no GPU needed):

```bash
python scripts/04_evaluate.py --experiments lince_hineng_mbert_bilstm_esf1
python scripts/04_evaluate.py --subset cm
python scripts/04_evaluate.py --subset mono
```

Results go to `outputs/results/<experiment_name>/` (`summary.json`, one `seed_<n>.json`
per seed, and per-sentence dev predictions).

## Notes

- Results are not bit-identical across GPUs or library versions; expect differences of a few
  hundredths of an F1 point, which is why every result is reported over 3 seeds.
- Paths in configs are relative to the repository root.