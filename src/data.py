"""
src/data.py - data pipeline for word-level LID.

Flow:
  parquet -> examples {id, words, labels}
          -> encoded {input_ids, word_starts, labels}   (tokenise once)
          -> drop train sentences > max_subtokens      (train only, never dev)
          -> split train into train / val (early stopping)
          -> DataLoaders with dynamic padding

Model contract (what each batch contains):
  input_ids      (B, L)  subtoken ids, padded with the tokenizer's pad id
  attention_mask (B, L)  1 = real subtoken, 0 = padding
  word_starts    (B, W)  position in L of each word's FIRST subtoken
  word_index     (B, L)  word number of each subtoken, -1 for [CLS]/[SEP]/padding
                         (used for mean pooling over all subtokens of a word)
  word_mask      (B, W)  1 = real word, 0 = padding
  labels         (B, W)  word-level label ids, -100 on padding
The model gathers BERT outputs at word_starts, so the BiLSTM runs over WORDS
(as in Fig. 9 of the paper), not over subtokens.
"""
import random
from collections import Counter

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .utils import resolve_path

IGNORE_INDEX = -100  # CrossEntropyLoss ignores this label


# --------------------------------------------------------------------------- #
# 1. Loading
# --------------------------------------------------------------------------- #
def load_split(path, tok_col="sentence", lab_col="labels"):
    """Read one parquet split into a list of dicts."""
    df = pd.read_parquet(resolve_path(path))
    examples = []
    for ex_id, words, labels in zip(df["id"], df[tok_col], df[lab_col]):
        words, labels = list(words), list(labels)
        if len(words) != len(labels):
            raise ValueError(f"{ex_id}: {len(words)} words vs "
                             f"{len(labels)} labels")
        examples.append({"id": str(ex_id), "words": words, "labels": labels})
    return examples


def build_label_maps(train_examples, eval_examples=()):
    """label2id from TRAIN labels only, sorted so the ids are identical on
    every run and machine. A label that appears only in eval could never be
    predicted, so we fail loudly instead of silently mis-indexing."""
    train_tags = {t for ex in train_examples for t in ex["labels"]}
    eval_tags = {t for ex in eval_examples for t in ex["labels"]}
    unseen = eval_tags - train_tags
    if unseen:
        raise ValueError(f"labels in eval but not in train: {sorted(unseen)}")
    labels = sorted(train_tags)
    label2id = {t: i for i, t in enumerate(labels)}
    id2label = {i: t for t, i in label2id.items()}
    return label2id, id2label


def is_code_mixed(labels, lang_tags=("lang1", "lang2"), mixed_tag="mixed"):
    """Same definition as 01_verify_data.py (the paper gives none)."""
    s = set(labels)
    return set(lang_tags) <= s or mixed_tag in s


# --------------------------------------------------------------------------- #
# 2. Tokenisation and label alignment
# --------------------------------------------------------------------------- #
def encode_example(ex, tokenizer, label2id):
    """Tokenise one sentence and record where each word starts.

    Example (mBERT):
      words       : ["Virat", "so", "nicee"]
      subtokens   : [CLS] Vi ##rat so nice ##e [SEP]
      word_ids()  : None  0   0    1   2    2   None
      word_starts : [1, 3, 4]        <- first subtoken of each word
    Only the first subtoken of a word represents it; the other pieces still
    pass through BERT (they shape the context) but are never gathered.
    """
    # A word that tokenises to NOTHING (e.g. zero-width or control chars)
    # would get no position in word_ids() and shift every later label by
    # one. Replace such words with [UNK] so each word keeps a slot.
    words, n_replaced = [], 0
    for w in ex["words"]:
        if tokenizer.tokenize(w):
            words.append(w)
        else:
            words.append(tokenizer.unk_token)
            n_replaced += 1

    enc = tokenizer(words, is_split_into_words=True,
                    truncation=False, add_special_tokens=True)

    word_index = [-1 if w is None else w for w in enc.word_ids()]
    word_starts, prev = [], None
    for pos, wid in enumerate(enc.word_ids()):
        # word ids are contiguous, so a change of id = start of a new word
        if wid is not None and wid != prev:
            word_starts.append(pos)
        prev = wid

    # Hard guarantee: exactly one start per word, otherwise labels misalign
    assert len(word_starts) == len(words), (
        f"{ex['id']}: {len(word_starts)} starts for {len(words)} words")

    return {
        "id": ex["id"],
        "input_ids": enc["input_ids"],
        "word_starts": word_starts,
        "word_index": word_index,
        "labels": [label2id[t] for t in ex["labels"]],
        "n_replaced": n_replaced,
    }


# --------------------------------------------------------------------------- #
# 3. Filtering and splitting
# --------------------------------------------------------------------------- #
def drop_long(encoded, max_subtokens):
    """Remove sentences longer than the encoder limit (incl. [CLS]/[SEP]).
    Call on TRAIN only. For LinCE this removes ids 217 and 1902."""
    kept = [e for e in encoded if len(e["input_ids"]) <= max_subtokens]
    dropped = [e["id"] for e in encoded if len(e["input_ids"]) > max_subtokens]
    return kept, dropped


def split_train_val(encoded, val_fraction, split_seed):
    """Hold out part of train for early stopping, so the official dev set is
    used ONLY for final reporting (the paper appears to early-stop on dev).

    split_seed is separate from the training seed: every training seed then
    sees the same train/val split, so differences between seeds come from
    initialisation and shuffling only."""
    idx = list(range(len(encoded)))
    random.Random(split_seed).shuffle(idx)
    n_val = int(round(len(idx) * val_fraction))
    val = [encoded[i] for i in idx[:n_val]]
    train = [encoded[i] for i in idx[n_val:]]
    return train, val


# --------------------------------------------------------------------------- #
# 4. Dataset and dynamic padding
# --------------------------------------------------------------------------- #
class LIDDataset(Dataset):
    def __init__(self, encoded):
        self.items = encoded

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


class Collator:
    """Pads each batch only to ITS longest sentence (mean ~33 subtokens on
    LinCE) instead of 512, which makes training several times faster."""

    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        max_sub = max(len(b["input_ids"]) for b in batch)
        max_words = max(len(b["word_starts"]) for b in batch)

        input_ids, attn, starts, wmask, labels, ids = [], [], [], [], [], []
        windex = []
        for b in batch:
            n_sub, n_w = len(b["input_ids"]), len(b["word_starts"])
            input_ids.append(b["input_ids"] + [self.pad_id] * (max_sub - n_sub))
            attn.append([1] * n_sub + [0] * (max_sub - n_sub))
            # padded word slots point at position 0 ([CLS]); they are masked
            # out by word_mask and labelled -100, so they never affect loss
            starts.append(b["word_starts"] + [0] * (max_words - n_w))
            windex.append(b["word_index"] + [-1] * (max_sub - n_sub))
            wmask.append([1] * n_w + [0] * (max_words - n_w))
            labels.append(b["labels"] + [IGNORE_INDEX] * (max_words - n_w))
            ids.append(b["id"])

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "word_starts": torch.tensor(starts, dtype=torch.long),
            "word_index": torch.tensor(windex, dtype=torch.long),
            "word_mask": torch.tensor(wmask, dtype=torch.bool),
            "labels": torch.tensor(labels, dtype=torch.long),
            "ids": ids,  # kept for error analysis / inference outputs
        }


# --------------------------------------------------------------------------- #
# 5. Everything together
# --------------------------------------------------------------------------- #
def build_dataloaders(cfg, tokenizer, train_seed, logger=None):
    """Returns (loaders, label2id, id2label, stats).
    loaders = {"train", "val", "dev"}; stats is saved with the results."""
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer is required for word_ids().")

    log = logger.info if logger else print
    d = cfg["data"]

    train_raw = load_split(d["train_path"], d["token_col"], d["label_col"])
    dev_raw = load_split(d["dev_path"], d["token_col"], d["label_col"])

    # Optional CM-only setting: filter TRAIN only; dev stays the full set,
    # so ALL and CM runs are evaluated on the same data.
    if d.get("cm_only", False):
        before = len(train_raw)
        train_raw = [e for e in train_raw
                     if is_code_mixed(e["labels"], tuple(d["cm_lang_tags"]))]
        log(f"cm_only: kept {len(train_raw)}/{before} train sentences")

    label2id, id2label = build_label_maps(train_raw, dev_raw)

    train_enc = [encode_example(e, tokenizer, label2id) for e in train_raw]
    dev_enc = [encode_example(e, tokenizer, label2id) for e in dev_raw]

    train_enc, dropped = drop_long(train_enc, d["max_subtokens"])
    too_long_dev = [e["id"] for e in dev_enc
                    if len(e["input_ids"]) > d["max_subtokens"]]
    if too_long_dev:  # cannot drop eval data; must be handled explicitly
        raise ValueError(f"dev sentences over the limit: {too_long_dev}")

    train_enc, val_enc = split_train_val(train_enc, d["val_fraction"],
                                         d["split_seed"])

    stats = {
        "n_train": len(train_enc), "n_val": len(val_enc),
        "n_dev": len(dev_enc), "dropped_train_ids": dropped,
        "n_unk_replaced": sum(e["n_replaced"]
                              for e in train_enc + val_enc + dev_enc),
        "label2id": label2id,
        "train_label_counts": dict(Counter(
            id2label[l] for e in train_enc for l in e["labels"])),
    }
    log(f"train={stats['n_train']} val={stats['n_val']} dev={stats['n_dev']} "
        f"| dropped (>{d['max_subtokens']} subtokens): {dropped} "
        f"| words replaced by [UNK]: {stats['n_unk_replaced']}")
    log(f"labels: {label2id}")

    collate = Collator(tokenizer.pad_token_id)
    bs = cfg["train"]["batch_size"]
    # seeded generator -> the train shuffling order is reproducible per seed
    g = torch.Generator().manual_seed(train_seed)
    loaders = {
        "train": DataLoader(LIDDataset(train_enc), batch_size=bs,
                            shuffle=True, collate_fn=collate, generator=g),
        "val": DataLoader(LIDDataset(val_enc), batch_size=bs,
                          shuffle=False, collate_fn=collate),
        "dev": DataLoader(LIDDataset(dev_enc), batch_size=bs,
                          shuffle=False, collate_fn=collate),
    }
    return loaders, label2id, id2label, stats