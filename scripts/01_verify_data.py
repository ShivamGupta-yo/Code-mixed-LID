"""
Step 1: Verify the LinCE HI-EN LID data against the paper (Chanda & Pal 2025).
  - Table 1: sentence / token / monolingual / code-mixed counts
  - Table 7: dev-set tag distribution
  - Token-label alignment, empty rows
  - mBERT subtoken lengths vs the 512 limit

Usage:
  python verify_lince.py --train train-00000-of-00001.parquet \
                         --dev   dev-00000-of-00001.parquet
  (add --no-tokenizer to skip the subtoken-length check)
"""
import argparse
from collections import Counter

import pandas as pd

# ---- Expected values from the paper ----
EXPECTED = {
    "train": {"sentences": 4823, "tokens": 95224, "mono": 2702, "cm": 2121},
    "dev":   {"sentences": 744,  "tokens": 15446, "mono": 419,  "cm": 325},
}
EXPECTED_DEV_TAGS = {  # Table 7
    "lang1": 8997, "lang2": 3306, "other": 2230, "ne": 875,
    "fw": 29, "unk": 2, "mixed": 5, "ambiguous": 2,
}
LANG_TAGS = {"lang1", "lang2"}


def check(name, got, exp):
    status = "OK " if got == exp else "MISMATCH"
    print(f"  [{status}] {name:<12} got={got:<8} expected={exp}")


def is_code_mixed(labels):
    """Code-mixed = both lang1 and lang2 present, or any 'mixed' token.
    The paper does not define this; we report it so the definition can be
    checked against Table 1."""
    s = set(labels)
    return (LANG_TAGS <= s) or ("mixed" in s)


def verify_split(split, df, tok_col, lab_col):
    print(f"\n===== {split.upper()} =====")
    exp = EXPECTED[split]

    # Alignment and empties
    bad = df[df[tok_col].apply(len) != df[lab_col].apply(len)]
    empty = df[df[tok_col].apply(len) == 0]
    print(f"  token/label length mismatches: {len(bad)}")
    print(f"  empty sentences:               {len(empty)}")

    n_sent = len(df)
    n_tok = int(df[tok_col].apply(len).sum())
    cm_mask = df[lab_col].apply(lambda x: is_code_mixed(list(x)))
    n_cm = int(cm_mask.sum())
    n_mono = n_sent - n_cm

    check("sentences", n_sent, exp["sentences"])
    check("tokens", n_tok, exp["tokens"])
    check("monolingual", n_mono, exp["mono"])
    check("code-mixed", n_cm, exp["cm"])

    tags = Counter(t for labs in df[lab_col] for t in labs)
    print("  tag distribution:")
    for t, c in tags.most_common():
        print(f"    {t:<10} {c:>7}  ({100 * c / n_tok:.2f}%)")

    if split == "dev":
        print("  Table 7 comparison:")
        for t, e in EXPECTED_DEV_TAGS.items():
            check(t, tags.get(t, 0), e)
        extra = set(tags) - set(EXPECTED_DEV_TAGS)
        if extra:
            print(f"  tags not in Table 7: {sorted(extra)}")

    lens = df[tok_col].apply(len)
    print(f"  words/sentence: min={lens.min()} mean={lens.mean():.1f} "
          f"max={lens.max()}")
    return tags


def subtoken_check(split, df, tok_col, tokenizer):
    lens = []
    for words in df[tok_col]:
        enc = tokenizer(list(words), is_split_into_words=True,
                        add_special_tokens=True, truncation=False)
        lens.append(len(enc["input_ids"]))
    s = pd.Series(lens)
    over = int((s > 512).sum())
    print(f"  [{split}] mBERT subtokens: mean={s.mean():.1f} "
          f"p99={s.quantile(0.99):.0f} max={s.max()} | >512: {over}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--tok-col", default="sentence")
    ap.add_argument("--lab-col", default="labels")
    ap.add_argument("--no-tokenizer", action="store_true")
    args = ap.parse_args()

    dfs = {"train": pd.read_parquet(args.train),
           "dev": pd.read_parquet(args.dev)}
    for split, df in dfs.items():
        print(f"{split}: columns={list(df.columns)}")

    tag_sets = {}
    for split, df in dfs.items():
        tag_sets[split] = verify_split(split, df, args.tok_col, args.lab_col)

    unseen = set(tag_sets["dev"]) - set(tag_sets["train"])
    print(f"\nTags in dev but not in train: {sorted(unseen) or 'none'}")

    if not args.no_tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        print("\n===== SUBTOKEN LENGTHS (mBERT-base-cased) =====")
        for split, df in dfs.items():
            subtoken_check(split, df, args.tok_col, tok)


if __name__ == "__main__":
    main()