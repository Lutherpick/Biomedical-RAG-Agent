#!/usr/bin/env python3
"""
4.7 Deduplication – query-level TF-IDF cosine dedup

Goal
- Ensure a diverse, non-redundant query set.
- Keep at least one query per cluster_id.

Method
1) Vectorize all query_text strings via TF-IDF.
2) Compute cosine similarities between all queries.
3) Greedily mark later queries as duplicates if similarity >= threshold.
4) Guarantee at least one query per cluster_id remains (if that column exists).

This version intentionally avoids transformers / sentence-transformers to
sidestep version conflicts. It only requires scikit-learn + numpy + pandas.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True, help="input queries JSONL")
    ap.add_argument("--out", required=True, help="output deduplicated JSONL")

    # Kept for CLI compatibility but not used (we no longer call HF models here).
    ap.add_argument(
        "--st-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="unused (for backwards compatibility)",
    )

    ap.add_argument(
        "--threshold",
        type=float,
        default=0.90,
        help="cosine similarity threshold above which two queries are treated as duplicates",
    )
    args = ap.parse_args()

    inp_path = Path(args.inp)
    if not inp_path.is_file():
        raise FileNotFoundError(f"Input file not found: {inp_path}")

    # ---- Load queries -------------------------------------------------
    with inp_path.open("r", encoding="utf-8") as f:
        df = pd.read_json(f, lines=True)

    if "query_text" not in df.columns:
        raise ValueError("Input JSONL must contain a 'query_text' column")

    texts = df["query_text"].astype(str).tolist()
    n = len(texts)
    if n == 0:
        raise ValueError("No queries found in input")

    # ---- TF-IDF vectorization ----------------------------------------
    # L2 normalization is on by default for TfidfVectorizer, so
    # X * X.T gives cosine similarity.
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=8192,
        lowercase=True,
    )
    X = vec.fit_transform(texts)          # shape: (n_queries, n_features)

    # Dense similarity matrix; for ~600 queries this is tiny (600x600).
    S = (X * X.T).toarray().astype("float32")
    np.fill_diagonal(S, -1.0)

    keep = np.ones(n, dtype=bool)

    # ---- Greedy dedup: keep earlier queries, drop later near-duplicates
    for i in range(n):
        if not keep[i]:
            continue
        # indices j where similarity >= threshold
        sim_idx = np.where(S[i] >= args.threshold)[0]
        for j in sim_idx:
            if j > i:
                keep[j] = False
        if sim_idx.size:
            S[i, sim_idx] = -1.0

    kept = df[keep].copy()

    # ---- Ensure ≥1 query per cluster_id (if present) ------------------
    if "cluster_id" in df.columns:
        groups = df.groupby("cluster_id").groups
        present = set(kept["cluster_id"].unique().tolist())
        for cid, idxs in groups.items():
            if cid not in present:
                first_idx = list(idxs)[0]
                kept = pd.concat([kept, df.iloc[[first_idx]]], ignore_index=True)
                present.add(cid)

    # ---- Write JSONL output -------------------------------------------
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for _, row in kept.iterrows():
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")

    removed = int((~keep).sum())
    print(
        f"kept: {len(kept)}  removed (pre-restore): {removed}  "
        f"threshold: {args.threshold}"
    )


if __name__ == "__main__":
    main()
