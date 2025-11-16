#!/usr/bin/env python
"""
4.8 Query–Label Alignment and Evaluation Setup

Inputs
------
- 4_7_deduplication/queries_gpt51_dedup.jsonl
    JSONL with at least: query_id (optional), query_text, cluster_id, label, source_chunk_id (optional)

- 4_5_labeling/cluster_labels.json
    { "0": "cluster label 0", "1": "cluster label 1", ... }

Outputs (written to 4_8_eval_setup/)
-------
- queries_eval.jsonl
    Per-query evaluation record with canonical cluster label.

- retrieval_config.json
    Configuration for retrieval (Qdrant collection + embedding model etc.).
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import pandas as pd


def infer_default_paths(this_file: Path) -> Dict[str, Path]:
    """
    Infer defaults relative to this script location.

    this_file = .../src/pmc_chunker/out/4_8_eval_setup/eval_setup.py
    out_root  = .../src/pmc_chunker/out
    """
    out_dir = this_file.parent
    out_root = out_dir.parent

    return {
        "out_dir": out_dir,
        "queries": out_root / "4_7_deduplication" / "queries_gpt51_dedup.jsonl",
        "labels": out_root / "4_5_labeling" / "cluster_labels.json",
    }


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve()
    defaults = infer_default_paths(here)

    p = argparse.ArgumentParser(
        description="4.8 Query–Label Alignment and Evaluation Setup",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--queries",
        type=Path,
        default=defaults["queries"],
        help="Deduplicated queries JSONL (output of 4.7).",
    )
    p.add_argument(
        "--labels",
        type=Path,
        default=defaults["labels"],
        help="Cluster labels JSON (output of 4.5).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=defaults["out_dir"],
        help="Directory to write queries_eval.jsonl and retrieval_config.json.",
    )
    p.add_argument(
        "--collection",
        type=str,
        default="biomed_paragraphs",
        help="Name of the Qdrant collection containing paragraph chunks.",
    )
    p.add_argument(
        "--embedding-model",
        type=str,
        default="text-embedding-3-small",
        help="Embedding model used for both chunks and queries.",
    )
    p.add_argument(
        "--embedding-model-version",
        type=str,
        default="v0.2",
        help="Version tag for the embedding model.",
    )
    p.add_argument(
        "--index-type",
        type=str,
        default="HNSW",
        help="Index type used in Qdrant.",
    )
    p.add_argument(
        "--distance",
        type=str,
        default="cosine",
        help="Distance metric used in Qdrant.",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Default top-k to retrieve in evaluation.",
    )

    return p.parse_args()


def load_cluster_labels(path: Path) -> Dict[int, str]:
    with path.open("r", encoding="utf8") as f:
        data = json.load(f)

    labels: Dict[int, str] = {}
    for k, v in data.items():
        try:
            cid = int(k)
        except (TypeError, ValueError):
            continue
        labels[cid] = str(v or "").strip()
    return labels


def canonicalize_labels(
        df: pd.DataFrame, cluster_labels: Dict[int, str]
) -> pd.Series:
    """
    For each row, prefer the canonical label from cluster_labels; if missing,
    fall back to the label already present in the queries file.
    """
    def _canon(row: pd.Series) -> str:
        cid = int(row["cluster_id"])
        canon = cluster_labels.get(cid, "")
        if canon:
            return canon
        existing = str(row.get("label", "") or "").strip()
        return existing

    return df.apply(_canon, axis=1)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load inputs ---
    if not args.queries.exists():
        raise SystemExit(f"Queries file not found: {args.queries}")

    if not args.labels.exists():
        raise SystemExit(f"Cluster labels file not found: {args.labels}")

    df = pd.read_json(args.queries, lines=True)

    required_cols = {"query_text", "cluster_id"}
    missing = required_cols - set(df.columns)
    if missing:
        raise SystemExit(f"Queries file is missing required columns: {missing}")

    # Ensure integer cluster ids
    df["cluster_id"] = df["cluster_id"].astype(int)

    # Ensure unique / present query_ids
    if "query_id" not in df.columns:
        df = df.copy()
        df["query_id"] = [f"q_{i:06d}" for i in range(len(df))]
    else:
        # If there are duplicates, rewrite them to be unique
        if df["query_id"].nunique() != len(df):
            df = df.copy()
            df["query_id"] = [f"q_{i:06d}" for i in range(len(df))]

    # --- Align labels ---
    cluster_labels = load_cluster_labels(args.labels)
    df["cluster_label"] = canonicalize_labels(df, cluster_labels)

    # Optional sanity check: how many label mismatches vs existing "label" column?
    if "label" in df.columns:
        existing = df["label"].astype(str).str.strip().str.lower()
        canon = df["cluster_label"].astype(str).str.strip().str.lower()
        mismatches = (existing != "") & (existing != canon)
        n_mismatch = int(mismatches.sum())
        if n_mismatch:
            print(f"[info] label mismatches (queries vs canonical): {n_mismatch}")

    # --- Build evaluation dataset ---
    base_cols = ["query_id", "query_text", "cluster_id", "cluster_label"]
    optional_cols = [c for c in ("source_chunk_id", "label") if c in df.columns]
    cols = [c for c in base_cols + optional_cols if c in df.columns]
    eval_df = df[cols].copy()

    queries_path = args.out_dir / "queries_eval.jsonl"
    eval_df.to_json(
        queries_path,
        orient="records",
        lines=True,
        force_ascii=False,
    )

    # --- Write retrieval configuration ---
    cfg: Dict[str, Any] = {
        "collection": args.collection,
        "embedding_model": args.embedding_model,
        "embedding_model_version": args.embedding_model_version,
        "index_type": args.index_type,
        "distance": args.distance,
        "top_k": int(args.top_k),
        "n_queries": int(len(eval_df)),
        "n_clusters": int(eval_df["cluster_id"].nunique()),
    }

    cfg_path = args.out_dir / "retrieval_config.json"
    with cfg_path.open("w", encoding="utf8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    print(f"[4.8] Wrote {len(eval_df)} queries to {queries_path}")
    print(f"[4.8] Wrote retrieval config to {cfg_path}")
    print("[4.8] Done.")


if __name__ == "__main__":
    main()
