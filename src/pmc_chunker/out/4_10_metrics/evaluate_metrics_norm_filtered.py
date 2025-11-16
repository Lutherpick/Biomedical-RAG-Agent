#!/usr/bin/env python3
"""
4.10 metrics – normalized labels, filtered by label frequency.

Keeps only labels that have at least MIN_QUERIES queries, then computes
precision@K and recall@K on that subset.

Inputs (by default, resolved relative to this file):
  - ../4_8_eval_setup/queries_eval_norm.jsonl
  - ../4_9_retrieval/retrieval_results.jsonl

Outputs (in this directory, or --out-dir):
  - metrics_report_norm_min5.json
  - evaluation_summary_norm_min5.csv
  - pipeline_log_norm_min5.json
"""

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


HERE = Path(__file__).resolve().parent
OUT_DEFAULT = HERE
K_DEFAULT = 5
MIN_QUERIES_DEFAULT = 5


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def detect_column(df: pd.DataFrame, candidates: List[str], name: str) -> str:
    """Pick the first column from candidates that exists in df."""
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find {name} column; tried: {candidates}")


def sha256_of(path: Path) -> str:
    """Compute SHA256 checksum of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_list(obj: Any) -> List[Any]:
    """Ensure obj is a Python list; decode JSON strings if needed."""
    if isinstance(obj, list):
        return obj
    if obj is None:
        return []
    if isinstance(obj, str):
        s = obj.strip()
        if not s:
            return []
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            return [s]
    return [obj]


# ---------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------


def add_per_query_metrics(
        df: pd.DataFrame,
        label_col: str,
        k: int,
) -> pd.DataFrame:
    """
    Given a merged queries+retrieval DataFrame, compute per-query
    hits_at_k, precision_at_k, and recall_at_k.

    Recall@k here is a hit-rate: 1 if at least one retrieved label
    matches, otherwise 0.
    """

    def _row(row: pd.Series) -> Dict[str, Any]:
        gold = row[label_col]
        retrieved = ensure_list(row["retrieved_labels"])[:k]
        hits = sum(1 for lbl in retrieved if lbl == gold)
        precision = float(hits) / float(k) if k > 0 else 0.0
        recall = 1.0 if hits > 0 else 0.0
        return {
            "hits_at_k": hits,
            "precision_at_k": precision,
            "recall_at_k": recall,
        }

    m = df.apply(_row, axis=1, result_type="expand")
    df_out = df.copy()
    for col in ["hits_at_k", "precision_at_k", "recall_at_k"]:
        if col in df_out.columns:
            del df_out[col]
    df_out = pd.concat([df_out.reset_index(drop=True), m.reset_index(drop=True)], axis=1)
    return df_out


def aggregate_metrics(
        df: pd.DataFrame,
        label_col: str,
        k: int,
) -> Dict[str, Any]:
    """Aggregate macro + per-label metrics from a dataframe that already
    has precision_at_k and recall_at_k columns."""

    macro_precision = float(df["precision_at_k"].mean())
    macro_recall = float(df["recall_at_k"].mean())

    per_label_df = (
        df.groupby(label_col)
        .agg(
            n_queries=("query_id", "count"),
            precision_at_k=("precision_at_k", "mean"),
            recall_at_k=("recall_at_k", "mean"),
        )
        .reset_index()
        .rename(columns={label_col: "label"})
        .sort_values("n_queries", ascending=False)
    )

    per_label_df["precision_at_k"] = per_label_df["precision_at_k"].astype(float)
    per_label_df["recall_at_k"] = per_label_df["recall_at_k"].astype(float)

    metrics = {
        "k": int(k),
        "n_queries": int(len(df)),
        "macro_precision_at_k": macro_precision,
        "macro_recall_at_k": macro_recall,
        "per_label": [
            {
                "label": str(row["label"]),
                "n_queries": int(row["n_queries"]),
                "precision_at_k": float(row["precision_at_k"]),
                "recall_at_k": float(row["recall_at_k"]),
            }
            for _, row in per_label_df.iterrows()
        ],
    }

    return metrics, per_label_df


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="4.10 metrics – normalized labels, filtered by min_queries."
    )
    p.add_argument(
        "--queries",
        type=Path,
        default=None,
        help="Path to queries_eval_norm.jsonl (normalized). "
             "If omitted, resolve relative to this file.",
    )
    p.add_argument(
        "--retrieval",
        type=Path,
        default=None,
        help="Path to retrieval_results.jsonl. "
             "If omitted, resolve relative to this file.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: this script's directory).",
    )
    p.add_argument(
        "--k",
        type=int,
        default=K_DEFAULT,
        help=f"Rank cutoff K for precision@K / recall@K (default: {K_DEFAULT}).",
    )
    p.add_argument(
        "--min-queries",
        type=int,
        default=MIN_QUERIES_DEFAULT,
        help=f"Minimum #queries per label to keep (default: {MIN_QUERIES_DEFAULT}).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.out_dir is None:
        args.out_dir = OUT_DEFAULT
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.queries is None:
        args.queries = HERE.parent / "4_8_eval_setup" / "queries_eval_norm.jsonl"
    if args.retrieval is None:
        args.retrieval = HERE.parent / "4_9_retrieval" / "retrieval_results.jsonl"

    if not args.queries.is_file():
        raise FileNotFoundError(f"queries file not found: {args.queries}")
    if not args.retrieval.is_file():
        raise FileNotFoundError(f"retrieval file not found: {args.retrieval}")

    # ---- Load inputs ---------------------------------------------------
    queries_df = pd.read_json(args.queries, lines=True)
    retrieval_df = pd.read_json(args.retrieval, lines=True)

    if queries_df.empty or retrieval_df.empty:
        raise RuntimeError("Input files are empty.")

    label_col = detect_column(
        queries_df,
        ["label", "cluster_label", "target_label", "normalized_label"],
        "label",
    )

    if "query_id" not in queries_df.columns or "query_id" not in retrieval_df.columns:
        raise ValueError("Both queries and retrieval files must contain 'query_id'.")

    merged = queries_df.merge(
        retrieval_df,
        on="query_id",
        how="inner",
        suffixes=("_q", "_r"),
    )
    if merged.empty:
        raise RuntimeError("No overlapping query_id between queries and retrieval results.")

    # ---- First pass: add per-query metrics on ALL queries --------------
    df_all = add_per_query_metrics(merged, label_col=label_col, k=args.k)
    metrics_all, per_label_all = aggregate_metrics(df_all, label_col=label_col, k=args.k)

    # ---- Filter labels by frequency ------------------------------------
    labels_keep = per_label_all.loc[
        per_label_all["n_queries"] >= int(args["min_queries"])
        if isinstance(args, dict)
        else per_label_all["n_queries"] >= int(args.min_queries),
        "label",
    ]
    labels_keep = set(labels_keep.tolist())

    df_filt = df_all[df_all[label_col].isin(labels_keep)].copy()
    if df_filt.empty:
        raise RuntimeError(
            f"No labels have >= {args.min_queries} queries; nothing to evaluate."
        )

    metrics_filt, per_label_filt = aggregate_metrics(
        df_filt, label_col=label_col, k=args.k
    )

    # ---- Write metrics_report_norm_min5.json ---------------------------
    metrics_report = {
        "min_queries": int(args.min_queries),
        "retrieval_all": metrics_all,
        "retrieval_filtered": metrics_filt,
    }

    metrics_path = args.out_dir / "metrics_report_norm_min5.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_report, f, indent=2, ensure_ascii=False)

    # ---- Write evaluation_summary_norm_min5.csv ------------------------
    summary_df = per_label_filt.copy()
    summary_df["is_macro_row"] = False

    macro_row = pd.DataFrame(
        [
            {
                "label": "__ALL__",
                "n_queries": metrics_filt["n_queries"],
                "precision_at_k": metrics_filt["macro_precision_at_k"],
                "recall_at_k": metrics_filt["macro_recall_at_k"],
                "is_macro_row": True,
            }
        ]
    )
    summary_df = pd.concat([summary_df, macro_row], ignore_index=True)

    summary_path = args.out_dir / "evaluation_summary_norm_min5.csv"
    summary_df.to_csv(summary_path, index=False)

    # ---- Write pipeline_log_norm_min5.json -----------------------------
    pipeline_log = {
        "run_at": dt.datetime.now().isoformat(),
        "k": int(args.k),
        "min_queries": int(args.min_queries),
        "paths": {
            "queries_eval_norm": str(args.queries),
            "retrieval_results": str(args.retrieval),
            "metrics_report_norm_min5": str(metrics_path),
            "evaluation_summary_norm_min5": str(summary_path),
        },
        "input_checksums": {
            "queries_eval_norm_sha256": sha256_of(args.queries),
            "retrieval_results_sha256": sha256_of(args.retrieval),
        },
        "software": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "pandas_version": pd.__version__,
        },
        "env": {"cwd": os.getcwd()},
        "n_queries_all": int(metrics_all["n_queries"]),
        "n_queries_filtered": int(metrics_filt["n_queries"]),
    }

    log_path = args.out_dir / "pipeline_log_norm_min5.json"
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(pipeline_log, f, indent=2, ensure_ascii=False)

    # ---- Console summary -----------------------------------------------
    print(
        f"[INFO] All topics      : n_queries={metrics_all['n_queries']}, "
        f"P@{args.k}={metrics_all['macro_precision_at_k']:.3f}, "
        f"R@{args.k}={metrics_all['macro_recall_at_k']:.3f}"
    )
    print(
        f"[INFO] Frequent topics : n_queries={metrics_filt['n_queries']} "
        f"(min_queries={args.min_queries}), "
        f"P@{args.k}={metrics_filt['macro_precision_at_k']:.3f}, "
        f"R@{args.k}={metrics_filt['macro_recall_at_k']:.3f}"
    )
    print(f"[INFO] metrics_report_norm_min5.json    -> {metrics_path}")
    print(f"[INFO] evaluation_summary_norm_min5.csv -> {summary_path}")
    print(f"[INFO] pipeline_log_norm_min5.json      -> {log_path}")


if __name__ == "__main__":
    main()
