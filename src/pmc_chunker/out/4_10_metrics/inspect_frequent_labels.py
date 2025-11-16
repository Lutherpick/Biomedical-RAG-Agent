#!/usr/bin/env python3
r"""
Utility to quickly inspect "all topics vs frequent topics" metrics
for 4.10, using the outputs of:

  - evaluate_metrics_norm.py
      -> metrics_report_norm.json
      -> evaluation_summary_norm.csv

  - evaluate_metrics_norm_filtered.py
      -> metrics_report_norm_min5.json
      -> evaluation_summary_norm_min5.csv

Example:

  python .\src\pmc_chunker\out\4_10_metrics\inspect_frequent_labels.py
"""

import argparse
import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare metrics: all topics vs topics with >= min_queries."
    )
    p.add_argument(
        "--metrics-all",
        type=Path,
        default=None,
        help="metrics_report_norm.json (default: resolve in this directory).",
    )
    p.add_argument(
        "--metrics-freq",
        type=Path,
        default=None,
        help="metrics_report_norm_min5.json (default: resolve in this directory).",
    )
    p.add_argument(
        "--summary-all",
        type=Path,
        default=None,
        help="evaluation_summary_norm.csv (default: resolve in this directory).",
    )
    p.add_argument(
        "--summary-freq",
        type=Path,
        default=None,
        help="evaluation_summary_norm_min5.csv (default: resolve in this directory).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.metrics_all is None:
        args.metrics_all = HERE / "metrics_report_norm.json"
    if args.metrics_freq is None:
        args.metrics_freq = HERE / "metrics_report_norm_min5.json"
    if args.summary_all is None:
        args.summary_all = HERE / "evaluation_summary_norm.csv"
    if args.summary_freq is None:
        args.summary_freq = HERE / "evaluation_summary_norm_min5.csv"

    with args.metrics_all.open("r", encoding="utf-8") as f:
        m_all = json.load(f)
    with args.metrics_freq.open("r", encoding="utf-8") as f:
        m_freq = json.load(f)

    r_all = m_all["retrieval"]
    r_freq = m_freq["retrieval"]

    print("=== Macro metrics (K = {k}) ===".format(k=r_all["k"]))
    print(
        f"All topics       : P@{r_all['k']:.0f} = {r_all['macro_precision_at_k']:.3f}, "
        f"R@{r_all['k']:.0f} = {r_all['macro_recall_at_k']:.3f} "
        f"(n_queries = {r_all['n_queries']})"
    )
    print(
        f"Frequent topics  : P@{r_freq['k']:.0f} = {r_freq['macro_precision_at_k']:.3f}, "
        f"R@{r_freq['k']:.0f} = {r_freq['macro_recall_at_k']:.3f} "
        f"(n_queries = {r_freq['n_queries']})"
    )
    print()

    # Optional: show top-N labels table side-by-side
    df_all = pd.read_csv(args.summary_all)
    df_freq = pd.read_csv(args.summary_freq)

    df_all = df_all[~df_all["is_macro_row"]].copy()
    df_freq = df_freq[~df_freq["is_macro_row"]].copy()

    df_all = df_all.sort_values("n_queries", ascending=False)
    df_freq = df_freq.sort_values("n_queries", ascending=False)

    print("=== Top 10 labels (all topics) ===")
    print(
        df_all.head(10)[
            ["label", "n_queries", "precision_at_k", "recall_at_k"]
        ].to_string(index=False)
    )
    print()
    print("=== Top 10 labels (frequent topics only) ===")
    print(
        df_freq.head(10)[
            ["label", "n_queries", "precision_at_k", "recall_at_k"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
