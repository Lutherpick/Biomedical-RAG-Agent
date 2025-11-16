#!/usr/bin/env python3
"""
Peek into representative chunks for a given label.

Reads:
  - paragraph_chunks_4000_labeled_ranked_merged.jsonl

Usage example:

  python src/pmc_chunker/out/4_5_labeling/peek_label_chunks.py ^
    --label "medical imaging" ^
    --k 10

This prints the top-k chunks (closest to centroid) for that label.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inspect representative chunks for a given label.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--chunks",
        type=Path,
        default=Path("src/pmc_chunker/out/4_5_labeling/paragraph_chunks_4000_labeled_ranked_merged.jsonl"),
        help="Path to labeled & ranked chunks JSONL",
    )
    p.add_argument(
        "--label",
        required=True,
        help="Cluster label to inspect (exact match after normalization)",
    )
    p.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of representative chunks to print",
    )
    return p.parse_args()


def truncate(text: str, max_len: int = 320) -> str:
    t = (text or "").replace("\n", " ").strip()
    if len(t) > max_len:
        return t[: max_len - 3] + "..."
    return t


def main() -> None:
    args = parse_args()
    if not args.chunks.exists():
        raise SystemExit(f"chunks file not found: {args.chunks}")

    df = pd.read_json(args.chunks, lines=True)

    needed = {"pmcid", "cluster_label", "text"}
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"Chunks missing columns: {missing}")

    # rank & distance are optional but used if present
    has_rank = "rank_in_cluster" in df.columns
    has_dist = "centroid_distance" in df.columns

    sub = df[df["cluster_label"] == args.label].copy()
    if sub.empty:
        print(f"No chunks found for label: {args.label!r}")
        return

    # sort: closest to centroid first if we have a distance; otherwise by rank if present
    if has_dist:
        sub = sub.sort_values("centroid_distance", ascending=True)
    elif has_rank:
        sub = sub.sort_values("rank_in_cluster", ascending=True)

    print("=" * 72)
    print(f"Representative chunks for label: {args.label!r}")
    print("=" * 72)
    print(f"Total chunks with this label: {len(sub)}")
    print()

    cols: List[str] = ["pmcid"]
    if has_rank:
        cols.append("rank_in_cluster")
    if has_dist:
        cols.append("centroid_distance")

    for idx, (_, row) in enumerate(sub.head(args.k).iterrows(), start=1):
        header_parts = [f"{c}={row[c]}" for c in cols]
        hdr = " | ".join(header_parts)

        print("-" * 72)
        print(f"#{idx}  {hdr}")
        print(truncate(str(row["text"])))
    print("-" * 72)


if __name__ == "__main__":
    main()
