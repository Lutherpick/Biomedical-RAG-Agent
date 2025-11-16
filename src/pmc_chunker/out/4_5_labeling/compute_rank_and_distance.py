#!/usr/bin/env python3
"""
Compute centroid_distance and rank_in_cluster for each paragraph chunk and
augment paragraph_chunks_4000_labeled.jsonl with these fields.

We DO NOT use centroids_k613.npy (which is in 100-dim PCA space).
Instead, we recompute centroids directly in the original 384-dim
embedding space using the cluster assignments.

Inputs (defaults assume current layout):

  --chunks   src/pmc_chunker/out/4_5_labeling/paragraph_chunks_4000_labeled.jsonl
  --assign   src/pmc_chunker/out/4_4_clustering/cluster_assignments_k613.csv
  --emb      src/pmc_chunker/out/embeddings_v5.npy   # (N x 384)

Output:

  src/pmc_chunker/out/4_5_labeling/paragraph_chunks_4000_labeled_ranked.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    base = Path("src/pmc_chunker/out")

    p = argparse.ArgumentParser(
        description="Compute centroid_distance and rank_in_cluster for each chunk.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--chunks",
        type=Path,
        default=base / "4_5_labeling" / "paragraph_chunks_4000_labeled.jsonl",
        help="Labeled chunks JSONL.",
    )
    p.add_argument(
        "--assign",
        type=Path,
        default=base / "4_4_clustering" / "cluster_assignments_k613.csv",
        help="Cluster assignments CSV (one row per chunk).",
    )
    p.add_argument(
        "--emb",
        type=Path,
        default=base / "embeddings_v5.npy",
        help="Embeddings used for clustering (N x D, MiniLM all-MiniLM-L6-v2).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=base / "4_5_labeling"
                / "paragraph_chunks_4000_labeled_ranked.jsonl",
        help="Output JSONL with rank_in_cluster and centroid_distance.",
    )
    return p.parse_args()


def load_embeddings(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    return arr


def normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def main() -> None:
    args = parse_args()

    print(f"[rank] Loading labeled chunks from {args.chunks}")
    df_chunks = pd.read_json(args.chunks, lines=True)

    for col in ("pmcid", "chunk_index"):
        if col not in df_chunks.columns:
            raise SystemExit(f"[rank] Column '{col}' missing in chunks file.")
    df_chunks["pmcid"] = df_chunks["pmcid"].astype(str)
    df_chunks["chunk_index"] = df_chunks["chunk_index"].astype(int)

    print(f"[rank] Loaded {len(df_chunks)} labeled chunks")

    print(f"[rank] Loading cluster assignments from {args.assign}")
    df_assign = pd.read_csv(args.assign, dtype=str)

    if "cluster_id" not in df_assign.columns:
        raise SystemExit("[rank] 'cluster_id' column missing in assignments CSV")
    if "pmcid" not in df_assign.columns:
        raise SystemExit("[rank] 'pmcid' column missing in assignments CSV")
    if "chunk_index" not in df_assign.columns:
        raise SystemExit("[rank] 'chunk_index' column missing in assignments CSV")

    df_assign["pmcid"] = df_assign["pmcid"].astype(str)
    df_assign["chunk_index"] = (
        df_assign["chunk_index"]
        .astype(str)
        .str.extract(r"(\d+)", expand=False)
        .astype(int)
    )
    df_assign["cluster_id"] = df_assign["cluster_id"].astype(int)

    print(f"[rank] Loaded {len(df_assign)} assignment rows")

    print(f"[rank] Loading embeddings from {args.emb}")
    emb = load_embeddings(args.emb)
    print(f"[rank] Embeddings shape: {emb.shape}")

    if emb.shape[0] != len(df_assign):
        raise SystemExit(
            f"[rank] Row mismatch: {len(df_assign)} assignments vs {emb.shape[0]} embeddings"
        )

    # Normalize embeddings once
    emb_norm = normalize_rows(emb)

    cluster_ids = df_assign["cluster_id"].to_numpy()
    k_min, k_max = int(cluster_ids.min()), int(cluster_ids.max())
    n_clusters = k_max + 1
    print(
        f"[rank] Cluster id range: [{k_min}, {k_max}] → assuming {n_clusters} clusters"
    )
    if k_min < 0:
        raise SystemExit("[rank] Negative cluster_id found, aborting.")

    # Compute centroids in 384-d embedding space
    dim = emb_norm.shape[1]
    centroids = np.zeros((n_clusters, dim), dtype=np.float32)
    counts = np.zeros(n_clusters, dtype=np.int64)

    print("[rank] Computing centroids in 384-d space...")
    for cid in range(n_clusters):
        mask = cluster_ids == cid
        if not np.any(mask):
            continue
        centroids[cid] = emb_norm[mask].mean(axis=0)
        counts[cid] = mask.sum()

    centroids = normalize_rows(centroids)

    # Distance to own-cluster centroid
    print("[rank] Computing centroid distances...")
    cent_for_rows = centroids[cluster_ids]  # (N, D)
    cos_sim = np.sum(emb_norm * cent_for_rows, axis=1)
    centroid_distance = 1.0 - cos_sim

    df_extra = df_assign[["pmcid", "chunk_index", "cluster_id"]].copy()
    df_extra["centroid_distance"] = centroid_distance

    print("[rank] Ranking within each cluster (0 = closest to centroid)")
    df_extra["rank_in_cluster"] = (
            df_extra.groupby("cluster_id")["centroid_distance"]
            .rank(method="first")
            .astype(int)
            - 1
    )

    # Merge into chunks
    print("[rank] Merging rank/distance into labeled chunks...")
    df_merged = df_chunks.merge(
        df_extra[["pmcid", "chunk_index", "centroid_distance", "rank_in_cluster"]],
        on=["pmcid", "chunk_index"],
        how="left",
    )

    covered = df_merged["centroid_distance"].notna().sum()
    print(
        f"[rank] Coverage: {covered} / {len(df_merged)} chunks "
        f"({covered / max(1, len(df_merged)):.1%}) got rank/distance"
    )

    print(f"[rank] Writing output JSONL to {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_json(args.out, orient="records", lines=True, force_ascii=False)

    print("[rank] Done.")


if __name__ == "__main__":
    main()
