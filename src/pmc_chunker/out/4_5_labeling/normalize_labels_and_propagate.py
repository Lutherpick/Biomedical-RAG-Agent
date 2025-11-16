#!/usr/bin/env python3
"""
Normalize cluster labels and propagate them to all downstream files.

Reads:
  - 4_5_labeling/cluster_labels.json
  - 4_5_labeling/paragraph_chunks_4000_labeled_ranked.jsonl
  - 4_6_query_generation/queries_gpt51.jsonl
  - 4_7_deduplication/queries_gpt51_dedup.jsonl
  - 4_8_eval_setup/queries_eval.jsonl

Writes:
  - 4_5_labeling/cluster_labels_merged.json
  - 4_5_labeling/paragraph_chunks_4000_labeled_ranked_merged.jsonl
  - 4_6_query_generation/queries_gpt51_norm.jsonl
  - 4_7_deduplication/queries_gpt51_dedup_norm.jsonl
  - 4_8_eval_setup/queries_eval_norm.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd


# ---------------------------------------------------------------------
# Manual normalization rules
# ---------------------------------------------------------------------

_MANUAL_LOWER_MAP: Dict[str, str] = {
    # obviously broken / missing
    "missing": "unknown",
    "t": "unknown",
    "p": "unknown",

    # covid / sars family
    "covid19": "COVID-19",
    "covid-19": "COVID-19",
    "covid 19": "COVID-19",
    "covid19 lockdown": "COVID-19",
    "covid19 pandemic": "COVID-19",
    "covid19 vaccination": "COVID-19 vaccination",
    "covid-19 vaccination": "COVID-19 vaccination",
    "sars-cov-2": "COVID-19",
    "sars cov": "COVID-19",
    "sars-cov-2 diagnostics": "COVID-19 diagnostics",

    # machine learning / AI
    "machine learning": "machine learning",
    "deep learning": "machine learning",
    "artificial intelligence": "machine learning",
    "pattern recognition": "machine learning",

    # imaging – expanded group
    "medical imaging": "medical imaging",
    "brain imaging": "medical imaging",
    "microscopy imaging": "medical imaging",
    "cancer imaging": "medical imaging",
    "neuroimaging": "medical imaging",
    "brain mri": "medical imaging",
    "muscle mri": "medical imaging",
    "structural mri": "medical imaging",

    # statistical methods – expanded group
    "statistical": "statistical methods",
    "statistical modeling": "statistical methods",
    "clinical statistics": "statistical methods",
    "analytical": "statistical methods",
    "methods": "statistical methods",
    "models": "statistical methods",
    "measurements": "statistical methods",

    # inflammation / immune group – expanded
    "inflammation": "inflammation",
    "lung inflammation": "inflammation",
    "microglial inflammation": "inflammation",
    "immune": "inflammation",
    "immune regulation": "inflammation",
    "immune responses": "inflammation",
    "autoimmune diseases": "inflammation",
}


def normalize_label(raw: str) -> str:
    """
    Normalize a raw label string to a canonical form.

    Strategy:
      - empty/None -> 'unknown'
      - lower-case, strip
      - direct lookup in _MANUAL_LOWER_MAP
      - fallback strip plural 's' and lookup again
      - otherwise: return original (with original casing)
    """
    if raw is None:
        return "unknown"
    s = str(raw).strip()
    if not s:
        return "unknown"

    key = s.lower()

    # direct mapping
    if key in _MANUAL_LOWER_MAP:
        return _MANUAL_LOWER_MAP[key]

    # simple singularization heuristic: drop trailing 's'
    if key.endswith("s"):
        base = key[:-1]
        if base in _MANUAL_LOWER_MAP:
            return _MANUAL_LOWER_MAP[base]

    # default: keep original string (not lower-cased) to preserve readability
    return s


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def normalize_cluster_labels(
        src_path: Path, dst_path: Path
) -> Dict[str, str]:
    """
    Normalize the labels in cluster_labels.json and save as
    cluster_labels_merged.json.
    """
    print(f"[labels] loading from: {src_path}")
    if not src_path.exists():
        raise SystemExit(f"cluster_labels.json not found: {src_path}")

    raw = json.loads(src_path.read_text(encoding="utf-8"))
    merged: Dict[str, str] = {}
    raw_labels = []

    for cid, lab in raw.items():
        merged[cid] = normalize_label(lab)
        raw_labels.append(lab)

    raw_distinct = len(set(raw_labels))
    norm_distinct = len(set(merged.values()))
    print(
        f"[labels] clusters: {len(merged)} | "
        f"distinct raw labels: {raw_distinct}"
    )
    print(f"[labels] distinct normalized labels: {norm_distinct}")
    dst_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[labels] written merged labels to: {dst_path}")
    return merged


def _normalize_df_labels(
        df: pd.DataFrame,
        merged_labels: Dict[str, str],
        id_col: str = "cluster_id",
        label_col: str = "cluster_label",
        fallback_label_cols: Iterable[str] = ("cluster_label", "label"),
) -> pd.DataFrame:
    """
    Given a DataFrame with cluster info, normalize its label column.

    1) If there is a cluster_id and we have a merged label for it,
       use that.
    2) Otherwise, normalize the text in cluster_label/label column.
    """
    df = df.copy()

    # ensure cluster_id is string for mapping
    if id_col in df.columns:
        df[id_col] = df[id_col].astype(str)

    # decide which source column to use as the label text
    col_src = None
    for c in fallback_label_cols:
        if c in df.columns:
            col_src = c
            break

    if col_src is None:
        # nothing to normalize, just return
        return df

    def _norm_row(row) -> str:
        cid = str(row[id_col]) if id_col in row and pd.notna(row[id_col]) else None
        if cid is not None and cid in merged_labels:
            return merged_labels[cid]
        return normalize_label(row[col_src])

    df["cluster_label"] = df.apply(_norm_row, axis=1)
    return df


def normalize_jsonl_file(
        src_path: Path,
        dst_path: Path,
        merged_labels: Dict[str, str],
        id_col: str = "cluster_id",
) -> None:
    """
    Load a JSONL file with cluster labels, normalize them, and write out.
    """
    print(f"[file] normalizing: {src_path}")
    if not src_path.exists():
        print(f"[file]   -> SKIP (not found)")
        return

    df = pd.read_json(src_path, lines=True)
    before_labels = set(df.get("cluster_label", []))
    df = _normalize_df_labels(df, merged_labels, id_col=id_col)
    after_labels = set(df.get("cluster_label", []))

    print(
        f"[file]   labels before: {len(before_labels)} | "
        f"after: {len(after_labels)}"
    )
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(dst_path, orient="records", lines=True, force_ascii=False)
    print(f"[file]   written: {dst_path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    base = Path("src/pmc_chunker/out")

    # input paths
    cluster_labels_in = base / "4_5_labeling" / "cluster_labels.json"

    chunks_in = base / "4_5_labeling" / "paragraph_chunks_4000_labeled_ranked.jsonl"
    queries_raw_in = base / "4_6_query_generation" / "queries_gpt51.jsonl"
    queries_dedup_in = base / "4_7_deduplication" / "queries_gpt51_dedup.jsonl"
    queries_eval_in = base / "4_8_eval_setup" / "queries_eval.jsonl"

    # output paths
    cluster_labels_out = base / "4_5_labeling" / "cluster_labels_merged.json"

    chunks_out = base / "4_5_labeling" / "paragraph_chunks_4000_labeled_ranked_merged.jsonl"
    queries_raw_out = base / "4_6_query_generation" / "queries_gpt51_norm.jsonl"
    queries_dedup_out = base / "4_7_deduplication" / "queries_gpt51_dedup_norm.jsonl"
    queries_eval_out = base / "4_8_eval_setup" / "queries_eval_norm.jsonl"

    # 1) normalize the master cluster label map
    merged_labels = normalize_cluster_labels(cluster_labels_in, cluster_labels_out)

    # 2) propagate to chunks
    print(f"[chunks] normalizing: {chunks_in}")
    if not chunks_in.exists():
        print("[chunks]   -> SKIP (no chunks file)")
    else:
        df_chunks = pd.read_json(chunks_in, lines=True)
        before = set(df_chunks.get("cluster_label", []))
        df_chunks = _normalize_df_labels(df_chunks, merged_labels, id_col="cluster_id")
        after = set(df_chunks.get("cluster_label", []))
        print(
            f"[chunks] labels before: {len(before)} | "
            f"after: {len(after)}"
        )
        chunks_out.parent.mkdir(parents=True, exist_ok=True)
        df_chunks.to_json(chunks_out, orient="records", lines=True, force_ascii=False)
        print(f"[chunks] written: {chunks_out}")

    # 3) propagate to query files
    normalize_jsonl_file(queries_raw_in, queries_raw_out, merged_labels, id_col="cluster_id")
    normalize_jsonl_file(queries_dedup_in, queries_dedup_out, merged_labels, id_col="cluster_id")
    normalize_jsonl_file(queries_eval_in, queries_eval_out, merged_labels, id_col="cluster_id")


if __name__ == "__main__":
    main()
