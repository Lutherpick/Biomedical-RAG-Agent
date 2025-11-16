#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Upload paragraph embeddings + payloads to Qdrant.

Typical usage (paragraph-level, labeled):

  .\.venv\Scripts\python.exe .\src\qdrant_upload_paragraphs.py ^
    --inp .\src\pmc_chunker\out\4_5_labeling\paragraph_chunks_4000_labeled.jsonl ^
    --emb .\src\pmc_chunker\out\embeddings_v5.npy ^
    --collection biomed_paragraphs ^
    --distance cosine ^
    --recreate

Notes
------
- This script is responsible for uploading the *corpus* paragraphs and their
  embeddings plus all per-chunk metadata needed at retrieval time.

- It consumes only:
    * paragraph_chunks_*.jsonl  (e.g. paragraph_chunks_4000_labeled.jsonl)
    * embeddings_v*.npy
  It does NOT read cluster_labels.json, cluster_exemplars.json or
  queries_gpt51*.jsonl; those are used only by the evaluation scripts.

- From the chunks JSONL it will populate these payload keys in Qdrant:
    pmcid, section_path, chunk_index, chunk_text, token_count,
    section_type, section_title, pmid, doi, year, journal, topic,
    license, version, cluster_id, cluster_label,
    rank_in_cluster, centroid_distance
"""

from __future__ import annotations

import argparse
import json
import re
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, OptimizersConfigDiff

# ----------------- utils -----------------
_PMCID_RX = re.compile(r"(PMC\d+)", re.I)


def s(x: Any) -> str:
    """Safe string. None/NaN/float→''; else str(x)."""
    try:
        if x is None:
            return ""
        if isinstance(x, float):
            # NaN-safe
            if x != x:  # NaN
                return ""
        return str(x)
    except Exception:
        return ""


def s_strip(x: Any) -> str:
    return s(x).strip()


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                # skip bad lines
                continue


def get_tokenizer():
    """Return a token length function, preferring tiktoken if available."""
    try:
        import tiktoken  # type: ignore

        enc = tiktoken.get_encoding("cl100k_base")
        return lambda z: len(enc.encode(s(z)))
    except Exception:
        return lambda z: len(s(z).split())


toklen = get_tokenizer()


def derive_pmcid(r: Dict[str, Any]) -> Optional[str]:
    pmcid = s_strip(r.get("pmcid") or r.get("PMCID"))
    if pmcid:
        return pmcid.upper()
    src = s_strip(r.get("source_file") or r.get("source"))
    m = _PMCID_RX.search(src)
    return m.group(1).upper() if m else None


def section_path_of(r: Dict[str, Any]) -> Optional[str]:
    sec = s_strip(r.get("section") or r.get("section_type") or r.get("type"))
    sub = s_strip(r.get("subsection") or r.get("section_title"))
    if sec and sub:
        return f"{sec}/{sub}"
    return sec or sub or None


def coerce_null(v: Any, policy: str) -> Any:
    """Apply null policy to a single value."""
    if v is None or v == "":
        if policy == "empty":
            return ""
        if policy == "unknown":
            return "UNKNOWN"
        return None
    return v


def normalize_row(r: Dict[str, Any], null_policy: str) -> Optional[Dict[str, Any]]:
    """
    Map one paragraph row from the chunks JSONL into a normalized payload
    dictionary for Qdrant. Returns None if there is no usable text.
    """
    text = s_strip(r.get("text") or r.get("chunk_text"))
    if not text:
        return None

    pmcid = derive_pmcid(r)
    pmid = s_strip(r.get("pmid") or r.get("PMID"))
    doi = s_strip(r.get("doi"))
    year = s_strip(r.get("year"))
    journal = s_strip(r.get("journal"))
    topic = s_strip(r.get("topic"))
    license_ = s_strip(r.get("license"))
    version = s_strip(r.get("version")) or "v_paragraphs"

    sec_type = s_strip(r.get("section") or r.get("section_type") or r.get("type"))
    sec_title = s_strip(r.get("subsection") or r.get("section_title"))
    spath = section_path_of(r)

    cidx = r.get("chunk_index") or r.get("chunk_id") or 0
    try:
        cidx = int(cidx)
    except Exception:
        cidx = -1

    # Cluster metadata (may be absent for older runs)
    cluster_id = r.get("cluster_id")
    try:
        if cluster_id is not None and s_strip(cluster_id) != "":
            cluster_id = int(str(cluster_id).strip())
        else:
            cluster_id = None
    except Exception:
        cluster_id = None

    cluster_label = s_strip(r.get("cluster_label"))

    rank_in_cluster = r.get("rank_in_cluster")
    try:
        if rank_in_cluster is not None and s_strip(rank_in_cluster) != "":
            rank_in_cluster = int(str(rank_in_cluster).strip())
        else:
            rank_in_cluster = None
    except Exception:
        rank_in_cluster = None

    centroid_distance = r.get("centroid_distance")
    try:
        if centroid_distance is not None and s_strip(centroid_distance) != "":
            centroid_distance = float(str(centroid_distance).strip())
        else:
            centroid_distance = None
    except Exception:
        centroid_distance = None

    row = {
        "pmcid": pmcid,
        "section_path": spath,
        "chunk_index": cidx,
        "chunk_text": text,
        "token_count": toklen(text),
        "section_type": sec_type or None,
        "section_title": sec_title or None,
        "pmid": pmid or None,
        "doi": doi or None,
        "year": year or None,
        "journal": journal or None,
        "topic": topic or None,
        "license": license_ or None,
        "version": version,
        "cluster_id": cluster_id,
        "cluster_label": cluster_label or None,
        "rank_in_cluster": rank_in_cluster,
        "centroid_distance": centroid_distance,
    }

    # Apply null policy
    for k in list(row.keys()):
        row[k] = coerce_null(row[k], null_policy)

    return row


def stable_id(r: Dict[str, Any]) -> str:
    """
    Deterministic UUID based on pmcid, section_path and chunk_index.
    This allows safe re-upload without creating duplicates.
    """
    key = f"{r.get('pmcid','NA')}|{r.get('section_path','/')}|{r.get('chunk_index','?')}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "biomed://" + key))


def ensure_collection(cli: QdrantClient, name: str, dim: int, recreate: bool, distance: str):
    if recreate:
        try:
            cli.delete_collection(name)
        except Exception:
            pass

    dist = Distance.COSINE if distance.lower() == "cosine" else Distance.DOT

    if not cli.collection_exists(name):
        cli.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=dist),
        )
        cli.update_collection(
            name,
            optimizers_config=OptimizersConfigDiff(memmap_threshold=20000),
        )


def batched(seq, n: int):
    buf: List[Any] = []
    for x in seq:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf


# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser("Upload paragraph embeddings + payloads to Qdrant")
    ap.add_argument(
        "--inp",
        required=True,
        help="paragraph_chunks_*.jsonl or *_labeled.jsonl (paragraph-level chunks with metadata)",
    )
    ap.add_argument(
        "--emb",
        required=True,
        help="embeddings .npy file (same number of rows as JSONL)",
    )
    ap.add_argument("--collection", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=6333)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--distance", default="cosine", choices=["cosine", "dot"])
    ap.add_argument("--batch-size", type=int, default=2000)
    ap.add_argument(
        "--recreate",
        action="store_true",
        help="drop and recreate collection before upload",
    )
    ap.add_argument(
        "--null-policy",
        default="none",
        choices=["none", "empty", "unknown"],
        help="how to store empty values in payload",
    )
    args = ap.parse_args()

    inp_path = Path(args.inp)
    emb_path = Path(args.emb)

    if not inp_path.is_file():
        raise SystemExit(f"[ERROR] Input JSONL not found: {inp_path}")
    if not emb_path.is_file():
        raise SystemExit(f"[ERROR] Embeddings file not found: {emb_path}")

    rows_raw = list(read_jsonl(inp_path))
    rows: List[Dict[str, Any]] = []
    for r in rows_raw:
        nr = normalize_row(r, args.null_policy)
        if nr:
            rows.append(nr)

    vecs = np.load(emb_path)
    if len(rows) != vecs.shape[0]:
        raise SystemExit(
            f"[ERROR] Row/vector mismatch: {len(rows)} rows vs {vecs.shape[0]} vectors"
        )

    # basic sanity
    none_pmcid = sum(1 for r in rows if r.get("pmcid") in (None, ""))
    if none_pmcid > 0:
        print(f"[warn] {none_pmcid} rows missing PMCID")

    dup_keys = len(rows) - len(
        {(r["pmcid"], r["section_path"], r["chunk_index"]) for r in rows}
    )
    if dup_keys:
        print(f"[warn] {dup_keys} duplicate (pmcid,section_path,chunk_index) keys")

    cli = QdrantClient(
        host=args.host, port=args.port, api_key=args.api_key, prefer_grpc=False
    )
    ensure_collection(
        cli, args.collection, dim=vecs.shape[1], recreate=args.recreate, distance=args.distance
    )

    total = 0
    for batch in batched(list(zip(rows, range(len(rows)))), args.batch_size):
        pts: List[PointStruct] = []
        for (row, i) in batch:
            pts.append(
                PointStruct(
                    id=stable_id(row),
                    vector=vecs[i].tolist(),
                    payload=row,
                )
            )
        cli.upsert(collection_name=args.collection, points=pts, wait=True)
        total += len(batch)

    print(
        f"[upload] uploaded {total} points to '{args.collection}' "
        f"(dim={vecs.shape[1]}, null_policy={args.null_policy})"
    )


if __name__ == "__main__":
    main()
