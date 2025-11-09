#!/usr/bin/env python3
# qdrant_upload_paragraphs.py
from __future__ import annotations

import argparse
import json
import re
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


# --------------------- helpers ---------------------
def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def get_tokenizer():
    try:
        import tiktoken  # type: ignore
        enc = tiktoken.get_encoding("cl100k_base")
        return lambda s: len(enc.encode(s))
    except Exception:
        return lambda s: len((s or "").split())


toklen = get_tokenizer()
_PMCID_RX = re.compile(r"(PMC\d+)", re.I)


def derive_pmcid(r: Dict[str, Any]) -> str | None:
    pmcid = (r.get("pmcid") or r.get("PMCID") or "").strip()
    if pmcid:
        return pmcid
    src = (r.get("source_file") or r.get("source") or "").strip()
    m = _PMCID_RX.search(src)
    return m.group(1).upper() if m else None


def section_path_of(r: Dict[str, Any]) -> str | None:
    sec = (r.get("section") or r.get("section_type") or r.get("type") or "").strip()
    sub = (r.get("subsection") or r.get("section_title") or "").strip()
    if sec and sub:
        return f"{sec}/{sub}"
    return sec or sub or None


def coerce_null(v: Any, policy: str) -> Any:
    if v is None:
        if policy == "empty":
            return ""
        if policy == "unknown":
            return "UNKNOWN"
    return v


# --------------------- normalization ---------------------
def normalize_row(r: Dict[str, Any], null_policy: str) -> Dict[str, Any] | None:
    text = (r.get("text") or r.get("chunk_text") or "").strip()
    if not text:
        return None

    pmcid = derive_pmcid(r)
    pmid = r.get("pmid") or r.get("PMID")
    doi = r.get("doi")
    year = r.get("year")
    journal = r.get("journal")
    topic = r.get("topic")
    license_ = r.get("license")
    version = r.get("version") or "v_paragraphs"

    sec_type = r.get("section") or r.get("section_type") or r.get("type")
    sec_title = r.get("subsection") or r.get("section_title")
    spath = section_path_of(r)
    cidx = r.get("chunk_index") or r.get("chunk_id") or 0
    try:
        cidx = int(cidx)
    except Exception:
        pass

    row = {
        "pmcid": pmcid,
        "section_path": spath,
        "chunk_index": cidx,
        "chunk_text": text,
        "token_count": toklen(text),
        "section_type": sec_type,
        "section_title": sec_title,
        "pmid": pmid,
        "doi": doi,
        "year": year,
        "journal": journal,
        "topic": topic,
        "license": license_,
        "version": version,
    }

    for k in list(row.keys()):
        row[k] = coerce_null(row[k], null_policy)
    return row


def stable_id(r: Dict[str, Any]) -> str:
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


def batched(seq, n: int):
    buf: List[Any] = []
    for x in seq:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf


# --------------------- main ---------------------
def main():
    ap = argparse.ArgumentParser("Upload paragraph embeddings + payloads to Qdrant")
    ap.add_argument("--inp", required=True, help="paragraph_chunks_*.jsonl")
    ap.add_argument("--emb", required=True, help="embeddings .npy file")
    ap.add_argument("--collection", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=6333)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--distance", default="cosine", choices=["cosine", "dot"])
    ap.add_argument("--batch-size", type=int, default=2000)
    ap.add_argument("--recreate", action="store_true")
    ap.add_argument("--null-policy", default="none", choices=["none", "empty", "unknown"])
    args = ap.parse_args()

    rows_raw = list(read_jsonl(Path(args.inp)))
    rows: List[Dict[str, Any]] = []
    for r in rows_raw:
        nr = normalize_row(r, args.null_policy)
        if nr:
            rows.append(nr)

    vecs = np.load(args.emb)
    if len(rows) != vecs.shape[0]:
        raise SystemExit(f"Row/vector mismatch: {len(rows)} rows vs {vecs.shape[0]} vectors")

    cli = QdrantClient(host=args.host, port=args.port, api_key=args.api_key, prefer_grpc=False)
    ensure_collection(cli, args.collection, dim=vecs.shape[1], recreate=args.recreate, distance=args.distance)

    total = 0
    for batch in batched(list(zip(rows, range(len(rows)))), args.batch_size):
        pts: List[PointStruct] = []
        for (row, i) in batch:
            pts.append(PointStruct(id=stable_id(row), vector=vecs[i].tolist(), payload=row))
        cli.upsert(collection_name=args.collection, points=pts, wait=True)
        total += len(batch)

    print(f"[upload] uploaded {total} points to '{args.collection}' (dim={vecs.shape[1]}, null_policy={args.null_policy})")


if __name__ == "__main__":
    main()
