#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, uuid
from pathlib import Path
from typing import Iterable, List, Dict, Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


# -------- IO --------
def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# -------- schema normalization --------
def norm_row(r: Dict[str, Any]) -> Dict[str, Any]:
    # Accept both old and new keys
    pmcid = r.get("PMCID") or r.get("pmcid")
    section_path = r.get("section_path") or r.get("section") or r.get("section_title")
    chunk_index = r.get("chunk_id") if "chunk_id" in r else r.get("chunk_index")
    text = r.get("text") or r.get("chunk_text")
    token_count = r.get("token_count") or r.get("tokens") or r.get("token_len")

    # Optional legacy fields
    version = r.get("version")
    section_type = r.get("section_type")
    section_title = r.get("section_title") or r.get("section")

    return {
        "pmcid": pmcid,
        "section_path": section_path,
        "chunk_index": chunk_index,
        "text": text,
        "token_count": token_count,
        "version": version,
        "section_type": section_type,
        "section_title": section_title,
        # pass-through common biblio if present
        "pmid": r.get("pmid"),
        "doi": r.get("doi"),
        "year": r.get("year"),
        "journal": r.get("journal"),
    }


def stable_uuid(row: Dict[str, Any]) -> str:
    key = f"{row.get('pmcid','NA')}|{row.get('section_path','/')}|{row.get('chunk_index','?')}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "biomed://" + key))


def ensure_collection(client: QdrantClient, name: str, dim: int, recreate: bool, distance: str):
    if recreate:
        try:
            client.delete_collection(name)
        except Exception:
            pass
    try:
        exists = client.collection_exists(name)
    except Exception:
        exists = False
    dist = Distance.COSINE if distance.lower() == "cosine" else Distance.DOT
    if not exists:
        client.create_collection(collection_name=name, vectors_config=VectorParams(size=dim, distance=dist))


def batched(seq, n: int):
    buf = []
    for x in seq:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf


# -------- main --------
def main():
    ap = argparse.ArgumentParser("Upload embeddings + metadata to Qdrant")
    ap.add_argument("--inp", required=True, help="chunks.jsonl")
    ap.add_argument("--emb", required=True, help="embeddings .npy")
    ap.add_argument("--collection", required=True)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=6333)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--distance", default="cosine", choices=["cosine", "dot"])
    ap.add_argument("--batch-size", type=int, default=2000)
    ap.add_argument("--recreate", action="store_true")
    ap.add_argument("--include-text", action="store_true")
    args = ap.parse_args()

    inp = Path(args.inp)
    emb = Path(args.emb)

    vecs = np.load(emb)
    if vecs.ndim != 2:
        raise ValueError("embeddings must be 2D")

    rows_raw = list(read_jsonl(inp))
    rows = [norm_row(r) for r in rows_raw]

    if len(rows) != vecs.shape[0]:
        raise ValueError(f"row/vec mismatch: {len(rows)} rows vs {vecs.shape[0]} vectors")

    client = QdrantClient(host=args.host, port=args.port, api_key=args.api_key, prefer_grpc=False)
    ensure_collection(client, args.collection, dim=vecs.shape[1], recreate=args.recreate, distance=args.distance)

    sent = 0
    for batch in batched(list(zip(rows, range(len(rows)))), args.batch_size):
        pts: List[PointStruct] = []
        for (row, idx) in batch:
            payload = {
                "pmcid": row["pmcid"],
                "section_path": row["section_path"],
                "chunk_index": row["chunk_index"],
                "token_count": row["token_count"],
                "version": row.get("version"),
                "section_type": row.get("section_type"),
                "section_title": row.get("section_title"),
                "pmid": row.get("pmid"),
                "doi": row.get("doi"),
                "year": row.get("year"),
                "journal": row.get("journal"),
            }
            if args.include_text:
                payload["chunk_text"] = row["text"]

            pid = stable_uuid(row)
            pts.append(PointStruct(id=pid, vector=vecs[idx].tolist(), payload=payload))

        client.upsert(collection_name=args.collection, points=pts)
        sent += len(batch)

    print(f"[qdrant] uploaded {sent} points to '{args.collection}' (dim={vecs.shape[1]})")


if __name__ == "__main__":
    main()
