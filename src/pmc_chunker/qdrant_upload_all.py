#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, uuid
from pathlib import Path
from typing import Iterable, List, Dict, Any
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# ---------- IO ----------
def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

# ---------- Schema normalization ----------
def norm_row(r: Dict[str, Any]) -> Dict[str, Any]:
    pmcid = r.get("PMCID") or r.get("pmcid")
    section_path = r.get("section_path") or r.get("section_title")
    chunk_index = r.get("chunk_id") if "chunk_id" in r else r.get("chunk_index")
    text = r.get("text") or r.get("chunk_text")
    token_count = r.get("token_count") or r.get("tokens") or r.get("token_len")

    return {
        "pmcid": pmcid,
        "section_path": section_path,
        "chunk_index": chunk_index,
        "text": text,
        "token_count": token_count,
        "version": r.get("version"),
        "section_type": r.get("section_type"),
        "section_title": r.get("section_title"),
        "pmid": r.get("pmid"),
        "doi": r.get("doi"),
        "year": r.get("year"),
        "journal": r.get("journal"),
        "topic": r.get("topic"),
        "license": r.get("license"),
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
    dist = Distance.COSINE if distance.lower() == "cosine" else Distance.DOT
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=dist)
        )

def batched(seq, n: int):
    buf = []
    for x in seq:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf

# ---------- Path resolver ----------
def resolve_paths(args) -> tuple[Path, Path]:
    base = Path("src/pmc_chunker/out")
    if args.inp and args.emb:
        return Path(args.inp), Path(args.emb)
    if not args.version:
        raise SystemExit("Provide --version or both --inp and --emb paths.")
    inp = base / f"chunks_{args.version}.jsonl"
    emb = base / f"emb_{args.version}.npy"
    return inp, emb

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser("Upload embeddings + metadata to Qdrant")
    ap.add_argument("--version", choices=["v1", "v2", "v3"], help="Auto-pick paths by version")
    ap.add_argument("--inp", help="Optional explicit chunks.jsonl")
    ap.add_argument("--emb", help="Optional explicit emb.npy")
    ap.add_argument("--collection", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=6333)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--distance", default="cosine", choices=["cosine", "dot"])
    ap.add_argument("--batch-size", type=int, default=2000)
    ap.add_argument("--recreate", action="store_true")
    ap.add_argument("--include-text", action="store_true")
    args = ap.parse_args()

    inp, emb = resolve_paths(args)
    vecs = np.load(emb)
    rows = [norm_row(r) for r in read_jsonl(inp)]
    if len(rows) != vecs.shape[0]:
        raise ValueError(f"Row/vector mismatch: {len(rows)} rows vs {vecs.shape[0]} vectors")

    client = QdrantClient(host=args.host, port=args.port, api_key=args.api_key, prefer_grpc=False)
    ensure_collection(client, args.collection, dim=vecs.shape[1], recreate=args.recreate, distance=args.distance)

    total = 0
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
                "topic": row.get("topic"),
                "license": row.get("license"),
            }
            if args.include_text:
                payload["chunk_text"] = row["text"]
            pid = stable_uuid(row)
            pts.append(PointStruct(id=pid, vector=vecs[idx].tolist(), payload=payload))
        client.upsert(collection_name=args.collection, points=pts)
        total += len(batch)

    print(f"[qdrant] uploaded {total} points to '{args.collection}' (dim={vecs.shape[1]})")

if __name__ == "__main__":
    main()
