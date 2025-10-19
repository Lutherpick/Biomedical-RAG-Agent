# data_pipeline/uploader_qdrant.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict
import os, json, uuid

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ---------- Paths ----------
ROOT      = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data_pipeline" / "data"
CHUNKS    = DATA_ROOT / "processed" / "chunks" / "chunks.jsonl"

# ---------- Tunables (increase if you have RAM/VRAM & fast network) ----------
COLL = "biomed_chunks"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384 dims
UPLOAD_BATCH_SIZE  = 1024     # vectors per upsert to Qdrant
ENCODE_BATCH_SIZE  = 512      # texts per encode() call
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

# ---------- Qdrant client (gRPC preferred for speed) ----------
q = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, prefer_grpc=True, timeout=60)

if not q.collection_exists(COLL):
    q.create_collection(
        collection_name=COLL,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
        # Keep default optimizer so indexing starts in the background soon after we cross a few K points.
        hnsw_config=models.HnswConfigDiff(m=16, ef_construct=128),
    )
    # Payload indexes (fast filtering); cheap to create up front.
    for name, schema in [
        ("pmcid", "keyword"),
        ("pmid", "keyword"),
        ("doi", "keyword"),
        ("journal", "keyword"),
        ("year", "keyword"),
        ("topic", "keyword"),
        ("section_type", "keyword"),
        ("section_path", "keyword"),
        ("chunk_index", "integer"),
        ("token_count", "integer"),
        ("is_caption", "bool"),
    ]:
        q.create_payload_index(COLL, field_name=name, field_schema=schema)

# ---------- Encoder ----------
device = "cuda" if os.getenv("FORCE_CPU", "0") != "1" else "cpu"
try:
    import torch
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
except Exception:
    device = "cpu"

model = SentenceTransformer(EMB_MODEL, device=device)

def encode_texts(texts: List[str]):
    # convert_to_numpy is faster for large batches; tolist() only at the end for client
    embs = model.encode(texts, batch_size=ENCODE_BATCH_SIZE, convert_to_numpy=True, normalize_embeddings=False)
    return embs.tolist()

def flush(batch: List[Dict]):
    if not batch:
        return
    texts = [b["text"] for b in batch]
    vectors = encode_texts(texts)

    # Build points in-place to minimize copies
    points = [
        models.PointStruct(
            id=str(uuid.uuid4()),
            vector=vec,
            payload=rec,
        )
        for vec, rec in zip(vectors, batch)
    ]
    # gRPC upsert (fast). If you want even more throughput, run Qdrant with higher write-threads.
    q.upsert(collection_name=COLL, points=points, wait=False)

def main():
    assert CHUNKS.exists(), f"Missing {CHUNKS}"
    batch: List[Dict] = []
    total = 0

    with CHUNKS.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Uploading (encode+upsert)", unit="rec"):
            try:
                rec = json.loads(line)
            except Exception:
                continue

            payload = {
                "pmcid": rec.get("pmcid"),
                "pmid": rec.get("pmid"),
                "doi": rec.get("doi"),
                "journal": rec.get("journal"),
                "year": rec.get("year"),
                "title": rec.get("title"),
                "topic": rec.get("topic", ""),
                "section_type": rec.get("section_type"),
                "section_path": rec.get("section_path"),
                "chunk_index": rec.get("chunk_index", 0),
                "token_count": rec.get("token_count", 0),
                "text": rec.get("text", ""),
                "is_caption": bool(rec.get("is_caption", False)),
                "figure_ids": rec.get("figure_ids", []),
                "image_paths": rec.get("image_paths", []),
            }
            if not payload["text"]:
                continue

            batch.append(payload)
            if len(batch) >= UPLOAD_BATCH_SIZE:
                flush(batch); total += len(batch); batch.clear()

    flush(batch); total += len(batch)

    print("✅ Upload complete.")
    print("Total uploaded chunks:", total)
    try:
        print("Points in collection:", q.count(COLL, exact=True).count)
    except Exception:
        pass

if __name__ == "__main__":
    main()
