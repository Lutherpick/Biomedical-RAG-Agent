# data_pipeline/uploader_qdrant.py
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from pathlib import Path
from tqdm import tqdm
import json, uuid

CHUNK_DIR = Path("data/processed/chunks")
COLL = "biomed_chunks"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384 dims
BATCH_SIZE = 128

q = QdrantClient(host="localhost", port=6333)

# Create collection if missing (no deprecation)
if not q.collection_exists(COLL):
    q.create_collection(
        collection_name=COLL,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=2000),
        hnsw_config=models.HnswConfigDiff(m=16, ef_construct=128),
        quantization_config=None,  # keep full precision for now
    )
    # Add payload indexes for fast filtering later
    q.create_payload_index(COLL, field_name="doc_id", field_schema="keyword")
    q.create_payload_index(COLL, field_name="chunk_index", field_schema="integer")
    q.create_payload_index(COLL, field_name="chunk_total", field_schema="integer")

model = SentenceTransformer(EMB_MODEL)

def upsert_batch(batch):
    if not batch: return
    texts = [t for _,_,t in batch]
    embs = model.encode(texts, convert_to_numpy=True).tolist()
    points = []
    for i,(f,rec,_) in enumerate(batch):
        points.append(
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=embs[i],
                payload={
                    "doc_id": rec["doc_id"],
                    "chunk_index": rec["chunk_index"],
                    "chunk_total": rec.get("chunk_total"),
                    "source_path": str(f),
                    "text": rec["text"],
                },
            )
        )
    q.upsert(collection_name=COLL, points=points)

files = list(CHUNK_DIR.glob("*.json"))
batch = []
for f in tqdm(files, desc="Uploading"):
    rec = json.loads(f.read_text(encoding="utf-8"))
    batch.append((f, rec, rec["text"]))
    if len(batch) >= BATCH_SIZE:
        upsert_batch(batch)
        batch.clear()
upsert_batch(batch)

print("Done. Points in collection:", q.count(COLL, exact=True).count)
