# data_pipeline/test_search.py
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

def test_semantic_search_returns_hits():
    q = QdrantClient(host="localhost", port=6333)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    vec = model.encode(["glioblastoma treatment with temozolomide"], convert_to_numpy=True)[0].tolist()

    hits = q.search(
        collection_name="biomed_chunks",
        query_vector=vec,
        limit=5,
        with_payload=True,
    )

    assert len(hits) > 0
    # Should be a reasonable cosine score if data is there:
    assert hits[0].score > 0.3
