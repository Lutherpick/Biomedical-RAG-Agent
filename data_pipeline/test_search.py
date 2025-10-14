from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

def test_semantic_search_returns_hits():
    q = QdrantClient(host="localhost", port=6333)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    vec = model.encode(["glioblastoma treatment with temozolomide"], convert_to_numpy=True)[0].tolist()

    res = q.query_points(
        collection_name="biomed_chunks",
        query=vec,
        limit=5,
        with_payload=True,
    )

    assert len(res.points) > 0
    assert res.points[0].score > 0.3
