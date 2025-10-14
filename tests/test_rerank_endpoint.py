# tests/test_rerank_endpoint.py
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_rerank_top5():
    r = client.post("/search", json={"query": "preclinical mouse model", "k": 5, "n_retrieve": 20})
    assert r.status_code == 200
    data = r.json()
    assert "passages" in data
    assert len(data["passages"]) <= 5
    # when data exists, should be >0
    assert isinstance(data["passages"], list)
    if data["passages"]:
        # fields present
        p0 = data["passages"][0]
        for key in ["text", "doc_id", "retriever_score", "rerank_score", "citation"]:
            assert key in p0
        assert "pmcid" in p0["citation"]
        assert "url" in p0["citation"]
