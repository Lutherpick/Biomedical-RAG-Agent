# api/main.py
# pip install fastapi uvicorn qdrant-client sentence-transformers pydantic-settings python-dotenv
from __future__ import annotations
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional, Any, Dict
from datetime import datetime
from pathlib import Path
import json

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer, CrossEncoder


# ----------------------------- Config -----------------------------
class Settings(BaseSettings):
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "biomed_chunks"

    # retriever (same as documents)
    EMB_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim

    # reranker (small + fast)
    RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # defaults
    DEFAULT_K: int = 5              # final top-k
    DEFAULT_N_RETRIEVE: int = 30    # candidates before rerank

    LOG_DIR: str = "logs"

    # Pydantic v2 style (replaces inner Config class)
    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()
Path(settings.LOG_DIR).mkdir(parents=True, exist_ok=True)
LOG_FILE = Path(settings.LOG_DIR) / "search_log.jsonl"


# ----------------------------- App singletons -----------------------------
app = FastAPI(title="Biomedical Semantic Search API", version="0.2.0")

qdrant = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
embedder = SentenceTransformer(settings.EMB_MODEL)
reranker = CrossEncoder(settings.RERANK_MODEL)


# ----------------------------- Schemas -----------------------------
class SearchRequest(BaseModel):
    query: str = Field(..., description="Natural language query")
    # retrieve + rerank controls
    n_retrieve: int = Field(default=settings.DEFAULT_N_RETRIEVE, ge=5, le=200)
    k: int = Field(default=settings.DEFAULT_K, ge=1, le=50)
    doc_id: Optional[str] = Field(None, description="optional filter by doc_id")

class Citation(BaseModel):
    pmcid: str
    url: str
    doc_id: str
    chunk_index: int

class Passage(BaseModel):
    # scores
    retriever_score: float
    rerank_score: float
    # content + metadata
    text: str
    doc_id: str
    chunk_index: int
    chunk_total: Optional[int] = None
    citation: Citation

class SearchResponse(BaseModel):
    query: str
    k: int
    passages: List[Passage]


# ----------------------------- Helpers -----------------------------
def build_filter(doc_id: Optional[str]) -> Optional[Filter]:
    if not doc_id:
        return None
    return Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])

def pmcid_from_doc_id(doc_id: str) -> str:
    # doc_id looks like "PMC12345678_0" -> take the prefix as PMCID
    return doc_id.split("_")[0] if doc_id else ""

def pmc_url(pmcid: str) -> str:
    return f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"

def log_search(payload: Dict[str, Any]) -> None:
    payload = dict(payload)
    payload["ts"] = datetime.utcnow().isoformat() + "Z"
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


# ----------------------------- Health -----------------------------
@app.get("/", tags=["health"])
def root():
    return {"ok": True, "service": "biomed-semantic-search", "collection": settings.QDRANT_COLLECTION}


# ----------------------------- Search + Rerank -----------------------------
@app.post("/search", response_model=SearchResponse, tags=["search"])
def search(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")
    if req.k > req.n_retrieve:
        # always retrieve at least k candidates
        req.n_retrieve = req.k

    # 1) Encode query (retriever)
    try:
        qvec = embedder.encode([req.query], convert_to_numpy=True)[0].tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

    # 2) Initial semantic retrieve from Qdrant
    q_filter = build_filter(req.doc_id)
    try:
        retrieved = qdrant.query_points(
            collection_name=settings.QDRANT_COLLECTION,
            query=qvec,
            limit=req.n_retrieve,
            with_payload=True,
            query_filter=q_filter,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Qdrant search failed: {e}")

    if not retrieved.points:
        return SearchResponse(query=req.query, k=req.k, passages=[])

    # 3) Prepare candidates for reranker
    cands = []
    for hit in retrieved.points:
        p = hit.payload or {}
        text = str(p.get("text", ""))[:3000]  # keep it sane for the cross-encoder
        cands.append({
            "text": text,
            "retriever_score": float(hit.score),
            "doc_id": str(p.get("doc_id", "")),
            "chunk_index": int(p.get("chunk_index", 0)),
            "chunk_total": (p.get("chunk_total") if p.get("chunk_total") is None else int(p.get("chunk_total"))),
        })

    # 4) Rerank with cross-encoder
    pairs = [(req.query, c["text"]) for c in cands]
    try:
        rerank_scores = reranker.predict(pairs)  # higher is better
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reranker failed: {e}")

    for c, s in zip(cands, rerank_scores):
        c["rerank_score"] = float(s)

    # 5) Sort by rerank score desc and trim to top-k
    cands.sort(key=lambda x: x["rerank_score"], reverse=True)
    top = cands[:req.k]

    # 6) Build response with citation metadata
    passages: List[Passage] = []
    for c in top:
        pmcid = pmcid_from_doc_id(c["doc_id"])
        passages.append(Passage(
            retriever_score=c["retriever_score"],
            rerank_score=c["rerank_score"],
            text=c["text"],
            doc_id=c["doc_id"],
            chunk_index=c["chunk_index"],
            chunk_total=c["chunk_total"],
            citation=Citation(
                pmcid=pmcid,
                url=pmc_url(pmcid) if pmcid else "",
                doc_id=c["doc_id"],
                chunk_index=c["chunk_index"],
            ),
        ))

    # 7) Log
    log_search({
        "query": req.query,
        "k": req.k,
        "n_retrieve": req.n_retrieve,
        "doc_id": req.doc_id,
        "n_hits_stage1": len(retrieved.points),
        "n_hits_final": len(passages),
        "sample": [
            {"doc_id": p.doc_id, "chunk_index": p.chunk_index, "ret": p.retriever_score, "rr": p.rerank_score}
            for p in passages
        ],
    })

    return SearchResponse(query=req.query, k=req.k, passages=passages)


# Optional: allow `python api/main.py` to run the server directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
