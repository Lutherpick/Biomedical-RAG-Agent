#!/usr/bin/env python3
"""
4.9 Testing the Retrieval Pipeline

Run semantic search in Qdrant for all evaluation queries and persist the
top-k results per query as retrieval_results.jsonl.

Supports two backends for query embeddings:
  --backend hf     → SentenceTransformers (same as paragraph vectors) [default]
  --backend openai → OpenAI embeddings API (text-embedding-3-small etc.)

Default locations (relative to this file):
- Queries:      ../4_8_eval_setup/queries_eval.jsonl
- Config:       ../4_8_eval_setup/retrieval_config.json
- Output dir:   ./   (4_9_retrieval/)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def batched(seq: Iterable[Any], batch_size: int) -> Iterable[List[Any]]:
    buf: List[Any] = []
    for x in seq:
        buf.append(x)
        if len(buf) >= batch_size:
            yield buf
            buf = []
    if buf:
        yield buf


@dataclass
class RetrievalConfig:
    collection: str
    embedding_model: str
    embedding_model_version: str = "v0.2"
    index_type: str = "HNSW"
    distance: str = "cosine"
    top_k: int = 5

    @classmethod
    def from_json(cls, path: Path) -> "RetrievalConfig":
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        return cls(
            collection=raw.get("collection", "biomed_paragraphs"),
            # For HF backend this is just informational
            embedding_model=raw.get(
                "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            embedding_model_version=raw.get("embedding_model_version", "v0.1"),
            index_type=raw.get("index_type", "HNSW"),
            distance=raw.get("distance", "cosine"),
            top_k=int(raw.get("top_k", 5)),
        )


def infer_default_paths(this_file: Path) -> Tuple[Path, Path, Path]:
    """
    this_file: .../src/pmc_chunker/out/4_9_retrieval/run_retrieval.py
    out_root: .../src/pmc_chunker/out
    """
    out_dir = this_file.parent
    out_root = out_dir.parent
    queries = out_root / "4_8_eval_setup" / "queries_eval.jsonl"
    cfg = out_root / "4_8_eval_setup" / "retrieval_config.json"
    return queries, cfg, out_dir


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve()
    def_q, def_cfg, def_out = infer_default_paths(here)

    p = argparse.ArgumentParser(
        description="4.9 Run retrieval against Qdrant for all evaluation queries.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--queries",
        type=Path,
        default=def_q,
        help="queries_eval.jsonl produced in 4.8",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=def_cfg,
        help="retrieval_config.json describing collection and embedding model",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=def_out,
        help="Directory where retrieval_results.jsonl and retrieval_errors.csv are written",
    )

    # Qdrant connection
    p.add_argument("--host", type=str, default="127.0.0.1", help="Qdrant host")
    p.add_argument("--port", type=int, default=6333, help="Qdrant HTTP port")
    p.add_argument(
        "--qdrant-api-key",
        type=str,
        default=None,
        help="Qdrant API key (optional)",
    )

    # Embedding backend
    p.add_argument(
        "--backend",
        type=str,
        choices=["hf", "openai"],
        default="hf",
        help="Embedding backend for queries (hf=SentenceTransformers, openai=OpenAI embeddings).",
    )

    # HF backend options
    p.add_argument(
        "--hf-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformers model name (must match paragraph embeddings).",
    )
    p.add_argument(
        "--hf-device",
        type=str,
        default="cuda",
        help="Device for HF model (cuda or cpu).",
    )

    # OpenAI backend options
    p.add_argument(
        "--openai-api-key",
        type=str,
        default=None,
        help="OpenAI API key (if not set via env).",
    )
    p.add_argument(
        "--openai-base-url",
        type=str,
        default=None,
        help="Custom base_url for OpenAI-compatible endpoints.",
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of queries to embed per batch.",
    )
    p.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Optional cap on number of queries to process (for debugging).",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Override top_k from retrieval_config.json (if provided).",
    )

    return p.parse_args()


# ---------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------


def embed_queries_hf(
        texts: List[str],
        model_name: str,
        device: str = "cuda",
        batch_size: int = 64,
) -> List[List[float]]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    out: List[List[float]] = []

    for batch in batched(texts, batch_size):
        emb = model.encode(
            batch,
            batch_size=len(batch),
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        for row in emb:
            out.append(row.astype(np.float32).tolist())

    if len(out) != len(texts):
        raise RuntimeError(f"HF: Expected {len(texts)} embeddings, got {len(out)}")
    return out


def embed_queries_openai(
        texts: List[str],
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        batch_size: int = 32,
) -> List[List[float]]:
    from openai import OpenAI  # type: ignore

    client_kwargs: Dict[str, Any] = {}
    if api_key:
        client_kwargs["api_key"] = api_key
    if base_url:
        client_kwargs["base_url"] = base_url

    client = OpenAI(**client_kwargs)

    out: List[List[float]] = []
    for batch in batched(texts, batch_size):
        resp = client.embeddings.create(model=model, input=batch)
        for item in resp.data:
            out.append(list(item.embedding))
    if len(out) != len(texts):
        raise RuntimeError(f"OpenAI: Expected {len(texts)} embeddings, got {len(out)}")
    return out


# ---------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------


def build_chunk_id(pmcid: Optional[str], chunk_index: Optional[int]) -> str:
    pmcid = (pmcid or "UNKNOWN").strip()
    try:
        idx = int(chunk_index) if chunk_index is not None else -1
    except Exception:
        idx = -1
    if idx >= 0:
        return f"{pmcid}_chunk{idx:05d}"
    return pmcid or "UNKNOWN"


def main() -> None:
    args = parse_args()

    if not args.queries.is_file():
        raise SystemExit(f"[4.9] queries_eval.jsonl not found: {args.queries}")
    if not args.config.is_file():
        raise SystemExit(f"[4.9] retrieval_config.json not found: {args.config}")

    cfg = RetrievalConfig.from_json(args.config)
    if args.top_k is not None:
        cfg.top_k = int(args.top_k)

    # -------------------- Load queries --------------------
    df = pd.read_json(args.queries, lines=True)

    required_cols = {"query_id", "query_text"}
    missing = required_cols - set(df.columns)
    if missing:
        raise SystemExit(f"[4.9] queries file missing columns: {missing}")

    if args.max_queries is not None:
        df = df.iloc[: int(args.max_queries)].copy()

    query_ids = df["query_id"].astype(str).tolist()
    texts = df["query_text"].astype(str).tolist()

    print(f"[4.9] Loaded {len(df)} queries from {args.queries}")
    print(
        f"[4.9] Using collection='{cfg.collection}', backend='{args.backend}', "
        f"model='{cfg.embedding_model}', top_k={cfg.top_k}"
    )

    # -------------------- Embed queries --------------------
    if args.backend == "hf":
        embeds = embed_queries_hf(
            texts,
            model_name=args.hf_model,
            device=args.hf_device,
            batch_size=max(1, args.batch_size),
        )
    else:
        embeds = embed_queries_openai(
            texts,
            model=cfg.embedding_model,
            api_key=args.openai_api_key,
            base_url=args.openai_base_url,
            batch_size=max(1, args.batch_size),
        )

    vecs = np.asarray(embeds, dtype=np.float32)

    # -------------------- Init Qdrant client --------------------
    cli = QdrantClient(
        host=args.host,
        port=args.port,
        api_key=args.qdrant_api_key,
        prefer_grpc=False,
    )

    # Optional: print expected vector size
    try:
        info = cli.get_collection(cfg.collection)
        dim = getattr(getattr(getattr(info, "config", None), "params", None), "vectors", None)
        size = getattr(dim, "size", None)
        print(f"[4.9] Qdrant collection dim={size}")
    except Exception as exc:
        print(f"[4.9] Could not inspect collection dim: {exc}")

    # -------------------- Run search --------------------
    out_path = args.out_dir / "retrieval_results.jsonl"
    err_path = args.out_dir / "retrieval_errors.csv"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results_rows: List[Dict[str, Any]] = []
    error_rows: List[Dict[str, Any]] = []

    for i, (qid, qvec) in enumerate(zip(query_ids, vecs)):
        try:
            hits = cli.search(
                collection_name=cfg.collection,
                query_vector=qvec.tolist(),
                limit=cfg.top_k,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            msg = str(exc)
            error_rows.append({"query_id": qid, "error": msg})
            if i == 0:
                print(f"[4.9] Example error for first failing query: {msg}")
            continue

        chunk_ids: List[str] = []
        labels: List[Optional[str]] = []
        scores: List[float] = []

        for h in hits:
            payload = getattr(h, "payload", None) or {}
            pmcid = payload.get("pmcid")
            cidx = payload.get("chunk_index")
            cid = build_chunk_id(pmcid, cidx)
            lbl = payload.get("cluster_label")
            chunk_ids.append(cid)
            labels.append(lbl if lbl is not None else "")
            scores.append(float(getattr(h, "score", 0.0)))

        results_rows.append(
            {
                "query_id": qid,
                "retrieved_chunk_ids": chunk_ids,
                "retrieved_labels": labels,
                "retrieved_scores": scores,
            }
        )

        if (i + 1) % 50 == 0 or (i + 1) == len(df):
            print(f"[4.9] Processed {i + 1}/{len(df)} queries")

    # -------------------- Persist outputs --------------------
    with out_path.open("w", encoding="utf-8") as f:
        for r in results_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(
        f"[4.9] Wrote retrieval results for {len(results_rows)} queries to {out_path}"
    )

    if error_rows:
        import csv

        with err_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["query_id", "error"])
            w.writeheader()
            w.writerows(error_rows)
        print(
            f"[4.9] Encountered errors for {len(error_rows)} queries; "
            f"details in {err_path}"
        )


if __name__ == "__main__":
    main()
