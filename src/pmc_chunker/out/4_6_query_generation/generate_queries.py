#!/usr/bin/env python3
# src/generate_queries.py — label + snippet–based query generation

from __future__ import annotations

import argparse
import json
import re
from typing import List, Any, Tuple
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from transformers import pipeline
except Exception:
    pipeline = None

# ---------------------------------------------------------------------
# PROMPT: generate query from TOPIC LABEL + EXEMPLAR SNIPPETS
# ---------------------------------------------------------------------
PROMPT = (
    "You are helping evaluate a biomedical literature search system.\n"
    "Topic label: \"{label}\".\n"
    "Here are example snippets from articles in this topic:\n"
    "{snippets}\n\n"
    "Write one concise research question that a scientist might ask "
    "to search for this kind of literature. The question should be "
    "general enough to be useful as a search query, but clearly focused "
    "on this topic. Output only the question."
)

# How many snippets per cluster and how long each snippet can be
MAX_SNIPPETS = 3
MAX_SNIPPET_CHARS = 220


def _clean_snippet(text: str, max_len: int = MAX_SNIPPET_CHARS) -> str:
    """
    Normalize whitespace and truncate long snippets at a word boundary.
    """
    s = re.sub(r"\s+", " ", str(text).strip())
    if len(s) <= max_len:
        return s

    s = s[:max_len]
    cut = s.rfind(" ")
    if cut > 40:  # keep at least ~40 chars if possible
        s = s[:cut]
    return s + "..."


# ---------------------------------------------------------------------
# Utilities kept for compatibility (not used in label-based mode)
# ---------------------------------------------------------------------
def _norm(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def _load_vectors(args, texts: List[str], nrows: int) -> np.ndarray:
    """
    Kept for backwards compatibility; not used in label-based mode.
    """
    if args.embeddings:
        X = np.load(args.embeddings)
        if X.shape[0] != nrows:
            raise ValueError(f"Row/vector mismatch: {nrows} vs {X.shape}")
        return X
    if args.st_model:
        from sentence_transformers import SentenceTransformer

        m = SentenceTransformer(args.st_model)
        return m.encode(
            texts,
            batch_size=args.embed_bs,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
    raise ValueError("Provide --embeddings or --st-model")


def _build_pipe(model_name: str, device: int, fp16: bool):
    if pipeline is None:
        raise RuntimeError("transformers not installed")
    kw = dict(model=model_name, task="text2text-generation", device=device)
    if fp16:
        kw["model_kwargs"] = {"torch_dtype": "auto"}
    return pipeline(**kw)


def _extract_generated_text(gen: Any) -> str:
    """
    Robustly extract text from HF pipeline outputs.
    """
    if isinstance(gen, dict):
        return (
                gen.get("generated_text")
                or gen.get("summary_text")
                or gen.get("text")
                or str(gen)
        )
    if isinstance(gen, list) and gen:
        g0 = gen[0]
        if isinstance(g0, dict):
            return (
                    g0.get("generated_text")
                    or g0.get("summary_text")
                    or g0.get("text")
                    or str(g0)
            )
        return str(g0)
    return str(gen)


# ---------------------------------------------------------------------
# HF backend
# ---------------------------------------------------------------------
def generate_queries_hf(prompts: List[str], args) -> List[str]:
    pipe = _build_pipe(args.hf_model, args.device, args.fp16)
    texts_out: List[str] = []

    for s in range(0, len(prompts), args.gen_bs):
        batch_prompts = prompts[s : s + args.gen_bs]
        gens = pipe(
            batch_prompts,
            batch_size=args.gen_bs,
            max_new_tokens=args.max_new,
            do_sample=False,
            num_beams=1,
            truncation=True,
        )
        # HF pipeline returns a list aligned with the input batch
        if isinstance(gens, list):
            for gen in gens:
                q = _extract_generated_text(gen).strip()
                texts_out.append(q)
        else:
            # Single object
            q = _extract_generated_text(gens).strip()
            texts_out.append(q)

    return texts_out


# ---------------------------------------------------------------------
# OpenAI backend
# ---------------------------------------------------------------------
def generate_queries_openai(prompts: List[str], args) -> List[str]:
    from openai import OpenAI

    client_kwargs = {}
    if getattr(args, "openai_api_key", None):
        client_kwargs["api_key"] = args.openai_api_key
    if getattr(args, "openai_base_url", None):
        client_kwargs["base_url"] = args.openai_base_url

    client = OpenAI(**client_kwargs)
    texts_out: List[str] = []

    system_msg = (
        "You are helping evaluate a biomedical literature search system. "
        "Given a topic label and exemplar snippets, you write a single, "
        "concise research question about that topic. The question should "
        "be general and not tied to any specific paper."
    )

    for prompt in prompts:
        resp = client.chat.completions.create(
            model=args.openai_model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            # For GPT-5.x: use max_completion_tokens, no temperature
            max_completion_tokens=args.openai_max_new,
        )
        msg = resp.choices[0].message.content or ""
        texts_out.append(msg.strip())

    return texts_out


# ---------------------------------------------------------------------
# MAIN: label + snippet–based query generation
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--embeddings")  # kept for CLI compatibility (unused)
    ap.add_argument("--st-model")    # kept for CLI compatibility (unused)
    ap.add_argument("--out", required=True)

    # LLM backend selection
    ap.add_argument("--llm", choices=["hf", "openai"], default="hf")

    # HF options
    ap.add_argument("--hf-model", default="google/flan-t5-small")
    ap.add_argument("--device", type=int, default=0)  # GPU id, -1 for CPU
    ap.add_argument("--fp16", action="store_true")

    # OpenAI options
    ap.add_argument("--openai-model", default="gpt-5-mini")
    ap.add_argument(
        "--openai-max-new",
        type=int,
        default=64,
        help="max_completion_tokens per query for OpenAI models",
    )
    ap.add_argument(
        "--openai-api-key",
        default=None,
        help="optional, otherwise OPENAI_API_KEY env var is used",
    )
    ap.add_argument(
        "--openai-base-url",
        default=None,
        help="optional override for base_url (Azure, proxy, etc.)",
    )

    # In label-based mode we interpret min-per-cluster as
    # "number of queries per cluster". max-per-cluster is ignored.
    ap.add_argument("--min-per-cluster", type=int, default=1)
    ap.add_argument("--max-per-cluster", type=int, default=1)

    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--embed-bs", type=int, default=256)  # unused here
    ap.add_argument("--gen-bs", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=32)
    args = ap.parse_args()

    # --- FIX: treat --chunks as a FILE PATH, not a JSON literal -------
    chunks_path = Path(args.chunks)
    if not chunks_path.is_file():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    with chunks_path.open("r", encoding="utf-8") as f:
        df = pd.read_json(f, lines=True)
    # ------------------------------------------------------------------

    # Normalize text column name if needed
    if "text" not in df.columns and "chunk_text" in df.columns:
        df = df.rename(columns={"chunk_text": "text"})

    required_cols = {"pmcid", "chunk_index", "cluster_id", "text"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"missing column(s): {missing}")

    if "cluster_label" not in df.columns:
        # For safety; but labeling pipeline should have filled this already.
        df["cluster_label"] = ""

    # -----------------------------------------------------------------
    # Build prompts: one or more queries per cluster, using
    # cluster_label + exemplar snippets from the cluster.
    # -----------------------------------------------------------------
    prompts: List[str] = []
    meta: List[Tuple[int, str, str]] = []  # (cluster_id, label, source_chunk_id)

    n_per_cluster = max(1, args.min_per_cluster)

    grouped = df.groupby("cluster_id", sort=True)
    for cid, g in grouped:
        cid_int = int(cid)
        lbl = str(g["cluster_label"].iloc[0] or "").strip()
        if not lbl:
            lbl = "other"

        # Build exemplar snippets for this cluster from the first few chunks
        texts = list(g["text"].head(MAX_SNIPPETS))
        snippets_list = [
            _clean_snippet(t)
            for t in texts
            if isinstance(t, str) and t.strip()
        ]
        if not snippets_list:
            snippet_block = "- (no exemplar text available)"
        else:
            snippet_block = "\n".join(f"- {s}" for s in snippets_list)

        # Use first chunk as a stable source_chunk_id (metadata only).
        row0 = g.iloc[0]
        source_chunk_id = f"{row0['pmcid']}_chunk{int(row0['chunk_index']):05d}"

        for _ in range(n_per_cluster):
            prompts.append(PROMPT.format(label=lbl, snippets=snippet_block))
            meta.append((cid_int, lbl, source_chunk_id))

    # -----------------------------------------------------------------
    # Run chosen LLM backend
    # -----------------------------------------------------------------
    if args.llm == "hf":
        texts = generate_queries_hf(prompts, args)
    else:
        texts = generate_queries_openai(prompts, args)

    if len(texts) != len(meta):
        raise RuntimeError(
            f"generated {len(texts)} queries for {len(meta)} prompts; mismatch"
        )

    # -----------------------------------------------------------------
    # Build rows
    # -----------------------------------------------------------------
    out_rows = []
    for qn, (qtext, (cid, lbl, src_id)) in enumerate(zip(texts, meta)):
        q = qtext.strip()
        if not q.endswith("?"):
            q = q.rstrip(".") + "?"
        out_rows.append(
            {
                "query_id": f"q_{qn:06d}",
                "query_text": q,
                "source_chunk_id": src_id,
                "cluster_id": cid,
                "label": lbl,
            }
        )

    # -----------------------------------------------------------------
    # Write JSONL
    # -----------------------------------------------------------------
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_clusters = df["cluster_id"].nunique()
    print(
        f"queries: {len(out_rows)}; clusters: {n_clusters}; "
        f"mean/cluster: {len(out_rows) / max(1, n_clusters):.2f}"
    )


if __name__ == "__main__":
    main()
