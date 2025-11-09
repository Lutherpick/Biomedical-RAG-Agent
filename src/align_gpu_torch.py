#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
align_gpu_torch.py

GPU alignment of clustering previews to paragraph chunks using Sentence-Transformers
and PyTorch cosine similarity. Works on Windows with CUDA. No FAISS needed.

Inputs
  --chunks   JSONL with: pmcid, pmid, chunk_index, text
  --assign   CSV  with: preview, cluster_id
  --out      CSV  output: pmcid, chunk_index, cluster_id

Options
  --cluster-col / --preview-col    column names in --assign
  --doc-hint manifest_4000.csv     also used to BACKFILL missing pmcid via PMID→PMCID
  --doc-max-chunks 80              chunks per doc to build doc-hint corpus
  --model sentence-transformers/all-MiniLM-L6-v2
  --batch 2048
  --topk 50
  --min-sim 0.45
  --doc-topk 3
  --fp16
"""
from __future__ import annotations
import argparse, sys, re
from pathlib import Path
from typing import List, Tuple, Dict
import pandas as pd
import numpy as np

def log(*a): print(*a, flush=True)

def has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

def load_model(name: str, fp16: bool):
    from sentence_transformers import SentenceTransformer
    device = "cuda" if has_cuda() else "cpu"
    model = SentenceTransformer(name, device=device)
    if fp16 and device == "cuda":
        try:
            model = model.half()
        except Exception:
            pass
    return model

PMC_RE = re.compile(r"(PMC\d+)", re.I)
PMID_RE = re.compile(r"(\d+)")

def norm_pmcid(x) -> str | None:
    s = "" if x is None else str(x)
    m = PMC_RE.search(s)
    return m.group(1).upper() if m else None

def norm_pmid(x) -> str | None:
    s = "" if x is None else str(x)
    m = PMID_RE.search(s)
    return m.group(1) if m else None

def resolve_manifest_cols(df: pd.DataFrame):
    cols = {c.lower(): c for c in df.columns}
    pmcid = cols.get("pmcid") or cols.get("pmc_id") or cols.get("pmc")
    pmid  = cols.get("pmid")  or cols.get("pubmed_id")
    title = cols.get("title") or cols.get("article_title") or cols.get("article-title")
    return pmcid, pmid, title

def build_doc_texts(chunks_df: pd.DataFrame, manifest_df: pd.DataFrame, doc_max_chunks: int) -> Dict[str,str]:
    pmcid_col, pmid_col, title_col = resolve_manifest_cols(manifest_df)
    # Title map for a small boost
    title_map = {}
    if title_col and title_col in manifest_df.columns:
        for _, r in manifest_df.iterrows():
            pmc = norm_pmcid(r.get(pmcid_col))
            if pmc:
                title_map[pmc] = str(r[title_col])

    head = (chunks_df.sort_values(["pmcid","chunk_index"])
    .groupby("pmcid", as_index=False).head(doc_max_chunks)[["pmcid","text"]])
    doc_texts = {}
    for pmc, g in head.groupby("pmcid"):
        parts = [title_map.get(pmc, "")]
        parts.extend(g["text"].astype(str).tolist())
        doc_texts[pmc] = " ".join(parts)
    return doc_texts

def encode_texts(model, texts: List[str], batch_size: int, device: str, fp16: bool):
    import torch
    with torch.no_grad():
        embs = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=True,
            normalize_embeddings=True,
            device=device
        )
        embs = embs.half() if (fp16 and device == "cuda") else embs.float()
        return torch.nn.functional.normalize(embs, p=2, dim=1)

def topk_cosine(q, K: int, corpus):
    import torch
    vals, idx = torch.topk(q @ corpus.t(), k=min(K, corpus.shape[0]), dim=1)
    return vals.float().cpu(), idx.long().cpu()

def backfill_pmcid_with_manifest(chunks: pd.DataFrame, manifest_path: Path) -> pd.DataFrame:
    """Fill empty pmcid from manifest via PMID→PMCID."""
    m = pd.read_csv(manifest_path)
    pmcid_col, pmid_col, _ = resolve_manifest_cols(m)
    if not pmid_col or not pmcid_col:
        return chunks

    mm = m[[pmid_col, pmcid_col]].copy()
    mm[pmid_col] = mm[pmid_col].map(norm_pmid)
    mm[pmcid_col] = mm[pmcid_col].map(norm_pmcid)
    mm = mm.dropna()
    mp = dict(mm.values.tolist())  # pmid->pmcid

    before_empty = (chunks["pmcid"].isna() | (chunks["pmcid"] == "")).sum()
    # fill where missing
    need = chunks["pmcid"].isna() | (chunks["pmcid"] == "")
    chunks.loc[need, "pmid_norm"] = chunks.loc[need, "pmid"].map(norm_pmid)
    chunks.loc[need, "pmcid_fill"] = chunks.loc[need, "pmid_norm"].map(mp)
    # use fill if present
    chunks["pmcid"] = np.where(
        need & chunks["pmcid_fill"].notna(),
        chunks["pmcid_fill"].astype(str),
        chunks["pmcid"].astype(str)
    )
    after_empty = (chunks["pmcid"].isna() | (chunks["pmcid"] == "")).sum()
    filled = int(before_empty - after_empty)
    log(f"[backfill] empty pmcid before={before_empty}  filled={filled}  remaining={after_empty}")
    return chunks.drop(columns=[c for c in ["pmid_norm","pmcid_fill"] if c in chunks.columns])

def main():
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--assign", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cluster-col", default="cluster_id")
    ap.add_argument("--preview-col", default="preview")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--min-sim", type=float, default=0.45)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--doc-hint", default=None, help="manifest_4000.csv; also used to backfill pmcid from pmid")
    ap.add_argument("--doc-max-chunks", type=int, default=80)
    ap.add_argument("--doc-topk", type=int, default=3)
    args = ap.parse_args()

    device = "cuda" if has_cuda() else "cpu"
    log(f"Device: {'CUDA' if device=='cuda' else 'CPU'}")

    chunks_path = Path(args.chunks)
    assign_path = Path(args.assign)
    if not chunks_path.exists(): sys.exit(f"missing --chunks: {chunks_path}")
    if not assign_path.exists(): sys.exit(f"missing --assign: {assign_path}")

    # Load
    chunks = pd.read_json(chunks_path, lines=True)
    need = {"pmcid","pmid","chunk_index","text"}
    miss = need - set(chunks.columns)
    if miss:
        sys.exit(f"chunks missing: {sorted(miss)}")
    chunks["pmcid"] = chunks["pmcid"].astype(str)

    # Backfill pmcid from manifest if provided
    if args.doc_hint:
        chunks = backfill_pmcid_with_manifest(chunks, Path(args.doc_hint))

    # Normalize and filter
    chunks["pmcid"] = chunks["pmcid"].map(lambda x: norm_pmcid(x) or "")
    chunks = chunks.dropna(subset=["chunk_index","text"]).copy()
    chunks["chunk_index"] = chunks["chunk_index"].astype(int)
    keep = chunks["pmcid"] != ""
    dropped = (~keep).sum()
    chunks = chunks.loc[keep].reset_index(drop=True)
    log(f"[filter] dropped rows without pmcid: {dropped}  kept: {len(chunks)}  unique_docs: {chunks['pmcid'].nunique()}")

    assign = pd.read_csv(assign_path)
    if args.cluster_col not in assign.columns or args.preview_col not in assign.columns:
        sys.exit(f"assign needs columns: {args.cluster_col}, {args.preview_col}")
    assign = assign.dropna(subset=[args.preview_col]).copy()
    assign[args.cluster_col] = assign[args.cluster_col].astype(int)

    # Model
    log("Loading model…")
    model = load_model(args.model, fp16=args.fp16)

    # Doc-hint embeddings
    if args.doc_hint:
        log("Building doc-hint embeddings…")
        manifest = pd.read_csv(args.doc_hint)
        doc_texts = build_doc_texts(chunks, manifest, args.doc_max_chunks)
        doc_ids = sorted(doc_texts.keys())
        doc_corpus = [doc_texts[pmc] for pmc in doc_ids]
        doc_emb = encode_texts(model, doc_corpus, batch_size=max(64, args.batch//8), device=device, fp16=args.fp16)
    else:
        doc_ids, doc_emb = [], None

    # Encode all chunk texts
    log("Encoding chunk texts…")
    corpus_texts = chunks["text"].astype(str).tolist()
    corpus_emb = encode_texts(model, corpus_texts, batch_size=args.batch, device=device, fp16=args.fp16)

    # Map pmcid -> corpus row indices
    doc_to_rows: Dict[str, np.ndarray] = {}
    for pmc, grp in chunks.reset_index().groupby("pmcid"):
        doc_to_rows[pmc] = grp["index"].values.astype(np.int64)

    # Encode previews and search
    previews = assign[args.preview_col].astype(str).tolist()
    cl_ids = assign[args.cluster_col].astype(int).values
    out_rows: List[Tuple[str,int,int]] = []

    B = args.batch
    topk = int(args.topk)
    min_sim = float(args.min_sim)
    doc_topk = int(args.doc_topk)

    log("Searching…")
    for s in range(0, len(previews), B):
        e = min(s + B, len(previews))
        q_texts = previews[s:e]
        q_emb = encode_texts(model, q_texts, batch_size=max(64, B//4), device=device, fp16=args.fp16)

        if doc_emb is not None and len(doc_ids) > 0:
            dv, di = topk_cosine(q_emb, max(doc_topk, 1), doc_emb)
            for qi in range(q_emb.shape[0]):
                chosen_docs = [doc_ids[j] for j in di[qi].tolist()]
                accept_rows = np.unique(
                    np.concatenate([doc_to_rows[d] for d in chosen_docs if d in doc_to_rows])
                ) if chosen_docs else None

                if accept_rows is None or accept_rows.size == 0:
                    v, idx = topk_cosine(q_emb[qi:qi+1], topk*3, corpus_emb)
                    sims = v[0].numpy(); rows = idx[0].numpy()
                else:
                    import torch
                    sub = corpus_emb.index_select(0, torch.from_numpy(accept_rows).to(corpus_emb.device))
                    v, idx = topk_cosine(q_emb[qi:qi+1], topk*3, sub)
                    sims = v[0].numpy(); rows = accept_rows[idx[0].numpy()]

                cid = int(cl_ids[s+qi])
                for k in range(len(rows)):
                    if float(sims[k]) < min_sim: break
                    row = int(rows[k])
                    pmc = chunks.iloc[row]["pmcid"]
                    cidx = int(chunks.iloc[row]["chunk_index"])
                    out_rows.append((pmc, cidx, cid))
                    break
        else:
            v, idx = topk_cosine(q_emb, topk*3, corpus_emb)
            for qi in range(q_emb.shape[0]):
                cid = int(cl_ids[s+qi])
                for k in range(idx.shape[1]):
                    if float(v[qi, k].item()) < min_sim: break
                    row = int(idx[qi, k].item())
                    pmc = chunks.iloc[row]["pmcid"]
                    cidx = int(chunks.iloc[row]["chunk_index"])
                    out_rows.append((pmc, cidx, cid))
                    break

        if (e % (B*10)) == 0:
            log(f"processed {e}/{len(previews)}  matched={len(out_rows)}")

    if not out_rows:
        sys.exit("No matches found. Lower --min-sim or increase --topk.")

    # Majority vote per (pmcid, chunk_index)
    from collections import Counter, defaultdict
    votes: Dict[Tuple[str,int], Counter] = defaultdict(Counter)
    for pmc,cidx,cid in out_rows:
        votes[(pmc,cidx)][cid] += 1
    pairs, clust = [], []
    for (pmc,cidx), cnt in votes.items():
        best = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        pairs.append((pmc, cidx)); clust.append(best)

    out = pd.DataFrame({"pmcid":[p[0] for p in pairs],
                        "chunk_index":[p[1] for p in pairs],
                        "cluster_id":clust}).astype({"chunk_index":int,"cluster_id":int})
    out = out.sort_values(["pmcid","chunk_index"])
    out.to_csv(args.out, index=False)

    log("=== TORCH GPU Alignment Report ===")
    log(f"assigned rows in: {len(assign)}")
    log(f"unique pairs out: {len(out)}")
    log(f"unique docs:      {out['pmcid'].nunique()}")
    log(f"wrote:            {args.out}")

if __name__ == "__main__":
    main()
