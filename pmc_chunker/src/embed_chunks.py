#!/usr/bin/env python3
from __future__ import annotations
import os, json, argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm

def resolve_chunks(cli_inp: str | None) -> Path:
    if cli_inp:
        p = Path(cli_inp); assert p.exists(), f"missing {p}"; return p
    from parse_chunk import OUT_JSON  # 4k scope
    p = Path(OUT_JSON); assert p.exists(), f"missing {p}"; return p

def stream_chunks(p: Path):
    with p.open(encoding="utf-8") as f:
        for line in f: yield json.loads(line)

def pick_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", help="override path to chunks.jsonl")
    ap.add_argument("--batch", type=int, default=int(os.getenv("EMB_BATCH","384")))
    ap.add_argument("--model", default=os.getenv("EMB_MODEL","sentence-transformers/all-MiniLM-L6-v2"))
    ap.add_argument("--dtype", choices=["fp16","fp32"], default=os.getenv("EMB_DTYPE","fp16"))
    args = ap.parse_args()

    inp = resolve_chunks(args.inp)
    outdir = inp.parent; outdir.mkdir(parents=True, exist_ok=True)
    oute = outdir / "embeddings.npy"
    outm = outdir / "emb_meta.jsonl"

    device = pick_device()
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model, device=device)

    texts, metas = [], []
    for r in tqdm(stream_chunks(inp), desc="[load]"):
        texts.append(r["text"])
        metas.append({"pmcid": r.get("pmcid"),
                      "chunk_id": f"{r.get('pmcid')}:{r.get('chunk_index')}"})

    embs = []
    for i in tqdm(range(0, len(texts), args.batch), desc=f"[embed:{device}]"):
        batch = texts[i:i+args.batch]
        embs.append(model.encode(batch, normalize_embeddings=True))
    X = np.concatenate(embs, axis=0)
    X = X.astype("float16" if args.dtype == "fp16" else "float32")
    np.save(oute, X)
    with outm.open("w", encoding="utf-8") as w:
        for m in metas: w.write(json.dumps(m) + "\n")
    print(f"[done] {X.shape} -> {oute} ; meta -> {outm}")

if __name__ == "__main__":
    main()
