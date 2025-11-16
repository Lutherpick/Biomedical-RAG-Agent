#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embed a JSONL of chunks/paragraphs into a .npy matrix using sentence-transformers.

Inputs
------
  --inp         Path to JSONL with one object per line.
                Each line should contain the text to embed in either:
                  * 'text'        (preferred)
                  * 'chunk_text'  (fallback)
                or a custom field given by --text-field.
  --text-field  Field name for text (default: text).
                If that field is missing or empty, the script will try
                'text' then 'chunk_text' automatically.
  --model       Sentence-Transformers model name
                (default: sentence-transformers/all-MiniLM-L6-v2, 384-dim)
  --batch       Encode batch size (default: 1024)
  --fp16        Save embeddings as float16 instead of float32
  --normalize   L2-normalize embeddings before saving (good for cosine)
  --out         Output .npy path (required)
  --ids-out     Optional CSV with row ids and metadata (id, pmcid, chunk_index)

Typical usage for your current pipeline
---------------------------------------
  python .\src\embed_chunks.py ^
    --inp .\src\pmc_chunker\out\4_5_labeling\paragraph_chunks_4000_labeled.jsonl ^
    --out .\src\pmc_chunker\out\embeddings_v5.npy ^
    --batch 1024 --fp16 --normalize
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--inp", required=True, help="Input JSONL with chunks/paragraphs")
    p.add_argument("--out", required=True, help="Output .npy file for embeddings")
    p.add_argument("--ids-out", help="Optional CSV with (row,id,pmcid,chunk_index)")
    p.add_argument(
        "--text-field",
        default="text",
        help="Primary field name for text; falls back to 'text' then 'chunk_text'",
    )
    p.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-Transformers model name",
    )
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--fp16", action="store_true", help="Store embeddings as float16")
    p.add_argument(
        "--normalize",
        action="store_true",
        help="L2-normalize embeddings (recommended for cosine)",
    )
    return p.parse_args()


def iter_jsonl(path: Path):
    """Yield (row_index, json_obj) for each line, keeping index even on errors."""
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                yield i, {}
                continue
            try:
                yield i, json.loads(line)
            except Exception:
                # keep alignment even if a line is bad
                yield i, {}


def main():
    args = parse_args()

    # Lazy imports so that --help is fast
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    from tqdm import tqdm

    # Pick device: cuda if available, otherwise cpu
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    print(f"[INFO] Loading model {args.model} on {device} ...")
    model = SentenceTransformer(args.model, device=device)

    rows_meta = []
    texts = []

    primary_field = args.text_field

    def extract_text(obj):
        # Try: primary field -> 'text' -> 'chunk_text'
        txt = obj.get(primary_field)
        if (txt is None or txt == "") and primary_field != "text":
            txt = obj.get("text")
        if (txt is None or txt == "") and primary_field != "chunk_text":
            txt = obj.get("chunk_text")
        if txt is None:
            txt = ""
        return str(txt)

    inp_path = Path(args.inp)
    if not inp_path.is_file():
        print(f"[ERROR] Input JSONL not found: {inp_path}", file=sys.stderr)
        sys.exit(2)

    # Read all rows and collect metadata
    for i, obj in iter_jsonl(inp_path):
        pmcid = obj.get("pmcid")
        cidx = obj.get("chunk_index")
        # Stable row id: prefer explicit id, else pmcid:chunk_index, else index
        rid = obj.get("id") or (
            f"{pmcid}:{cidx}" if pmcid is not None and cidx is not None else str(i)
        )

        txt = extract_text(obj)
        texts.append(txt)
        rows_meta.append({"row": i, "id": rid, "pmcid": pmcid, "chunk_index": cidx})

    n = len(texts)
    if n == 0:
        print("[ERROR] No rows found in JSONL.", file=sys.stderr)
        sys.exit(2)

    print(f"[INFO] Rows: {n}")
    print(f"[INFO] Batch: {args.batch}  fp16={args.fp16}  normalize={args.normalize}")

    all_vecs = []
    total_batches = math.ceil(n / args.batch)

    for b in tqdm(range(total_batches), desc="Encoding"):
        s = b * args.batch
        e = min(n, s + args.batch)
        batch_texts = texts[s:e]

        emb = model.encode(
            batch_texts,
            batch_size=args.batch,
            convert_to_numpy=True,
            normalize_embeddings=args.normalize,
            show_progress_bar=False,
        )

        if args.fp16:
            emb = emb.astype("float16", copy=False)
        else:
            emb = emb.astype("float32", copy=False)

        all_vecs.append(emb)

    X = np.vstack(all_vecs)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, X)
    print(f"[OK] Wrote embeddings: {args.out}  shape={X.shape}  dtype={X.dtype}")

    if args.ids_out:
        with Path(args.ids_out).open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["row", "id", "pmcid", "chunk_index"])
            w.writeheader()
            w.writerows(rows_meta)
        print(f"[OK] Wrote ids: {args.ids_out}")


if __name__ == "__main__":
    main()
