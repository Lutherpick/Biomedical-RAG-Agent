#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

# Optional progress bar. Falls back to no-op if not installed.
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):  # type: ignore
        return x

# -----------------------
# True token counting
# -----------------------
def _get_tokenizer():
    try:
        import tiktoken  # type: ignore
        enc = tiktoken.get_encoding("cl100k_base")
        return lambda s: enc.encode(s)
    except Exception:
        return lambda s: s.split()

_TOKENIZE = _get_tokenizer()

def count_tokens(text: str) -> int:
    return len(_TOKENIZE(text))

# -----------------------
# Embedding backend
# -----------------------
def _load_model(name: str, device: str):
    from sentence_transformers import SentenceTransformer
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    return SentenceTransformer(name, device=device)



# -----------------------
# IO
# -----------------------
def _read_jsonl_texts(
        path: Path,
        text_field: str,
        id_field: str | None,
        limit: int | None,
) -> tuple[List[str], List[Dict[str, Any]]]:
    texts: List[str] = []
    meta: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            if not line.strip():
                continue
            obj = json.loads(line)
            txt = obj.get(text_field, "")
            if not isinstance(txt, str) or not txt.strip():
                continue

            # token count here for audit; not used by the model
            obj["token_count_embed"] = count_tokens(txt)

            # build a stable id
            if id_field and id_field in obj:
                obj_id = str(obj[id_field])
            else:
                # Try a composite id if available
                pmcid = str(obj.get("pmcid", "NA"))
                ver = str(obj.get("version", "v?"))
                sec = str(obj.get("section_type", "Section"))
                idx = str(obj.get("chunk_index", i))
                obj_id = f"{pmcid}:{ver}:{sec}:{idx}"

            obj["_id"] = obj_id
            texts.append(txt)
            meta.append(obj)
    return texts, meta

def _write_sidecars(out_path: Path, embeddings: np.ndarray, meta: List[Dict[str, Any]]) -> None:
    # .npy main array already handled by caller
    # Write metadata JSONL so rows align with embeddings
    meta_path = out_path.with_suffix(".meta.jsonl")
    with meta_path.open("w", encoding="utf-8") as w:
        for m in meta:
            w.write(json.dumps({
                "_id": m["_id"],
                "pmcid": m.get("pmcid"),
                "version": m.get("version"),
                "section_type": m.get("section_type"),
                "section_title": m.get("section_title"),
                "chunk_index": m.get("chunk_index"),
                "token_count": m.get("token_count"),
                "token_count_embed": m.get("token_count_embed"),
            }, ensure_ascii=False) + "\n")

    # Simple shape file for quick checks
    shape_path = out_path.with_suffix(".shape.txt")
    with shape_path.open("w", encoding="utf-8") as w:
        w.write(f"{embeddings.shape[0]} x {embeddings.shape[1]}\n")

# -----------------------
# Main
# -----------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Embed JSONL chunks to a .npy matrix + .meta.jsonl")
    p.add_argument("--inp", required=True, help="Input JSONL (e.g., src/pmc_chunker/out/chunks_v1.jsonl)")
    p.add_argument("--out", required=True, help="Output .npy path (e.g., src/pmc_chunker/out/emb_v1.npy)")
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2",
                   help="Sentence-Transformers model")
    p.add_argument("--batch-size", type=int, default=1024, help="Encode batch size")
    p.add_argument("--normalize", action="store_true", help="L2-normalize embeddings")
    p.add_argument("--limit", type=int, default=None, help="Optional cap on number of lines")
    p.add_argument("--text-field", default="chunk_text", help="Field name with text")
    p.add_argument("--id-field", default=None, help="Optional field for stable IDs")
    p.add_argument("--device", default="auto", help="cuda | cpu | auto")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    inp = Path(args.inp)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    texts, meta = _read_jsonl_texts(
        path=inp,
        text_field=args.text_field,
        id_field=args.id_field,
        limit=args.limit,
    )
    if not texts:
        # still create empty outputs for reproducibility
        empty = np.zeros((0, 384), dtype=np.float32)
        np.save(out, empty)
        _write_sidecars(out, empty, [])
        print("[embed] 0 rows ->", out)
        return

    # 3) call with device
    model = _load_model(args.model, args.device)

    # Encode in batches to control memory
    all_vecs: List[np.ndarray] = []
    bs = max(1, int(args.batch_size))
    for i in tqdm(range(0, len(texts), bs), desc="encoding", unit="rows"):
        chunk = texts[i:i + bs]
        vec = model.encode(chunk, batch_size=min(bs, 64), convert_to_numpy=True, normalize_embeddings=args.normalize)
        # ensure float32
        if vec.dtype != np.float32:
            vec = vec.astype(np.float32, copy=False)
        all_vecs.append(vec)

    embeddings = np.vstack(all_vecs)
    np.save(out, embeddings)
    _write_sidecars(out, embeddings, meta)

    print(f"[embed] {embeddings.shape[0]} rows, dim={embeddings.shape[1]} -> {out}")

if __name__ == "__main__":
    main()
