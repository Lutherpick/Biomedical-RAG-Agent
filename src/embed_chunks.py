#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np

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
) -> Tuple[List[str], List[Dict[str, Any]]]:
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

            obj["token_count_embed"] = count_tokens(txt)

            if id_field and id_field in obj:
                obj_id = str(obj[id_field])
            else:
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

    shape_path = out_path.with_suffix(".shape.txt")
    with shape_path.open("w", encoding="utf-8") as w:
        w.write(f"{embeddings.shape[0]} x {embeddings.shape[1]}\n")

# -----------------------
# Path resolver
# -----------------------
def resolve_paths(args) -> Tuple[Path, Path]:
    # If explicit paths given, use them.
    if args.inp and args.out:
        return Path(args.inp), Path(args.out)

    # Else require version and dirs.
    if not args.version:
        raise SystemExit("Provide --version or both --inp and --out.")

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    in_map = {
        "v1": in_dir / "chunks_v1.jsonl",
        "v2": in_dir / "chunks_v2.jsonl",
        "v3": in_dir / "chunks_v3.jsonl",
    }
    out_map = {
        "v1": out_dir / "emb_v1.npy",
        "v2": out_dir / "emb_v2.npy",
        "v3": out_dir / "emb_v3.npy",
    }
    return in_map[args.version], out_map[args.version]

# -----------------------
# Main
# -----------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Embed JSONL chunks to .npy + sidecars")
    # New convenience flags
    p.add_argument("--version", choices=["v1", "v2", "v3"], help="If set, auto-pick input/output filenames")
    p.add_argument("--in-dir", default="src/pmc_chunker/out", help="Directory containing chunks_*.jsonl")
    p.add_argument("--out-dir", default="src/pmc_chunker/out", help="Directory to write emb_*.npy")
    # Backward-compatible explicit paths
    p.add_argument("--inp", help="Explicit input JSONL")
    p.add_argument("--out", help="Explicit output .npy")
    # Model/config
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--normalize", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--text-field", default="chunk_text")
    p.add_argument("--id-field", default=None)
    p.add_argument("--device", default="auto")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    inp, out = resolve_paths(args)
    out.parent.mkdir(parents=True, exist_ok=True)

    texts, meta = _read_jsonl_texts(
        path=inp,
        text_field=args.text_field,
        id_field=args.id_field,
        limit=args.limit,
    )
    if not texts:
        empty = np.zeros((0, 384), dtype=np.float32)
        np.save(out, empty)
        _write_sidecars(out, empty, [])
        print("[embed] 0 rows ->", out)
        return

    model = _load_model(args.model, args.device)

    all_vecs: List[np.ndarray] = []
    bs = max(1, int(args.batch_size))
    for i in tqdm(range(0, len(texts), bs), desc="encoding", unit="rows"):
        chunk = texts[i:i + bs]
        vec = model.encode(
            chunk,
            batch_size=min(bs, 64),
            convert_to_numpy=True,
            normalize_embeddings=args.normalize,
        )
        if vec.dtype != np.float32:
            vec = vec.astype(np.float32, copy=False)
        all_vecs.append(vec)

    embeddings = np.vstack(all_vecs)
    np.save(out, embeddings)
    _write_sidecars(out, embeddings, meta)
    print(f"[embed] {embeddings.shape[0]} rows, dim={embeddings.shape[1]} -> {out}")

if __name__ == "__main__":
    main()
