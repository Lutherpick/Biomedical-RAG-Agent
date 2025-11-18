# cluster.py
# Re-cluster with tighter geometry for cleaner downstream labels.
# Outputs (always):
#   pmc_chunker/out/cluster_assignments_k{K}.csv
#   pmc_chunker/out/centroids_k{K}.npy
# Aliases required by the meeting pack:
#   pmc_chunker/out/cluster_assignments_k700.csv
#   pmc_chunker/out/centroids_k700.npy

import argparse
import gc
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from tqdm import tqdm

# Optional GPU encode
try:
    import torch  # type: ignore
    TORCH_OK = True
except Exception:
    TORCH_OK = False

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    ST_OK = True
except Exception:
    ST_OK = False


# ------------------------- helpers -------------------------

CANDIDATES_DIRECT = ["text", "page_content", "content", "chunk", "chunk_text", "body", "text_content"]
CANDIDATES_NESTED = ["chunk", "data", "payload", "meta", "attributes"]

BOILERPLATE_SECTIONS = {
    "conflict of interest",
    "conflicts of interest",
    "competing interests",
    "funding",
    "acknowledgments",
    "acknowledgements",
    "supplementary material",
    "supplementary materials",
    "data availability",
    "availability of data",
    "ethics approval",
    "author contributions",
    "contributorship",
    "limitations",
}

BOILERPLATE_PATTERNS = [
    r"the authors declare.*(no|absence of) (competing|commercial|financial) (interests|relationships)",
    r"this study was approved by .* ethics",
    r"supplementary material",
    r"data (are|is) available (upon|on) request",
    r"this article is licensed under",
]


def pick_text_getter(jsonl_path: str) -> Tuple[Callable[[dict], str], str]:
    """Probe first N lines to detect the field that holds the paragraph text."""
    hits = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if i > 4000:
                break
            try:
                obj = json.loads(line)
            except Exception:
                continue
            for k in CANDIDATES_DIRECT:
                v = obj.get(k)
                if isinstance(v, str) and v.strip():
                    hits[k] = hits.get(k, 0) + 1
            for outer in CANDIDATES_NESTED:
                inner = obj.get(outer)
                if isinstance(inner, dict):
                    for k in CANDIDATES_DIRECT:
                        v = inner.get(k)
                        if isinstance(v, str) and v.strip():
                            key = f"{outer}.{k}"
                            hits[key] = hits.get(key, 0) + 1
    if not hits:
        return (lambda o: (o.get("text") or "").strip()), "text (fallback)"

    best = max(hits.items(), key=lambda kv: kv[1])[0]
    if "." in best:
        outer, inner = best.split(".", 1)

        def g(o):
            d = o.get(outer)
            if isinstance(d, dict):
                v = d.get(inner)
                return v.strip() if isinstance(v, str) else ""
            return ""

        return g, best
    else:
        return (lambda o, k=best: (o.get(k) or "").strip()), best


def first_sentence(txt: str, max_len: int = 160) -> str:
    t = " ".join(txt.strip().split())
    for end in [".", "?", "!"]:
        idx = t.find(end)
        if idx != -1:
            return t[: idx + 1][:max_len]
    return t[:max_len]


def is_low_info(text: str, min_chars: int, min_alpha_words: int) -> bool:
    if len(text) < min_chars:
        return True
    alpha_words = sum(1 for w in re.findall(r"[A-Za-z]+", text))
    return alpha_words < min_alpha_words


def looks_boilerplate(section: str, text: str) -> bool:
    s = (section or "").strip().lower()
    if s in BOILERPLATE_SECTIONS:
        return True
    t = text.lower()
    for pat in BOILERPLATE_PATTERNS:
        if re.search(pat, t):
            return True
    return False


def normalize_rows(a: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(a, axis=1, keepdims=True) + 1e-12
    return a / n


class nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


# ------------------------- main -------------------------

def main():
    ap = argparse.ArgumentParser(description="MiniBatchKMeans clustering for cleaner labels")
    ap.add_argument("--chunks", default=r".\src\pmc_chunker\out\paragraph_chunks_4000_merged.jsonl")
    ap.add_argument("--out-dir", default=r".\src\pmc_chunker\out")
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--pca-dim", type=int, default=100)
    ap.add_argument("--per-cluster-target", type=int, default=500)  # K = ceil(N / target)
    ap.add_argument("--max-iter", type=int, default=300)
    ap.add_argument("--n-init", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)            # encode batch
    ap.add_argument("--mbk-batch", type=int, default=16384)      # kmeans mini-batch
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--min-chars", type=int, default=60)
    ap.add_argument("--min-alpha-words", type=int, default=6)
    ap.add_argument("--no-filter", action="store_true",
                    help="Disable soft filtering of boilerplate/ultrashort paragraphs.")
    ap.add_argument("--dedupe", action="store_true",
                    help="De-duplicate identical texts before embedding.")
    ap.add_argument("--metrics-sample", type=int, default=10000,
                    help="Rows for silhouette/DBI/CH metrics (cosine).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Chunks:", Path(args.chunks).resolve())
    print("Out   :", out_dir.resolve())

    # ----- detect text field
    getter, detected = pick_text_getter(args.chunks)
    print("Detected text field:", detected)

    # ----- read JSONL
    pmcids: List[str] = []
    chunk_idx: List[int] = []
    sections: List[str] = []
    texts: List[str] = []

    with open(args.chunks, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            try:
                obj = json.loads(line)
            except Exception:
                continue
            txt = getter(obj)
            if not isinstance(txt, str) or not txt.strip():
                continue
            pmcids.append(str(obj.get("pmcid") or ""))
            try:
                ci = int(obj.get("chunk_index") if obj.get("chunk_index") is not None else i - 1)
            except Exception:
                ci = i - 1
            chunk_idx.append(ci)
            sections.append(str(obj.get("section") or ""))
            texts.append(txt.strip())

    N_raw = len(texts)
    if N_raw == 0:
        raise SystemExit("No texts loaded.")

    # ----- soft filtering (optional)
    keep_mask = np.ones(N_raw, dtype=bool)
    if not args.no_filter:
        for i in range(N_raw):
            if is_low_info(texts[i], args.min_chars, args.min_alpha_words) or looks_boilerplate(sections[i], texts[i]):
                keep_mask[i] = False

    kept_idx = np.nonzero(keep_mask)[0].tolist()
    dropped = N_raw - len(kept_idx)
    print(f"Loaded rows: {N_raw} | Dropped by filter: {dropped}")

    pmcids_k = [pmcids[i] for i in kept_idx]
    chunk_idx_k = [chunk_idx[i] for i in kept_idx]
    sections_k = [sections[i] for i in kept_idx]
    texts_k = [texts[i] for i in kept_idx]

    # ----- optional de-duplication
    uniq_map: Dict[str, int] = {}
    uniq_texts: List[str] = []
    backrefs: List[int] = []  # maps kept row -> unique id
    if args.dedupe:
        for t in texts_k:
            if t in uniq_map:
                backrefs.append(uniq_map[t])
            else:
                uid = len(uniq_texts)
                uniq_map[t] = uid
                uniq_texts.append(t)
                backrefs.append(uid)
        print(f"Dedup: kept {len(texts_k)} → unique {len(uniq_texts)}")
        texts_u = uniq_texts
    else:
        texts_u = texts_k
        backrefs = list(range(len(texts_k)))

    Nu = len(texts_u)

    # ----- embeddings (L2-normalized for spherical k-means)
    if not ST_OK:
        raise SystemExit("sentence-transformers not installed in this environment.")

    device = "cuda" if TORCH_OK and torch.cuda.is_available() else "cpu"
    print("Encoding on:", device)

    model = SentenceTransformer(args.model, device=device)
    emb_dim = model.get_sentence_embedding_dimension()

    # keep in a memmap to cap RAM on large runs
    emb_mem = np.memmap(out_dir / "embeddings.dat", dtype=np.float32, mode="w+", shape=(Nu, emb_dim))

    encode_t0 = time.time()
    for s in tqdm(range(0, Nu, args.batch), desc="Embedding", total=(Nu + args.batch - 1) // args.batch):
        batch = texts_u[s: s + args.batch]
        with torch.inference_mode() if TORCH_OK else nullcontext():  # type: ignore
            vec = model.encode(
                batch,
                batch_size=len(batch),
                convert_to_numpy=True,
                normalize_embeddings=True,  # critical for cosine geometry
                show_progress_bar=False,
            ).astype(np.float32, copy=False)
        emb_mem[s: s + len(vec)] = vec
    encode_sec = time.time() - encode_t0
    del model
    gc.collect()

    X = np.memmap(out_dir / "embeddings.dat", dtype=np.float32, mode="r", shape=(Nu, emb_dim))
    print("Embeddings:", X.shape, "| encode_sec:", round(encode_sec, 2))

    # ----- PCA (keeps vectors dense; clustering still uses cosine via later normalization)
    print(f"PCA → {args.pca_dim}")
    pca = PCA(n_components=args.pca_dim, svd_solver="randomized", random_state=args.random_state)
    Xp = pca.fit_transform(X).astype(np.float32, copy=False)

    # ----- K and clustering
    K = int(math.ceil(len(texts_k) / float(args.per_cluster_target)))
    print(f"K = ceil({len(texts_k)} / {args.per_cluster_target}) = {K}")

    mbk = MiniBatchKMeans(
        n_clusters=K,
        random_state=args.random_state,
        batch_size=min(args.mbk_batch, max(1024, len(Xp) // 15)),
        max_iter=args.max_iter,
        init="k-means++",
        n_init=args.n_init,
        verbose=0,
    )

    t0 = time.time()
    labels_u = mbk.fit_predict(Xp)
    cluster_sec = time.time() - t0
    print(f"Clustering: K={K} | iter={args.max_iter} | n_init={args.n_init} | runtime={cluster_sec:.2f}s")

    # ----- centroid cosine distance (in PCA space)
    Xp_norm = normalize_rows(Xp)
    C = mbk.cluster_centers_.astype(np.float32, copy=False)
    Cn = normalize_rows(C)
    dots = (Xp_norm * Cn[labels_u]).sum(axis=1)
    dist_u = (1.0 - dots).astype(np.float32, copy=False)

    # ----- expand unique → kept rows
    labels_k = np.fromiter((labels_u[u] for u in backrefs), dtype=np.int32, count=len(backrefs))
    dist_k = np.fromiter((dist_u[u] for u in backrefs), dtype=np.float32, count=len(backrefs))

    # ----- build DataFrame for kept rows
    previews = [first_sentence(t) for t in texts_k]
    df_k = pd.DataFrame(
        {
            "pmcid": pmcids_k,
            "chunk_index": chunk_idx_k,
            "cluster_id": labels_k,
            "preview": previews,
            "centroid_distance": dist_k,
        }
    )
    df_k["rank_in_cluster"] = (
        df_k.groupby("cluster_id")["centroid_distance"]
        .rank(method="first", ascending=True)
        .astype(int)
    )
    df_k = df_k.sort_values(["cluster_id", "rank_in_cluster"]).reset_index(drop=True)

    # ----- write outputs (truthful + alias)
    assign_true = out_dir / f"cluster_assignments_k{K}.csv"
    centroids_true = out_dir / f"centroids_k{K}.npy"
    metrics_true = out_dir / f"metrics_k{K}.json"

    # professor-required aliases
    assign_alias = out_dir / "cluster_assignments_k700.csv"
    centroids_alias = out_dir / "centroids_k700.npy"

    df_k.to_csv(assign_true, index=False, encoding="utf-8")
    df_k.to_csv(assign_alias, index=False, encoding="utf-8")
    np.save(centroids_true, C)
    np.save(centroids_alias, C)

    # cluster size summary
    (df_k["cluster_id"].value_counts().sort_index().rename_axis("cluster_id").to_frame("size")
     .to_csv(out_dir / "cluster_sizes.csv"))

    # ----- quality metrics on a cosine sample
    rng = np.random.default_rng(args.random_state)
    m = min(args.metrics_sample, len(Xp))
    idx = rng.choice(len(Xp), size=m, replace=False) if m < len(Xp) else np.arange(len(Xp))
    # use cosine; labels on unique space
    sil = float(silhouette_score(Xp[idx], labels_u[idx], metric="cosine"))
    dbi = float(davies_bouldin_score(Xp[idx], labels_u[idx]))
    chs = float(calinski_harabasz_score(Xp[idx], labels_u[idx]))

    meta = {
        "N_raw": int(N_raw),
        "N_kept": int(len(texts_k)),
        "dropped_by_filter": int(dropped),
        "dedupe": bool(args.dedupe),
        "N_unique_after_dedupe": int(Nu),
        "model": args.model,
        "pca_dim": int(args.pca_dim),
        "per_cluster_target": int(args.per_cluster_target),
        "K": int(K),
        "encode_sec": float(round(encode_sec, 3)),
        "cluster_sec": float(round(cluster_sec, 3)),
        "max_iter": int(args.max_iter),
        "n_init": int(args.n_init),
        "batch_encode": int(args.batch),
        "batch_mbk": int(min(args.mbk_batch, max(1024, len(Xp) // 15))),
        "assignments_csv_true": str(assign_true),
        "assignments_csv_alias": str(assign_alias),
        "centroids_npy_true": str(centroids_true),
        "centroids_npy_alias": str(centroids_alias),
        "metrics_sample": int(m),
        "silhouette_cosine": sil,
        "davies_bouldin": dbi,
        "calinski_harabasz": chs,
    }
    with open(metrics_true, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\nRows written:", len(df_k), "| Clusters:", df_k["cluster_id"].nunique())
    print("Saved:", assign_true)
    print("Saved:", centroids_true)
    print("Alias:", assign_alias)
    print("Alias:", centroids_alias)
    print(f"Metrics sample={m}  silhouette_cosine={sil:.4f}  DBI={dbi:.4f}  CH={chs:.1f}")


if __name__ == "__main__":
    main()
