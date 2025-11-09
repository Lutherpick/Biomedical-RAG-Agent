# cluster.py — emits pmcid, chunk_index, cluster_id (plus preview)
import os, json, gc, time, math
from pathlib import Path
from typing import Tuple, Callable

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans, KMeans

# ======= CONFIG =======
CHUNKS_PATH     = r"C:\Users\Administrator\Projects\TEAMPROJECT\biomed-rag-agent\src\pmc_chunker\out\paragraph_chunks_4000_merged.jsonl"
MODEL_NAME      = "BAAI/bge-small-en-v1.5"
BATCH_SIZE      = 256
NORMALIZE_EMB   = True
EMB_DTYPE       = np.float32

TARGET_SIZE       = 400     # aim ~80 items per cluster
MAX_SIZE          = 120    # hard cap before split
MAX_SPLIT_PASSES  = 2

PCA_DIM         = 100
RANDOM_STATE    = 42

MBK_BATCH       = 16384
MAX_ITER        = 300
N_INIT          = 20

RUN_TAG   = time.strftime("%Y%m%d-%H%M%S")
OUT_DIR   = f"./workspace/out_balanced_{RUN_TAG}"
os.makedirs(OUT_DIR, exist_ok=True)
ASSIGN_CSV      = f"{OUT_DIR}/cluster_assignments.csv"
EMB_MEMMAP      = f"{OUT_DIR}/embeddings.dat"
PCA_EMB_MEMMAP  = f"{OUT_DIR}/embeddings_pca.dat"

print("OUT_DIR:", Path(OUT_DIR).resolve())
print("CUDA available:", torch.cuda.is_available(), "| Device:", (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"))

# Candidate text keys to auto-detect if needed
CANDIDATES_DIRECT = ["text","page_content","content","chunk","chunk_text","text_content","body"]
CANDIDATES_NESTED = ["chunk","data","payload","meta","attributes"]

def pick_text_getter(path: str) -> Tuple[Callable[[dict], str], str]:
    import collections
    hits, nhits = collections.Counter(), collections.Counter()
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if i > 4000: break
            s=line.strip()
            if not s: continue
            try:
                obj=json.loads(s)
            except:
                continue
            found=False
            for k in CANDIDATES_DIRECT:
                v=obj.get(k)
                if isinstance(v,str) and v.strip():
                    hits[k]+=1; found=True; break
            if not found:
                for outer in CANDIDATES_NESTED:
                    inner=obj.get(outer)
                    if isinstance(inner,dict):
                        for k in CANDIDATES_DIRECT:
                            v=inner.get(k)
                            if isinstance(v,str) and v.strip():
                                nhits[(outer,k)]+=1; found=True; break
                    if found: break
    if hits:
        key,_=hits.most_common(1)[0]
        return (lambda o: (o.get(key) if isinstance(o.get(key),str) and o.get(key).strip() else None)), key
    if nhits:
        (outer,k),_=nhits.most_common(1)[0]
        def g(o):
            inner=o.get(outer)
            if isinstance(inner,dict):
                v=inner.get(k)
                if isinstance(v,str) and v.strip(): return v
            return None
        return g, f"{outer}.{k}"
    return (lambda o: (o.get("text") if isinstance(o.get("text"),str) and o.get("text").strip() else None)), "text (fallback)"

def read_jsonl_with_getter(path: str, getter) -> Tuple[list, list, list, list]:
    """Return pmcid_list, chunk_index_list, texts, previews_ids (for sanity)."""
    pmcids, chunk_idxs, texts, local_ids = [], [], [], []
    with open(path,"r",encoding="utf-8") as f:
        for i, line in enumerate(f,1):
            s=line.strip()
            if not s: continue
            try:
                obj=json.loads(s)
            except:
                continue
            v=getter(obj)
            if not (isinstance(v,str) and v.strip()):
                continue
            pmcid = obj.get("pmcid") or ""
            # if pmcid missing, allow pmid fallback but we still output pmcid column blank
            chunk_index = obj.get("chunk_index")
            # chunk_index must be numeric; fallback to 0-based sequence if absent
            if chunk_index is None:
                chunk_index = i - 1
            try:
                chunk_index = int(chunk_index)
            except:
                # last resort: strip non-digits
                try:
                    chunk_index = int(str(chunk_index).strip())
                except:
                    chunk_index = i - 1

            pmcids.append(str(pmcid))
            chunk_idxs.append(chunk_index)
            texts.append(v)
            # local id is for debugging only
            local_ids.append(obj.get("pmid") or obj.get("nihmsid") or f"row_{i}")
    return pmcids, chunk_idxs, texts, local_ids

p = Path(CHUNKS_PATH)
if not p.exists():
    raise FileNotFoundError(f"CHUNKS_PATH not found: {CHUNKS_PATH}")

getter, detected = pick_text_getter(CHUNKS_PATH)
print("Detected text field:", detected)

pmcids, chunk_idxs, texts, _ids = read_jsonl_with_getter(CHUNKS_PATH, getter)
N = len(texts)
if N == 0:
    raise RuntimeError("No texts found. Check CHUNKS_PATH and text field detection.")
print("Loaded texts:", N)
print("Sample:", texts[0][:160].replace("\n"," "))

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(MODEL_NAME, device=device)
emb_dim = model.get_sentence_embedding_dimension()
emb_map = np.memmap(EMB_MEMMAP, dtype=EMB_DTYPE, mode='w+', shape=(N, emb_dim))
prefix = "passage: " if "e5" in getattr(model,"name_or_path","").lower() else ""

for start in tqdm(range(0, N, BATCH_SIZE), total=(N+BATCH_SIZE-1)//BATCH_SIZE, desc="Embedding"):
    batch = [prefix + t for t in texts[start:start+BATCH_SIZE]]
    arr = model.encode(
        batch,
        batch_size=len(batch),
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=NORMALIZE_EMB
    ).astype(EMB_DTYPE, copy=False)
    emb_map[start:start+len(arr)] = arr

del model; gc.collect()
X = np.memmap(EMB_MEMMAP, dtype=EMB_DTYPE, mode='r', shape=(N, emb_dim))
print("Embeddings:", X.shape)

print("PCA to", PCA_DIM)
pca = PCA(n_components=PCA_DIM, svd_solver="randomized", random_state=RANDOM_STATE)
Xp = pca.fit_transform(X).astype(np.float32, copy=False)
Xp_map = np.memmap(PCA_EMB_MEMMAP, dtype=np.float32, mode='w+', shape=(N, PCA_DIM))
Xp_map[:] = Xp

initial_k = math.ceil(N / TARGET_SIZE)
print(f"Initial k = ceil({N} / {TARGET_SIZE}) = {initial_k}")

mbk = MiniBatchKMeans(
    n_clusters=initial_k,
    random_state=RANDOM_STATE,
    batch_size=min(MBK_BATCH, max(1024, N // 15)),
    max_iter=MAX_ITER,
    init="k-means++",
    n_init=N_INIT,
    verbose=0,
)
labels = mbk.fit_predict(Xp)
print("Initial clustering done. k =", initial_k)

def split_large_clusters(Xlow, labels, max_size, target_size, random_state=42):
    labels = labels.copy()
    unique, counts = np.unique(labels, return_counts=True)
    next_label = labels.max() + 1
    for c, sz in zip(unique, counts):
        if sz <= max_size:
            continue
        idx = np.where(labels == c)[0]
        k_local = max(2, math.ceil(sz / target_size))
        km_local = KMeans(n_clusters=k_local, random_state=random_state, n_init=10, max_iter=200)
        sub = km_local.fit_predict(Xlow[idx])
        sub_ids = np.unique(sub)
        base = sub_ids[0]
        for sid in sub_ids:
            if sid == base:
                labels[idx[sub == sid]] = c
            else:
                labels[idx[sub == sid]] = next_label
                next_label += 1
    uniq = sorted(np.unique(labels))
    remap = {old:new for new,old in enumerate(uniq)}
    return np.vectorize(remap.get)(labels)

print("\nSplit-only size control passes...")
for pass_idx in range(1, MAX_SPLIT_PASSES+1):
    sizes = pd.Series(labels).value_counts()
    big = sizes[sizes > MAX_SIZE]
    print(f"Pass {pass_idx}: clusters={sizes.shape[0]}, >{MAX_SIZE}={big.shape[0]}")
    if big.empty:
        print("No clusters above MAX_SIZE; stop.")
        break
    labels = split_large_clusters(Xp, labels, MAX_SIZE, TARGET_SIZE, random_state=RANDOM_STATE)

# Remap labels to 0..K-1
uniq = sorted(np.unique(labels))
remap = {old:new for new,old in enumerate(uniq)}
final_labels = np.vectorize(remap.get)(labels)

def first_sentence(txt: str, max_len=160) -> str:
    txt = txt.strip().replace("\n"," ")
    for end in [".","?","!"]:
        idx = txt.find(end)
        if idx != -1:
            return txt[:idx+1][:max_len].strip()
    return txt[:max_len].strip()

df = pd.DataFrame({
    "pmcid": pmcids,
    "chunk_index": chunk_idxs,
    "cluster_id": final_labels,
    "preview": [first_sentence(t) for t in texts],
})
df = df.sort_values(["pmcid","chunk_index"]).reset_index(drop=True)
df.to_csv(ASSIGN_CSV, index=False, encoding="utf-8")

print("\nSaved:", Path(ASSIGN_CSV).resolve())
print("Rows:", df.shape[0], "| Clusters:", df.cluster_id.nunique())
