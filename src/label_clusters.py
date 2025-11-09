#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cluster labeling with robust cleaning, reference mapping, and domain-biased TF-IDF.
Inputs
  --chunks pmc_chunker/out/paragraph_chunks_4000.jsonl
  --assign pmc_chunker/out/cluster_assignments_norm.csv  (pmcid,chunk_index,cluster_id)
Outputs
  pmc_chunker/out/cluster_labels.json
  pmc_chunker/out/paragraph_chunks_4000_labeled.jsonl
  pmc_chunker/out/cluster_exemplars.json
LLM modes
  --llm none | hf | openai   (hf default: google/flan-t5-small)
"""
from __future__ import annotations
import argparse, json, os, re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances

# ---------- citation/garbage filters ----------
CITE_BRACKETS = re.compile(r"\[(?:\d+(?:\s*[,\-–]\s*\d+)*|[^\]]{1,10})\]")
PARENS_CITES = re.compile(r"\((?:[^\)]{0,40}\d{1,4}[^\)]{0,40})\)")
PAGE_GARBAGE  = re.compile(r"^\s*page\s+\d+(\s+of\s+\d+)?\s*$", re.I)

CUSTOM_STOP = {
    # generic function words
    "the","and","of","in","to","for","with","from","on","by","as","at","or","an","be","we","our",
    "into","between","over","across","per","without","within","also","using","use","used",
    # boilerplate
    "figure","fig","supplementary","author","manuscript","table","data","study","results","methods",
    "materials","introduction","discussion","conclusion","observed","shown","performed","analysis",
    "analyses","experiment","experiments","copyright","doi","pmid","pmcid",
    # citation tokens
    "et","al","al.","ibid","ref","references","citation","citations",
    # units/common
    "mm","cm","ml","mg","kg","hz","nm","μm","um","min","hr","hrs","sec","s","ms",
    # artifacts seen
    "labele","and","study","currently","there","content","solely"
}

PHRASE_BLACKLIST = {
    "currently there","content solely","et al","et al.","results methods",
    "introduction discussion","citations","citation","references"
}

# domain bias terms
BIOMED_DOMAIN = {
    "zebrafish","xenograft","metastasis","angiogenesis","angiography","perfusion",
    "gfp","fluorescence","confocal","microscopy","tumor","oncology","carcinoma",
    "apoptosis","invasion","migration","genotype","phenotype","knockdown","knockout",
    "mrna","rna","dna","crispr","western","immunohistochemistry","metastatic","melanoma",
    "colorectal","breast","gastric","leukemia","cytotoxicity","chemotherapy","microenvironment",
    "vasculature","endothelial","embryo","larvae","danio","rerio","metastases","angiogenic"
}

TOKEN_PATTERN = r"(?u)\b[a-zA-Z][a-zA-Z\-]{3,}\b"  # words ≥4 letters

# ---------------- text utils ----------------
def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    t = CITE_BRACKETS.sub(" ", s)
    t = PARENS_CITES.sub(" ", t)
    t = re.sub(r"\bet\s+al\.?\b", " ", t, flags=re.I)
    t = re.sub(r"\b[Aa]uthor\s+[Mm]anuscript\b", " ", t)
    return " ".join(t.split())

def drop_bad_row(section: str, typ: str, subsection: str, text: str) -> bool:
    s = (section or "").strip().lower()
    if s.startswith("references") or (typ or "") == "reference_entry":
        return True
    if PAGE_GARBAGE.match(subsection or ""):
        return True
    return not bool(text and text.strip())

def canon_label(s: str) -> str:
    s = (s or "").lower().strip().replace("-", " ").replace("_", " ")
    return " ".join(w for w in s.split() if w not in CUSTOM_STOP)

def jaccard(a: str, b: str) -> float:
    A = set(canon_label(a).split()); B = set(canon_label(b).split())
    if not A or not B: return 0.0
    return len(A & B) / len(A | B)

def dedup_words(label: str) -> str:
    seen, out = set(), []
    for w in label.split():
        if w not in seen:
            seen.add(w); out.append(w)
    return " ".join(out)

def is_bad_phrase(label: str) -> bool:
    l = label.strip().lower()
    return (not l) or (l in PHRASE_BLACKLIST) or (len(l) < 3) or all(not ch.isalpha() for ch in l)

def postprocess_label(raw: str) -> str:
    toks = re.findall(TOKEN_PATTERN, (raw or "").lower())
    toks = [t for t in toks if t not in CUSTOM_STOP]
    if not toks:
        return "other"
    label = dedup_words(" ".join(toks[:2]))
    return "other" if is_bad_phrase(label) else label

# --------------- LLM hooks ---------------
def label_with_openai(texts: List[str], model: str = "gpt-4o-mini") -> str:
    try:
        from openai import OpenAI
    except Exception as e:
        raise ImportError("openai package not available") from e
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    client = OpenAI(api_key=api_key)
    samples = "\n\n".join(f"- {normalize_text(t)[:400]}" for t in texts if t)[:2500]
    prompt = ("Give a biomedical topic label in one or two words. "
              "Letters only. No citations. No filler phrases.\n\n"+samples)
    rsp = client.chat.completions.create(
        model=model, messages=[{"role":"user","content":prompt}],
        temperature=0.0, max_tokens=6
    )
    return postprocess_label(rsp.choices[0].message.content)

def label_with_hf(texts: List[str], model_name: str = "google/flan-t5-small",
                  device: str | None = None, max_new_tokens: int = 6) -> str:
    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
        import torch
    except Exception as e:
        raise ImportError("transformers/torch not installed") from e
    if device is None:
        if torch.cuda.is_available(): device = 0
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(): device = "mps"
        else: device = -1
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    gen = pipeline(task="text2text-generation", model=mdl, tokenizer=tok, device=device)
    samples = "\n\n".join(f"- {normalize_text(t)[:400]}" for t in texts if t)[:2500]
    prompt = ("Return a biomedical topic label in one or two words. "
              "Letters only. No citations. No filler phrases.\n\n"+samples)
    out = gen(prompt, max_new_tokens=max_new_tokens, do_sample=False, truncation=True)
    if isinstance(out, list):
        text = out[0].get("generated_text") or out[0].get("summary_text", "")
    else:
        text = str(out)
    return postprocess_label(text)

# -------- safe exemplar selection --------
def _tfidf(texts: List[str], **kwargs):
    vec = TfidfVectorizer(**kwargs)
    X = vec.fit_transform(texts)
    return vec, X

def select_exemplars_tfidf(group_df: pd.DataFrame, k: int = 4) -> List[int]:
    import numpy as np
    sub = group_df[~group_df["_drop"]].copy()
    if sub.empty:
        sub = group_df.copy()
    texts = [normalize_text(t) for t in sub["text"].tolist()]

    for attempt in [
        dict(max_df=0.6, min_df=3, token_pattern=TOKEN_PATTERN, ngram_range=(1,2), stop_words="english", sublinear_tf=True),
        dict(max_df=0.8, min_df=2, token_pattern=TOKEN_PATTERN, ngram_range=(1,1), stop_words="english", sublinear_tf=True),
        dict(max_df=1.0, min_df=1, token_pattern=r"(?u)\b\w\w+\b", ngram_range=(1,1), stop_words="english", sublinear_tf=True),
    ]:
        try:
            _, X = _tfidf(texts, **attempt)
            if X.shape[0] == 0 or X.shape[1] == 0:
                raise ValueError("empty tfidf")
            centroid = X.mean(axis=0)
            centroid = centroid.A if hasattr(centroid, "A") else centroid
            if getattr(centroid, "ndim", 1) == 1:
                import numpy as np
                centroid = np.asarray(centroid)[None, :]
            d = cosine_distances(X, centroid).ravel()
            order = d.argsort()[:min(k, X.shape[0])]
            return sub.reset_index().iloc[order]["index"].tolist()
        except Exception:
            continue

    # fallback: longest texts
    return sub["text"].fillna("").str.len().nlargest(min(k, len(sub))).index.tolist()

# ---------------- IO helpers ----------------
def load_inputs(chunks_path: Path, assign_path: Path,
                pmcid_col: str, chunk_col: str, cluster_col: str) -> pd.DataFrame:
    chunks = pd.read_json(chunks_path, lines=True)
    for req in ("text", pmcid_col, chunk_col, "section", "type", "subsection"):
        if req not in chunks.columns:
            raise ValueError(f"chunks JSONL missing '{req}'")
    assign = pd.read_csv(assign_path, low_memory=False, dtype=str)
    for col in (pmcid_col, chunk_col, cluster_col):
        if col not in assign.columns:
            raise ValueError(f"assign CSV missing '{col}'")
    assign[pmcid_col] = assign[pmcid_col].astype(str)
    assign[chunk_col] = assign[chunk_col].astype(str).str.extract(r"(\d+)").astype(int)
    assign[cluster_col] = assign[cluster_col].astype(str).str.extract(r"(\d+)").astype(int)

    merged = pd.merge(
        chunks, assign[[pmcid_col, chunk_col, cluster_col]],
        on=[pmcid_col, chunk_col], how="inner", validate="many_to_one"
    )
    merged["_drop"] = merged.apply(
        lambda r: drop_bad_row(r.get("section"), r.get("type"), r.get("subsection"), r.get("text")), axis=1
    )
    return merged

def write_cluster_labels(out_dir: Path, labels: Dict[int, str]) -> Path:
    out = out_dir / "cluster_labels.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in labels.items()}, f, ensure_ascii=False, indent=2)
    return out

def write_labeled_jsonl(out_dir: Path, df: pd.DataFrame,
                        pmcid_col: str, chunk_col: str,
                        cluster_col: str, label_col: str) -> Path:
    out = out_dir / "paragraph_chunks_4000_labeled.jsonl"
    cols = list(df.columns)
    with out.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            rec = {c: row[c] for c in cols}
            rec["cluster_id"] = int(row[cluster_col])
            rec["cluster_label"] = dedup_words(str(row.get(label_col, "other")))
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return out

def merge_near_duplicates(cluster_labels: Dict[int, str], threshold: float = 0.6) -> Dict[int, str]:
    items = list(cluster_labels.items())
    canonical: Dict[str, str] = {}
    for cid, lab in items:
        best_key, best_sim = None, 0.0
        for key, val in list(canonical.items()):
            sim = jaccard(val, lab)
            if sim > best_sim:
                best_sim, best_key = sim, key
        if not (best_key is not None and best_sim >= threshold):
            ck = canon_label(lab) or "other"
            canonical[ck] = lab
    remapped = {}
    for cid, lab in cluster_labels.items():
        ck = canon_label(lab) or "other"
        remapped[cid] = canonical.get(ck, lab)
    return remapped

# -------- label chooser --------
def tfidf_label(texts: List[str], top_k: int = 2) -> str:
    texts = [normalize_text(t) for t in texts if isinstance(t, str) and t.strip()]
    if not texts:
        return "other"

    try:
        vec = TfidfVectorizer(
            max_df=0.9, min_df=1,
            stop_words=list(CUSTOM_STOP) + ["english"],
            token_pattern=TOKEN_PATTERN, ngram_range=(1, 2),
            sublinear_tf=True, smooth_idf=True,
        )
        X = vec.fit_transform(texts)
    except ValueError:
        return "other"
    except Exception:
        return "other"

    if X.shape[0] == 0 or X.shape[1] == 0:
        return "other"

    try:
        scores = X.sum(axis=0).A1
        vocab = vec.get_feature_names_out()
    except Exception:
        return "other"

    order = scores.argsort()[::-1]
    candidates: List[Tuple[str, float]] = []
    for i in order:
        term = vocab[i].lower()
        if any(ch.isdigit() for ch in term):
            continue
        if term in CUSTOM_STOP or term in PHRASE_BLACKLIST:
            continue
        if not re.fullmatch(r"[a-z\-]{3,}(?:\s[a-z\-]{3,})?", term):
            continue
        words = term.split()
        boost = sum(1 for w in words if w in BIOMED_DOMAIN)
        candidates.append((term, scores[i] + 0.5 * boost))

    if not candidates:
        return "other"

    candidates.sort(key=lambda x: x[1], reverse=True)
    chosen = []
    for term, _ in candidates:
        for w in term.split():
            if w not in chosen:
                chosen.append(w)
            if len(chosen) >= top_k:
                break
        if len(chosen) >= top_k:
            break
    label = dedup_words(" ".join(chosen))
    return "other" if is_bad_phrase(label) else label

# -------- reference mapping --------
def majority_section_label(group_df: pd.DataFrame) -> str | None:
    # If a cluster is mostly references or dropped rows, force 'references'
    sec_counts = group_df["section"].fillna("").str.lower().value_counts()
    dropped_ratio = group_df["_drop"].mean()
    if (sec_counts.index[:1].tolist() == ["references"]) or dropped_ratio >= 0.6:
        return "references"
    return None

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--assign", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pmcid-col", default="pmcid")
    ap.add_argument("--chunk-col", default="chunk_index")
    ap.add_argument("--cluster-col", default="cluster_id")
    ap.add_argument("--exemplars", type=int, default=4)
    ap.add_argument("--llm", choices=["none","hf","openai"], default="none")
    ap.add_argument("--hf-model", default="google/flan-t5-small")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--merge-threshold", type=float, default=0.6)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df = load_inputs(Path(args.chunks), Path(args.assign),
                     args.pmcid_col, args.chunk_col, args.cluster_col)

    labels: Dict[int, str] = {}
    exemplars_debug: Dict[int, List[Tuple[str, int]]] = {}

    for cid, g in df.groupby(args.cluster_col, sort=True):
        g = g.reset_index(drop=True)
        idxs = select_exemplars_tfidf(g, k=args.exemplars)
        texts = [g.loc[i, "text"] for i in idxs]
        exemplars_debug[int(cid)] = [(g.loc[i, "text"][:200], int(i)) for i in idxs]

        # rule 1: reference-heavy cluster → 'references'
        forced = majority_section_label(g)
        label = None
        if forced is None:
            if args.llm == "hf":
                try: label = label_with_hf(texts, model_name=args.hf_model)
                except Exception: label = None
            elif args.llm == "openai":
                try: label = label_with_openai(texts, model=args.model)
                except Exception: label = None
            if not label or is_bad_phrase(label):
                label = tfidf_label(texts, top_k=2)
            label = postprocess_label(label)
        else:
            label = forced

        if label in PHRASE_BLACKLIST or label in {"citation","citations"}:
            label = "references"

        labels[int(cid)] = label if not is_bad_phrase(label) else "other"

    labels = merge_near_duplicates(labels, threshold=args.merge_threshold)
    df["cluster_label"] = df[args.cluster_col].astype(int).map(labels).fillna("other")

    labels_path = write_cluster_labels(out_dir, labels)
    jsonl_path = write_labeled_jsonl(out_dir, df, args.pmcid_col, args.chunk_col, args.cluster_col, "cluster_label")

    with (out_dir / "cluster_exemplars.json").open("w", encoding="utf-8") as f:
        json.dump({str(cid): [{"text": t, "group_row": i} for (t, i) in exemplars_debug[cid]] for cid in exemplars_debug},
                  f, ensure_ascii=False, indent=2)

    print(f"Wrote: {labels_path}")
    print(f"Wrote: {jsonl_path}")
    print(f"Wrote: {out_dir / 'cluster_exemplars.json'}")

if __name__ == "__main__":
    main()
