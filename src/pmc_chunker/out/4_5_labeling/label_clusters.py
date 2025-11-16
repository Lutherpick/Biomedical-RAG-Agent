#!/usr/bin/env python3
"""
Label clusters of biomedical paragraphs with short topic labels.

Inputs:
- paragraph_chunks_4000_merged.jsonl  (or similar)
- cluster_assignments_*.csv with columns: pmcid, chunk_index, cluster_id

Outputs in --out-dir:
- cluster_labels.json          cluster_id -> label
- cluster_exemplars.json       cluster_id -> [{pmcid, section, text}, ...]
- paragraph_chunks_4000_labeled.jsonl  original chunks + cluster_id + cluster_label
- summary.txt                  compact cluster_id / label listing
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Set

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


# ---------------------------------------------------------------------
# Stopwords and generic tokens
# ---------------------------------------------------------------------

STOPWORDS: Set[str] = {
    # generic English
    "a", "an", "the", "and", "or", "but", "if", "then", "else",
    "for", "with", "without", "between", "among", "of", "in", "on",
    "to", "from", "by", "about", "as", "into", "like", "through",
    "after", "over", "during", "before", "under", "again", "further",
    "than", "once", "here", "there", "when", "where", "why", "how",
    "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same",
    "so", "too", "very", "can", "will", "just", "don", "should",
    "now",
    # verbs / function words that showed up in labels
    "was", "were", "is", "are", "be", "been", "being",
    "have", "has", "had",
    "this", "these", "those",
    "using", "used", "based",
    # very generic biomedical / paper words
    "patient", "patients",
    "subject", "subjects",
    "study", "studies",
    "trial", "trials",
    "group", "groups",
    "cohort", "cohorts",
    "data", "dataset", "datasets",
    "sample", "samples",
    "table", "tables",
    "figure", "figures",
    "result", "results",
    "conclusion", "conclusions",
    "method", "methods",
    "introduction", "discussion",
    "analysis", "analyses",
    "value", "values",
    "level", "levels",
    # generic bio entities
    "cell", "cells",
    "mouse", "mice",
    "rat", "rats",
    "animal", "animals",
    "protein", "proteins",
    "gene", "genes",
    "expression",
}

# tokens we aggressively drop from labels if possible
GENERIC_LABEL_TOKENS: Set[str] = {
    "patient", "patients",
    "subject", "subjects",
    "study", "studies",
    "trial", "trials",
    "group", "groups",
    "data", "dataset", "datasets",
    "sample", "samples",
    "table", "tables",
    "figure", "figures",
    "result", "results",
    "conclusion", "conclusions",
    "method", "methods",
    "analysis", "analyses",
    "value", "values",
    "cell", "cells",
    "mouse", "mice",
    "rat", "rats",
    "animal", "animals",
    "protein", "proteins",
    "gene", "genes",
    "expression",
    "this", "that", "these", "those",
}

NEGATIVE_LABELS = {
    "biomedical topic",
    "biomedical",
    "topic",
    "other",
    "miscellaneous",
    "general",
    "none",
    "null",
    "n/a",
    "unknown",
    "unlabeled",
    "unrelated",
    "various",
    "mixed",
    "misc",
    "multi-topic",
    "multi topic",
    "irrelevant",
    "background",
}


SYS_RULES = """You label clusters of biomedical research paragraphs.

Rules:
- Output ONLY the main biomedical topic as 1-2 words, lowercase.
- Prefer specific entities or concepts over generic words.
- If multiple topics appear, choose the most central biomedical concept.
- Do NOT use words like "topic", "biomedical", "general", "other", "miscellaneous".
- If the texts are about patients with a specific disease, use that disease name.
- If the texts are about a method or measurement, use that method/measurement name.
- No punctuation, no quotes.
"""

PROMPT_TEMPLATE = """{rules}

Below are short text snippets from one cluster.
Summarize the main biomedical topic of these texts in one or two words.

Snippets:
{snips}

Answer with only the label, nothing else.
Label:"""


def make_prompt(snips: List[str]) -> str:
    body_lines = []
    for i, t in enumerate(snips, 1):
        t = t.replace("\n", " ").strip()
        if len(t) > 420:
            t = t[:417] + "..."
        body_lines.append(f"{i}. {t}")
    return PROMPT_TEMPLATE.format(rules=SYS_RULES, snips="\n".join(body_lines))


def sanitize_label(raw: str) -> str:
    """
    Normalize the raw LLM / TF-IDF label to:
    - lowercase
    - at most 2 words
    - drop generic biomedical tokens if possible
    """
    if not raw:
        return "other"
    lab = raw.strip().strip('"').strip("'")
    lab = lab.replace("\n", " ").lower()

    cleaned = []
    for ch in lab:
        if ch.isalnum() or ch in {"-", " "}:
            cleaned.append(ch)
    lab = "".join(cleaned).strip()

    lab = " ".join(lab.split())
    if not lab:
        return "other"

    if lab.startswith("label:"):
        lab = lab[len("label:"):].strip()

    parts = lab.split()
    if not parts:
        return "other"

    # drop generic tokens IF anything more specific remains
    filtered = [p for p in parts if p not in GENERIC_LABEL_TOKENS]
    if filtered:
        parts = filtered

    # enforce max 2 words (project spec 1–2)
    if len(parts) > 2:
        parts = parts[:2]

    lab = " ".join(parts).strip()
    if not lab:
        return "other"

    return lab


def tfidf_label_for_group(df: pd.DataFrame, top_k: int = 3) -> str:
    """
    TF-IDF fallback label if LLM fails.
    Uses an aggressive stopword list to avoid junk like 'was were cells'.
    """
    texts = df["text"].astype(str).tolist()
    if not texts:
        return "other"

    vec = TfidfVectorizer(
        token_pattern=r"(?u)\b[a-zA-Z0-9][a-zA-Z0-9\-]{2,}\b",
        stop_words=list(STOPWORDS),
        ngram_range=(1, 2),
        sublinear_tf=True,
        smooth_idf=True,
        min_df=1,
        max_df=max(1, len(texts)),
    )
    X = vec.fit_transform(texts)
    scores = np.asarray(X.sum(axis=0)).ravel()
    idx = np.argsort(scores)[::-1][:top_k]
    feats = np.array(vec.get_feature_names_out())[idx]
    label = " ".join(feats)
    return sanitize_label(label)


def select_exemplars_mmr(
        texts: List[str],
        centroid_dist: np.ndarray,
        k: int,
        lambda_div: float = 0.35,
) -> List[int]:
    """
    Greedy Maximal Marginal Relevance over TF-IDF features, with 'relevance'
    defined as inverse normalized centroid_distance (if available).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    if not texts:
        return []

    vec = TfidfVectorizer(
        token_pattern=r"(?u)\b[a-zA-Z0-9][a-zA-Z0-9\-]{2,}\b",
        stop_words=list(STOPWORDS),
        ngram_range=(1, 2),
        sublinear_tf=True,
        smooth_idf=True,
        min_df=1,
        max_df=max(1, len(texts)),
    )
    X = vec.fit_transform([t or "" for t in texts])
    X = X.astype(np.float32)
    sims = (X @ X.T).toarray()

    n = len(texts)
    k = min(k, n)

    if centroid_dist is None or len(centroid_dist) != n or np.all(np.isnan(centroid_dist)):
        rel = np.ones(n, dtype=np.float32)
    else:
        d = np.asarray(centroid_dist, dtype=np.float32)
        d = np.nan_to_num(d, nan=d[np.isfinite(d)].mean() if np.isfinite(d).any() else 1.0)
        d = (d - d.min()) / (d.max() - d.min() + 1e-6)
        rel = 1.0 - d

    selected: List[int] = []
    candidates = list(range(n))

    first = int(np.argmax(rel))
    selected.append(first)
    candidates.remove(first)

    while len(selected) < k and candidates:
        best_score, best_i = -1.0, candidates[0]
        for i in candidates:
            div = 1.0 - max(float(sims[i, j]) for j in selected)
            score = lambda_div * div + (1.0 - lambda_div) * float(rel[i])
            if score > best_score:
                best_score, best_i = score, i
        selected.append(best_i)
        candidates.remove(best_i)

    return selected


# ==========================
# HuggingFace LLM labelling
# ==========================


def llm_labels_hf(
        df: pd.DataFrame,
        model_name: str,
        precision: str = "fp16",
        batch_size: int = 8,
        max_new_tokens: int = 8,
        temperature: float = 0.0,
        exemplars: int = 12,
        seed: int = 0,
) -> Tuple[Dict[int, str], Dict[int, List[dict]]]:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

    _ = np.random.default_rng(seed)

    load_kwargs = {"device_map": "auto"}
    is_t5 = "t5" in model_name.lower()
    if is_t5:
        load_kwargs["torch_dtype"] = torch.float16 if precision.lower() in ("fp16", "half") else torch.float32
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **load_kwargs)
    else:
        if precision.lower() == "4bit":
            try:
                from transformers import BitsAndBytesConfig

                load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            except Exception:
                pass
        elif precision.lower() == "8bit":
            try:
                from transformers import BitsAndBytesConfig

                load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            except Exception:
                pass
        else:
            load_kwargs["torch_dtype"] = torch.float16 if precision.lower() in ("fp16", "half") else torch.float32
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    tok = AutoTokenizer.from_pretrained(model_name)
    model.eval()

    labels: Dict[int, str] = {}
    exemplars_out: Dict[int, List[dict]] = {}

    groups = []
    for cid, g in df.groupby("cluster_id", sort=False):
        texts = g["text"].astype(str).tolist()
        cdist = g["centroid_distance"].to_numpy() if "centroid_distance" in g.columns else None
        if len(texts) == 0:
            groups.append((int(cid), make_prompt([]), g))
            continue

        idx = select_exemplars_mmr(texts, cdist, k=min(exemplars, len(texts)))
        snips = [texts[i] for i in idx]
        prompt = make_prompt(snips)
        groups.append((int(cid), prompt, g))

        exemplars_out[int(cid)] = [
            {
                "pmcid": str(g.iloc[i].pmcid),
                "section": str(getattr(g.iloc[i], "section", "")),
                "text": str(texts[i])[:240],
            }
            for i in idx
        ]

    i = 0
    while i < len(groups):
        batch = groups[i: i + batch_size]
        prompts = [p for _, p, _ in batch]

        if is_t5:
            inputs = tok(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
            gen_kwargs = dict(max_new_tokens=max_new_tokens, num_beams=1)
            if temperature > 0:
                gen_kwargs.update(dict(do_sample=True, temperature=float(temperature), top_p=0.9))
            out = model.generate(**inputs, **gen_kwargs)
            texts_out = tok.batch_decode(out, skip_special_tokens=True)
        else:
            inputs = tok(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
            do_sample = temperature > 0
            gen_kwargs = dict(max_new_tokens=max_new_tokens, num_beams=1, pad_token_id=tok.eos_token_id)
            if do_sample:
                gen_kwargs.update(dict(do_sample=True, temperature=float(temperature), top_p=0.9))
            out = model.generate(**inputs, **gen_kwargs)
            texts_out = tok.batch_decode(out, skip_special_tokens=True)

        for j, (cid, _, g) in enumerate(batch):
            lab = sanitize_label(texts_out[j])
            if lab in NEGATIVE_LABELS or lab == "other":
                lab = tfidf_label_for_group(df[df.cluster_id == cid])
            labels[int(cid)] = lab

        i += batch_size

    return labels, exemplars_out


# ==========================
# OpenAI LLM labelling
# ==========================


def llm_labels_openai(
        df: pd.DataFrame,
        model_name: str = "gpt-5-mini",
        max_new_tokens: int = 8,
        temperature: float = 1.0,
        exemplars: int = 12,
        seed: int = 0,
) -> Tuple[Dict[int, str], Dict[int, List[dict]]]:
    """
    Label clusters using OpenAI Chat Completions.

    Expects OPENAI_API_KEY in the environment.
    For GPT-5 / reasoning models, temperature is fixed by the API, so we
    do NOT send a temperature parameter to avoid 400 errors.
    """
    from openai import OpenAI

    _ = np.random.default_rng(seed)
    client = OpenAI()

    labels: Dict[int, str] = {}
    exemplars_out: Dict[int, List[dict]] = {}

    groups: List[Tuple[int, str, pd.DataFrame]] = []
    for cid, g in df.groupby("cluster_id", sort=False):
        texts = g["text"].astype(str).tolist()
        cdist = g["centroid_distance"].to_numpy() if "centroid_distance" in g.columns else None
        if len(texts) == 0:
            groups.append((int(cid), make_prompt([]), g))
            continue

        idx = select_exemplars_mmr(texts, cdist, k=min(exemplars, len(texts)))
        snips = [texts[i] for i in idx]
        prompt = make_prompt(snips)
        groups.append((int(cid), prompt, g))

        exemplars_out[int(cid)] = [
            {
                "pmcid": str(g.iloc[i].pmcid),
                "section": str(getattr(g.iloc[i], "section", "")),
                "text": str(texts[i])[:240],
            }
            for i in idx
        ]

    total = len(groups)
    for idx, (cid, prompt, g) in enumerate(groups, start=1):
        try:
            # Build kwargs so we can conditionally add temperature
            kwargs = dict(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=max_new_tokens,
            )

            # For GPT-5 models, the API only supports the default temperature;
            # do not send a temperature parameter to avoid 400 errors.
            if not model_name.startswith("gpt-5"):
                kwargs["temperature"] = float(temperature)

            print(f"[openai] cluster {cid} ({idx}/{total})", flush=True)
            resp = client.chat.completions.create(**kwargs)
            raw = resp.choices[0].message.content or ""
        except Exception as e:
            print(
                f"[warn] OpenAI labeling failed for cluster {cid}: {e}. "
                f"Falling back to TF-IDF."
            )
            labels[cid] = tfidf_label_for_group(df[df.cluster_id == cid])
            continue

        lab = sanitize_label(raw)
        if lab in NEGATIVE_LABELS or lab == "other":
            lab = tfidf_label_for_group(df[df.cluster_id == cid])
        labels[int(cid)] = lab

    return labels, exemplars_out


# =========
# Pipeline
# =========


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True, help="paragraph_chunks_4000_merged.jsonl")
    ap.add_argument(
        "--assign",
        required=True,
        help="cluster_assignments_*.csv with pmcid,chunk_index,cluster_id",
    )
    ap.add_argument("--out-dir", required=True)

    ap.add_argument("--exemplars", type=int, default=12, help="snippets per cluster")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--llm", choices=["none", "hf", "openai"], default="hf")

    ap.add_argument("--hf-model", default="google/flan-t5-large")
    ap.add_argument("--hf-precision", default="fp16", choices=["fp16", "8bit", "4bit"])
    ap.add_argument("--hf-batch", type=int, default=16)
    ap.add_argument("--hf-max-new", type=int, default=8)
    ap.add_argument("--hf-temp", type=float, default=0.0)

    ap.add_argument("--openai-model", default="gpt-5-mini")
    ap.add_argument("--openai-max-new", type=int, default=8)
    ap.add_argument("--openai-temp", type=float, default=1.0)

    ap.add_argument("--labels", help="existing cluster_labels.json", default=None)
    ap.add_argument(
        "--regex-relabel", help="regex_relabel.json with pattern->label", default=None
    )

    args = ap.parse_args()
    out = Path(args.out-dir) if hasattr(args, "out-dir") else Path(args.out_dir)  # safety in case of typo
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_json(args.chunks, lines=True)
    a = pd.read_csv(args.assign)

    need_chunks = {"pmcid", "chunk_index", "text"}
    if not need_chunks.issubset(df.columns):
        raise SystemExit(f"chunks missing columns: {need_chunks - set(df.columns)}")
    need_assign = {"pmcid", "chunk_index", "cluster_id"}
    if not need_assign.issubset(a.columns):
        raise SystemExit(f"assignments missing columns: {need_assign - set(a.columns)}")

    df["pmcid"] = df["pmcid"].astype(str)
    a["pmcid"] = a["pmcid"].astype(str)
    a["chunk_index"] = a["chunk_index"].astype(int)
    a["cluster_id"] = a["cluster_id"].astype(int)

    m = df.merge(
        a[["pmcid", "chunk_index", "cluster_id"]],
        on=["pmcid", "chunk_index"],
        how="inner",
    )
    print("merge coverage:", len(m), "/", len(df), f"({len(m) / max(1, len(df)):.1%})")
    print("clusters:", m["cluster_id"].nunique())
    missing = sorted(set(df.pmcid.unique()) - set(m.pmcid.unique()))
    if missing:
        print("sample unmapped pmcids:", set(missing[:40]))

    if args.llm == "hf":
        try:
            labels, exemplars = llm_labels_hf(
                m,
                model_name=args.hf_model,
                precision=args.hf_precision,
                batch_size=args.hf_batch,
                max_new_tokens=args.hf_max_new,
                temperature=args.hf_temp,
                exemplars=args.exemplars,
                seed=args.seed,
            )
        except Exception as e:
            print(f"[warn] HF labeling failed: {e}. Falling back to TF-IDF.")
            labels = {}
            exemplars = {}
            for cid, g in m.groupby("cluster_id", sort=False):
                labels[int(cid)] = tfidf_label_for_group(g)
                exemplars[int(cid)] = [
                    {
                        "pmcid": str(r.pmcid),
                        "section": str(getattr(r, "section", "")),
                        "text": str(r.text)[:240],
                    }
                    for _, r in g.sample(
                        min(args.exemplars, len(g)), random_state=args.seed
                    ).iterrows()
                ]
    elif args.llm == "openai":
        try:
            labels, exemplars = llm_labels_openai(
                m,
                model_name=args.openai_model,
                max_new_tokens=args.openai_max_new,
                temperature=args.openai_temp,
                exemplars=args.exemplars,
                seed=args.seed,
            )
        except Exception as e:
            print(f"[warn] OpenAI labeling failed: {e}. Falling back to TF-IDF.")
            labels = {}
            exemplars = {}
            for cid, g in m.groupby("cluster_id", sort=False):
                labels[int(cid)] = tfidf_label_for_group(g)
                exemplars[int(cid)] = [
                    {
                        "pmcid": str(r.pmcid),
                        "section": str(getattr(r, "section", "")),
                        "text": str(r.text)[:240],
                    }
                    for _, r in g.sample(
                        min(args.exemplars, len(g)), random_state=args.seed
                    ).iterrows()
                ]
    else:
        labels = {}
        exemplars = {}
        for cid, g in m.groupby("cluster_id", sort=False):
            labels[int(cid)] = tfidf_label_for_group(g)
            exemplars[int(cid)] = [
                {
                    "pmcid": str(r.pmcid),
                    "section": str(getattr(r, "section", "")),
                    "text": str(r.text)[:240],
                }
                for _, r in g.sample(
                    min(args.exemplars, len(g)), random_state=args.seed
                ).iterrows()
            ]

    if args.labels and Path(args.labels).exists():
        try:
            prev = json.loads(Path(args.labels).read_text(encoding="utf-8"))
            for k, v in prev.items():
                k_int = int(k)
                if k_int in labels:
                    labels[k_int] = v
            print(f"Patched in {len(prev)} prior labels from {args.labels}")
        except Exception as e:
            print(f"[warn] could not patch previous labels: {e}")

    if args.regex_relabel and Path(args.regex_relabel).exists():
        try:
            relabel_map = json.loads(
                Path(args.regex_relabel).read_text(encoding="utf-8")
            )
            import re as _re

            for cid, lab in list(labels.items()):
                txts = m.loc[m.cluster_id == cid, "text"].astype(str).tolist()
                joined = "\n".join(txts)
                for pat, new_lab in relabel_map.items():
                    if _re.search(pat, joined, flags=_re.IGNORECASE):
                        labels[cid] = sanitize_label(new_lab)
                        break
            print(f"Applied regex relabel map from {args.regex_relabel}")
        except Exception as e:
            print(f"[warn] regex relabel failed: {e}")

    labels_path = out / "cluster_labels.json"
    labels_path.write_text(
        json.dumps({int(k): v for k, v in labels.items()}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print("WROTE:", labels_path)

    ex_path = out / "cluster_exemplars.json"
    ex_path.write_text(json.dumps(exemplars, indent=2, ensure_ascii=False), encoding="utf-8")
    print("WROTE:", ex_path)

    m["cluster_label"] = m["cluster_id"].map(labels).fillna("other")

    out_jsonl = out / "paragraph_chunks_4000_labeled.jsonl"
    m.to_json(out_jsonl, orient="records", lines=True, force_ascii=False)
    print("WROTE:", out_jsonl)

    summary = (
        m[["cluster_id", "cluster_label"]]
        .drop_duplicates()
        .sort_values(["cluster_id"])
        .to_string(index=False)
    )
    (out / "summary.txt").write_text(summary, encoding="utf-8")
    print("WROTE:", out / "summary.txt")


if __name__ == "__main__":
    main()
