# pmc_chunker/src/build_manifest.py
from __future__ import annotations

import io
import os
import re
import math
import time
import gzip
import random
from pathlib import Path
from typing import List, Dict

import pandas as pd
import requests
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential_jitter
from dateutil.parser import parse as parse_dt
from dotenv import load_dotenv

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
load_dotenv()

OUT_DIR = Path("pmc_chunker/out")
DATA_DIR = Path("pmc_chunker/data")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

YEAR_MIN, YEAR_MAX = 2018, 2025
TARGET_TOTAL = 4000
BATCH_ESUMMARY = 200
SEED = 42

ALLOWED_TYPES = {
    "Journal Article", "Research-Article", "Research Article",
    "Review", "Systematic Review", "Short Report", "Brief Report",
    "Short Communication", "Brief Communication",
}
EXCLUDE_TYPES = {
    "Meta-Analysis", "Case Report", "Case Reports",
    "Editorial", "Letter", "Corrigendum", "Correction",
    "Retracted Publication", "Retraction of Publication", "Retraction",
}

TOPIC_PATTERNS = {
    "Oncology": r"\bonco|cancer|tumou?r|neoplasm|carcinoma|sarcoma|leukemi|lymphom|melanom",
    "Neuroscience": r"\bneuro|brain|cortex|hippocamp|spinal|neur(o|al|on)|synap|gli",
    "Infectious": r"\binfect|viral|virus|bacteri|fung|parasite|pathogen|sars|covid|influenza|hiv",
    "Immunology": r"\bimmune|immun|cytokin|t[- ]?cell|b[- ]?cell|innate|adaptive|antibod|interferon",
    "CardioMetabolic": r"\bcardio|heart|myocard|vascular|endothel|ather|metaboli|obes|diabet|insulin|lipid",
    "GeneticsGenomics": r"\bgenom|genetic|dna|rna|transcript|epigen|crispr|sequenc|variant|snp",
}

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "pmc-chunker/1.6"})

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def assign_topic(title: str) -> str:
    t = (title or "").lower()
    for topic, pat in TOPIC_PATTERNS.items():
        if re.search(pat, t):
            return topic
    return "Misc"

def _normalize_oa_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: c.strip() for c in df.columns})

    # File path
    if "File" not in df.columns:
        for alt in ("File path", "Filepath", "file", "FTP path", "ftp"):
            if alt in df.columns:
                df = df.rename(columns={alt: "File"})
                break
    if "File" not in df.columns:
        df["File"] = ""

    # PMCID
    if "PMCID" not in df.columns:
        for alt in ("Accession ID", "AccessionID", "Accession", "Article Id", "pmcid"):
            if alt in df.columns:
                df = df.rename(columns={alt: "PMCID"})
                break
    if "PMCID" not in df.columns:
        df["PMCID"] = ""

    # PMID
    if "PMID" not in df.columns:
        for alt in ("PubMed ID", "pubmed id", "pmid"):
            if alt in df.columns:
                df = df.rename(columns={alt: "PMID"})
                break
    if "PMID" not in df.columns:
        df["PMID"] = ""

    # License
    if "License" not in df.columns:
        for alt in ("license", "License Type", "license type"):
            if alt in df.columns:
                df = df.rename(columns={alt: "License"})
                break
    if "License" not in df.columns:
        df["License"] = ""

    # Retracted
    if "Retracted" not in df.columns:
        for alt in ("retracted", "IsRetracted", "is_retracted"):
            if alt in df.columns:
                df = df.rename(columns={alt: "Retracted"})
                break
    if "Retracted" not in df.columns:
        df["Retracted"] = ""

    for c in ("File", "PMCID", "PMID", "License", "Retracted"):
        df[c] = df[c].fillna("").astype(str).str.strip()

    # Normalize protocol for later tools
    df["File"] = df["File"].str.replace(r"^ftp://", "https://", regex=True)
    return df

def _read_csv_bytes(url: str, content: bytes) -> pd.DataFrame:
    # CSV or TXT, gz or not
    if url.endswith(".txt"):
        return pd.read_csv(io.BytesIO(content), sep="\t", dtype=str, on_bad_lines="skip")
    if url.endswith(".gz"):
        try:
            with gzip.open(io.BytesIO(content), "rt", encoding="utf-8", errors="replace") as gzf:
                head = gzf.readline(); gzf.seek(0)
                if "\t" in head:
                    return pd.read_csv(gzf, sep="\t", dtype=str, on_bad_lines="skip")
                return pd.read_csv(gzf, dtype=str, on_bad_lines="skip")
        except Exception:
            return pd.read_csv(io.BytesIO(content), dtype=str, on_bad_lines="skip")
    return pd.read_csv(io.BytesIO(content), dtype=str, on_bad_lines="skip")

def _candidate_oa_list_urls() -> List[str]:
    return [
        # Individual article file list (has PMCID, PMID, license, retracted)
        "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_file_list.csv",
        "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_file_list.txt",
        "http://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_file_list.csv",
        "http://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_file_list.txt",
    ]

def _download_oa_lists() -> pd.DataFrame:
    parts = []
    for u in _candidate_oa_list_urls():
        try:
            r = SESSION.get(u, timeout=180)
            r.raise_for_status()
            dfi = _read_csv_bytes(u, r.content)
            dfi = _normalize_oa_columns(dfi)
            parts.append(dfi)
            break  # first success is enough
        except Exception:
            continue
    if not parts:
        raise RuntimeError("Could not download PMC oa_file_list.")
    return pd.concat(parts, ignore_index=True).drop_duplicates(subset=["PMCID", "File"])

def load_or_download_oa(refresh: bool = False) -> pd.DataFrame:
    cached = DATA_DIR / "oa_file_list.csv.gz"
    if cached.exists() and not refresh:
        return pd.read_csv(cached, compression="gzip", dtype=str)
    df = _download_oa_lists()
    df.to_csv(cached, index=False, compression="gzip")
    return df

# ---------------------------------------------------------------------
# ID mapping and metadata
# ---------------------------------------------------------------------
@retry(stop=stop_after_attempt(5), wait=wait_exponential_jitter(1, 6))
def idconv_pmc_to_pmid(pmcids: List[str]) -> Dict[str, str]:
    url = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
    params = {"ids": ",".join(pmcids), "format": "json"}
    key = os.getenv("NCBI_API_KEY")
    if key:
        params["api_key"] = key
    r = SESSION.get(url, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    out = {}
    for rec in data.get("records", []):
        pmcid = (rec.get("pmcid") or rec.get("accid") or "").strip()
        pmid = str(rec.get("pmid") or "").strip()
        if pmcid and pmid:
            out[pmcid] = pmid
    return out

def backfill_missing_pmids(df: pd.DataFrame) -> pd.DataFrame:
    miss = df[df["PMID"].eq("")]["PMCID"].tolist()
    if not miss:
        return df
    for i in range(0, len(miss), 200):
        batch = miss[i:i+200]
        mp = idconv_pmc_to_pmid(batch)
        if mp:
            df.loc[df["PMCID"].isin(mp.keys()), "PMID"] = df["PMCID"].map(mp)
    df["PMID"] = df["PMID"].fillna("").astype(str).str.replace(r"\.0$", "", regex=True)
    return df

@retry(stop=stop_after_attempt(5), wait=wait_exponential_jitter(1, 6))
def esummary_pubmed(pmids: List[str]) -> Dict[str, dict]:
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    params = {"db": "pubmed", "retmode": "json", "id": ",".join(pmids)}
    k = os.getenv("NCBI_API_KEY")
    if k:
        params["api_key"] = k
    r = SESSION.get(url, params=params, timeout=60)
    r.raise_for_status()
    data = r.json().get("result", {})
    return {k: v for k, v in data.items() if k.isdigit()}

def _stratified_sample_pmids(df: pd.DataFrame, cap: int, seed: int = SEED) -> List[str]:
    random.seed(seed)
    if cap <= 0:
        return df["PMID"].tolist()
    # uniform random if no per-year info yet
    picks = df.sample(n=min(cap, len(df)), random_state=seed)["PMID"].tolist()
    return picks

def summarize_incrementally(df_base: pd.DataFrame, first_cap: int = 15000, max_passes: int = 4) -> pd.DataFrame:
    seen: set[str] = set()
    meta_rows: List[dict] = []
    remaining = df_base.copy()

    cap = first_cap
    for p in range(1, max_passes + 1):
        cand_pmids = [pid for pid in _stratified_sample_pmids(remaining, cap) if pid not in seen]
        if not cand_pmids:
            break

        steps = math.ceil(len(cand_pmids) / BATCH_ESUMMARY)
        sleep_s = 0.11 if os.getenv("NCBI_API_KEY") else 0.34

        for start in tqdm(range(0, len(cand_pmids), BATCH_ESUMMARY), total=steps,
                          desc=f"[2.{p}] PubMed ESummary ({len(cand_pmids)} pmids)"):
            chunk = cand_pmids[start:start+BATCH_ESUMMARY]
            meta = esummary_pubmed(chunk)
            for pid, rec in meta.items():
                if pid in seen:
                    continue
                seen.add(pid)
                title = (rec.get("title") or "").strip()
                lang = ",".join(rec.get("lang", []))
                pubtypes = set(rec.get("pubtype", []))
                artids = rec.get("articleids", [])
                doi = next((a.get("value") for a in artids if a.get("idtype") == "doi"), None)
                journal = rec.get("fulljournalname") or rec.get("source") or ""
                date_s = rec.get("epubdate") or rec.get("pubdate") or ""
                try:
                    year = parse_dt(date_s).year
                except Exception:
                    year = None
                meta_rows.append({
                    "PMID": pid, "title": title, "lang": lang,
                    "pubtypes": "|".join(sorted(pubtypes)),
                    "doi": doi, "journal": journal, "year": year
                })
            time.sleep(sleep_s)

        meta_df = pd.DataFrame(meta_rows)
        m = df_base.merge(meta_df, on="PMID", how="inner")

        # Filters after ESummary
        m = m[m["lang"].str.contains("eng", na=False)]

        def allowed(row) -> bool:
            types = set(filter(None, (t.strip() for t in row["pubtypes"].split("|"))))
            if types & EXCLUDE_TYPES:
                return False
            if "meta-analysis" in (row["title"] or "").lower():
                return False
            return bool(types & ALLOWED_TYPES)

        m = m[m.apply(allowed, axis=1)]
        m = m[m["year"].between(YEAR_MIN, YEAR_MAX, inclusive="both")]

        if len(m) >= TARGET_TOTAL:
            return m

        summarized_pmids = set(meta_df["PMID"]) if not meta_df.empty else set()
        remaining = remaining[~remaining["PMID"].isin(summarized_pmids)].copy()
        cap *= 2

    return df_base.merge(pd.DataFrame(meta_rows), on="PMID", how="inner")

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser(description="Build 4k PMC manifest from OA file list + PubMed ESummary.")
    ap.add_argument("--refresh-oa", action="store_true", help="Redownload OA list cache.")
    ap.add_argument("--first-cap", type=int, default=15000, help="Initial ESummary sample size.")
    ap.add_argument("--max-passes", type=int, default=4, help="Max incremental waves.")
    ap.add_argument("--seed", type=int, default=SEED, help="Sampling seed.")
    args = ap.parse_args()

    print("[1] Loading PMC OA master list…")
    df = load_or_download_oa(refresh=args.refresh_oa)
    print(f"[1] OA rows total: {len(df):,}")

    # Keep only tarballs, valid PMCID, not retracted if flag exists
    df = df[df["PMCID"].astype(str).str.len() > 0]
    df = df[df["File"].str.contains(r"\.tar\.gz$", case=False, na=False)]
    if "Retracted" in df.columns:
        df = df[df["Retracted"].str.lower().isin(["", "no", "false", "0"])]
    df = df.sort_values("File").drop_duplicates(subset=["PMCID"], keep="first")
    print(f"[1] After tarball + de-dup filter: {len(df):,}")

    # Ensure PMID
    df["PMID"] = df["PMID"].fillna("").astype(str).str.replace(r"\.0$", "", regex=True)
    before = (df["PMID"] != "").sum()
    df = backfill_missing_pmids(df)
    after = (df["PMID"] != "").sum()
    print(f"[1] PMID present before/after idconv: {before:,}/{after:,}")

    df = df[df["PMID"] != ""]
    if df.empty:
        raise SystemExit("[error] No OA candidates with PMID after backfill.")

    base = df[["PMCID", "PMID", "File", "License"]].rename(columns={"File": "file", "License": "license"})

    # PubMed ESummary waves and post-filters
    m = summarize_incrementally(
        base.rename(columns={"file": "File", "license": "License"}),
        first_cap=args.first_cap,
        max_passes=args.max_passes,
    )
    if m.empty:
        raise SystemExit("[error] PubMed ESummary returned no usable records.")

    # Topic balance and select 4k
    m["topic"] = m["title"].fillna("").apply(assign_topic)
    per_topic = max(1, TARGET_TOTAL // 6)
    picks, misc_left = [], []
    for topic, grp in m.groupby("topic"):
        if topic == "Misc":
            misc_left.append(grp)
        else:
            picks.append(grp.sample(n=min(per_topic, len(grp)), random_state=args.seed))
    selected = pd.concat(picks, ignore_index=True) if picks else pd.DataFrame()
    if len(selected) < TARGET_TOTAL and misc_left:
        need = TARGET_TOTAL - len(selected)
        misc_pool = pd.concat(misc_left, ignore_index=True)
        extra = misc_pool.sample(n=min(need, len(misc_pool)), random_state=args.seed)
        selected = pd.concat([selected, extra], ignore_index=True) if not selected.empty else extra
    selected = selected.head(TARGET_TOTAL)

    if len(selected) < TARGET_TOTAL:
        print(f"[warn] Only {len(selected)} articles after filters. Increase --first-cap or --max-passes.")

    out = selected[
        ["PMCID", "PMID", "doi", "year", "title", "journal", "topic", "File", "License"]
    ].rename(columns={"File": "file", "License": "license"})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "manifest_4000.csv"
    out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[done] {len(out)} rows -> {out_path}")
    print(f"[note] OA cache -> {DATA_DIR/'oa_file_list.csv.gz'}")

if __name__ == "__main__":
    main()
