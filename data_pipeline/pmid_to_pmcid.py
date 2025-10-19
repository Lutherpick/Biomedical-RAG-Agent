# data_pipeline/pmid_to_pmcid.py
from __future__ import annotations
from pathlib import Path
from Bio import Entrez
import os, json, time, sys, random

# --- NCBI etiquette ---
Entrez.email   = "bazefi01@thu.de"
Entrez.tool    = "biomed-rag-ingester"
Entrez.api_key = os.getenv("NCBI_API_KEY")  # optional but recommended

# --- Paths ---
ROOT = Path(__file__).resolve().parent
IN   = ROOT / "pubmed_120k.jsonl"  # produced by search_pubmed_sliced.py
OUT_JSONL = ROOT / "data" / "processed" / "pmid_pmcid_map_120k.jsonl"
OUT_JSON  = ROOT / "data" / "processed" / "pmid_pmcid_map_120k.json"

OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

BATCH = 200
SLEEP_BASE = 0.34  # polite pacing

def read_pmids(path: Path) -> list[str]:
    pmids = []
    with path.open(encoding="utf-8") as f:
        for ln in f:
            try:
                rec = json.loads(ln)
                v = str(rec.get("pmid")).strip()
                if v:
                    pmids.append(v)
            except Exception:
                continue
    return pmids

def load_existing(jsonl_path: Path) -> dict[str, str]:
    mapping = {}
    if jsonl_path.exists():
        with jsonl_path.open(encoding="utf-8") as f:
            for ln in f:
                try:
                    r = json.loads(ln)
                    p, c = r.get("pmid"), r.get("pmcid")
                    if p and c:
                        mapping[str(p)] = str(c)
                except Exception:
                    pass
    return mapping

def elink_with_retry(ids: list[str], retries: int = 6):
    delay = 0.5
    for attempt in range(1, retries + 1):
        try:
            with Entrez.elink(dbfrom="pubmed", db="pmc", id=",".join(ids), retmode="xml") as h:
                return Entrez.read(h)
        except Exception as e:
            print(f"[elink] attempt {attempt}/{retries} failed for {len(ids)} ids: {e}", file=sys.stderr)
            time.sleep(delay + random.uniform(0, 0.5))
            delay = min(delay * 2, 8)
    print("[elink] giving up on current batch", file=sys.stderr)
    return None

def extract_links(elink_data) -> dict[str, str]:
    """
    Return {pmid: PMCID} for links found. If multiple PMCIDs, prefer the first.
    """
    mapping = {}
    if not elink_data:
        return mapping
    for ls in elink_data:
        pid = None
        try:
            idlist = ls.get("IdList", [])
            pid = str(idlist[0]) if idlist else None
        except Exception:
            pid = None
        if not pid:
            continue
        pmcids = []
        for db in ls.get("LinkSetDb", []) or []:
            if db.get("DbTo") == "pmc":
                for link in db.get("Link", []) or []:
                    # ELink returns numeric PMCID without 'PMC' prefix
                    pmcids.append("PMC" + str(link.get("Id")))
        if pmcids:
            mapping[pid] = pmcids[0]
    return mapping

def main():
    if not IN.exists():
        print(f"[error] Input not found: {IN}", file=sys.stderr)
        sys.exit(1)

    pmids_all = read_pmids(IN)
    print(f"[input] PMIDs total in {IN}: {len(pmids_all)}", file=sys.stderr)

    existing = load_existing(OUT_JSONL)
    print(f"[resume] existing mappings found: {len(existing)} ({OUT_JSONL})", file=sys.stderr)

    # Filter out already mapped PMIDs
    remaining = [p for p in pmids_all if p not in existing]
    print(f"[work] remaining to map: {len(remaining)}", file=sys.stderr)

    # Stream output (append mode)
    out_jsonl_mode = "a" if OUT_JSONL.exists() else "w"
    with OUT_JSONL.open(out_jsonl_mode, encoding="utf-8") as outjl:
        for i in range(0, len(remaining), BATCH):
            chunk = remaining[i:i + BATCH]
            data = elink_with_retry(chunk)
            mapping = extract_links(data)

            # Write any new pairs
            wrote = 0
            for pmid in chunk:
                pmcid = mapping.get(pmid)
                if pmcid:
                    if pmid not in existing:
                        outjl.write(json.dumps({"pmid": pmid, "pmcid": pmcid}) + "\n")
                        existing[pmid] = pmcid
                        wrote += 1

            print(f"[progress] {i + len(chunk):>5}/{len(remaining)}  (+{wrote} new)  total={len(existing)}", file=sys.stderr)
            time.sleep(SLEEP_BASE)  # politeness

    # Emit consolidated JSON map for convenience
    OUT_JSON.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"[done] PMC links: {len(existing)} saved → {OUT_JSONL}")
    print(f"[done] Consolidated map → {OUT_JSON}")

if __name__ == "__main__":
    main()
