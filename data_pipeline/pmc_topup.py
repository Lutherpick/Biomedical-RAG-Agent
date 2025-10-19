# data_pipeline/pmc_topup.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List
from lxml import etree as ET
from Bio import Entrez
import os, sys, csv, time, threading

# ---- NCBI identity & limits ----
Entrez.email   = os.getenv("NCBI_EMAIL", "bazefi01@thu.de")
Entrez.tool    = os.getenv("NCBI_TOOL",  "biomed-rag-ingester")
Entrez.api_key = os.getenv("NCBI_API_KEY")

DEFAULT_RPS_WITH_KEY = 10
DEFAULT_RPS_NO_KEY   = 3
MAX_RPS = float(os.getenv("MAX_RPS", str(DEFAULT_RPS_WITH_KEY if Entrez.api_key else DEFAULT_RPS_NO_KEY)))

class TokenBucket:
    def __init__(self, rate: float):
        self.rate = max(0.5, rate); self.tokens = self.rate; self.last = time.perf_counter()
        self.lock = threading.Lock()
    def acquire(self):
        while True:
            with self.lock:
                now = time.perf_counter()
                self.tokens = min(self.rate, self.tokens + (now - self.last)*self.rate)
                self.last = now
                if self.tokens >= 1.0:
                    self.tokens -= 1.0; return
            time.sleep(0.002)

BUCKET = TokenBucket(MAX_RPS)

# ---- Paths ----
ROOT       = Path(__file__).resolve().parents[1]
DATA_ROOT  = ROOT / "data_pipeline" / "data"
PMID_ROOT  = DATA_ROOT / "pubmed_open_access"
RAW_PMCXML = DATA_ROOT / "raw" / "pmc_xml"
MAN_DIR    = DATA_ROOT / "manifests"
MANIFEST   = MAN_DIR / "oa_manifest.csv"
RAW_PMCXML.mkdir(parents=True, exist_ok=True)
PMID_ROOT.mkdir(parents=True, exist_ok=True)
MAN_DIR.mkdir(parents=True, exist_ok=True)

# ---- Helpers ----
COLS = ["pmid","pmcid","doi","title","journal","publication_date","language",
        "url_source","publisher_url","open_access",
        "metadata_file","abstract_file","fulltext_file"]

def read_manifest() -> List[Dict[str,str]]:
    if MANIFEST.exists():
        with MANIFEST.open("r", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    else:
        with MANIFEST.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLS); w.writeheader()
        return []

def write_manifest(rows: List[Dict[str,str]]):
    with MANIFEST.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS); w.writeheader()
        for r in rows:
            for c in COLS: r.setdefault(c, "")
            w.writerow({c:r[c] for c in COLS})

def count_fulltext(rows): return sum(1 for r in rows if r.get("fulltext_file","").endswith("fulltext.xml"))

# ---- Query config ----
TARGET_TOTAL = int(os.getenv("TARGET_TOTAL", "4000"))

# A conservative PMC query; adjust via PMC_TERM env if needed.
# Filters: 2010-2025, English or German, likely preclinical.
DEFAULT_TERM = '(2010:2025[pubdate]) AND (english[lang] OR german[lang]) AND (mouse OR mice OR rat OR rabbit OR "animal model" OR "preclinical")'
PMC_TERM = os.getenv("PMC_TERM", DEFAULT_TERM)

def pmc_esearch_ids(term: str, retmax=10000):
    """Yield PMCIDs (numeric) for the term."""
    retstart = 0
    while True:
        BUCKET.acquire()
        with Entrez.esearch(db="pmc", term=term, retmode="xml", retstart=retstart, retmax=retmax) as h:
            data = Entrez.read(h)
        ids = data.get("IdList", [])
        if not ids: break
        for _id in ids: yield _id  # numeric
        retstart += len(ids)
        if retstart >= int(data.get("Count", "0")): break

def pmc_fetch_batch(numerics: List[str]) -> List[bytes]:
    if not numerics: return []
    BUCKET.acquire()
    with Entrez.efetch(db="pmc", id=",".join(numerics), retmode="xml") as h:
        xml = h.read()
    parser = ET.XMLParser(recover=True, huge_tree=True)
    root = ET.fromstring(xml, parser=parser)
    return [ET.tostring(n, encoding="utf-8", xml_declaration=True)
            for n in root.xpath(".//*[local-name()='article']")]

def run():
    rows = read_manifest()
    have_pmcids = set(r["pmcid"] for r in rows if r.get("pmcid"))
    have_full   = count_fulltext(rows)
    print(f"[pmc_topup] starting with {len(rows)} rows; fulltext={have_full}; target={TARGET_TOTAL}", file=sys.stderr)

    if have_full >= TARGET_TOTAL:
        print("[pmc_topup] nothing to do (already at target).", file=sys.stderr); return

    # Index rows by PMID for quick attach
    by_pmid: Dict[str, Dict[str,str]] = {r["pmid"]: r for r in rows if r.get("pmid")}
    seen_new = 0

    batch, BATCH = [], 80
    for pmc_numeric in pmc_esearch_ids(PMC_TERM):
        pmcid = "PMC"+pmc_numeric
        if pmcid in have_pmcids:  # already in manifest
            continue
        batch.append(pmc_numeric)
        if len(batch) < BATCH:  # fill batch
            # early exit check
            if have_full + seen_new >= TARGET_TOTAL: break
            continue

        for art_bytes in pmc_fetch_batch(batch):
            # Extract PMCID / PMID from the article XML
            pmid, pmc = None, None
            if b"pub-id-type=\"pmcid\"" in art_bytes:
                i = art_bytes.find(b"pub-id-type=\"pmcid\"")
            # Quick regex-free extraction
            pmc_start = art_bytes.find(b">PMC")
            if pmc_start != -1:
                pmc_end = art_bytes.find(b"<", pmc_start)
                pmc = art_bytes[pmc_start+1:pmc_end].decode(errors="ignore") if pmc_end != -1 else None

            pmid_start = art_bytes.find(b"pub-id-type=\"pmid\"")
            if pmid_start != -1:
                v1 = art_bytes.find(b">", pmid_start)
                v2 = art_bytes.find(b"<", v1+1)
                if v1 != -1 and v2 != -1:
                    pmid = art_bytes[v1+1:v2].decode(errors="ignore")

            if not pmc:  # should not happen for PMC db
                continue

            if pmid and pmid in by_pmid:
                row = by_pmid[pmid]
            else:
                # Create a new manifest row if PMID unknown
                if not pmid:
                    # fallback unique key from PMCID
                    pmid = f"PMCID_{pmc}"
                row = {"pmid": pmid}
                rows.append(row); by_pmid[pmid] = row

            # Ensure folder
            d = PMID_ROOT / f"PMID_{pmid}"; d.mkdir(parents=True, exist_ok=True)
            # Save JATS
            (RAW_PMCXML / f"{pmc}.xml").write_bytes(art_bytes)
            (d / "fulltext.xml").write_bytes(art_bytes)

            # Update manifest row
            row["pmcid"] = pmc
            row["fulltext_file"] = str((d / "fulltext.xml").relative_to(DATA_ROOT))
            have_pmcids.add(pmc)
            seen_new += 1

            if have_full + seen_new >= TARGET_TOTAL:
                break

        batch = []
        if have_full + seen_new >= TARGET_TOTAL:
            break

    write_manifest(rows)
    print(f"[pmc_topup] added {seen_new} new fulltexts; total now ~{have_full + seen_new}", file=sys.stderr)
    print(f"[manifest] {MANIFEST}", file=sys.stderr)

if __name__ == "__main__":
    run()
