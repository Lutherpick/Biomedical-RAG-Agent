# data_pipeline/step_fulltext_from_manifest.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
from lxml import etree as ET
from Bio import Entrez
import os, sys, csv, re, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# ───────── NCBI identity (auto) ─────────
Entrez.email   = os.getenv("NCBI_EMAIL", "bazefi01@thu.de")
Entrez.tool    = os.getenv("NCBI_TOOL",  "biomed-rag-ingester")
Entrez.api_key = os.getenv("NCBI_API_KEY")  # enables higher rate

# ───────── Paths ─────────
ROOT       = Path(__file__).resolve().parents[1]
DATA_ROOT  = ROOT / "data_pipeline" / "data"
PMID_ROOT  = DATA_ROOT / "pubmed_open_access"
RAW_PMCXML = DATA_ROOT / "raw" / "pmc_xml"
RAW_PUBXML = DATA_ROOT / "raw" / "pubmed_xml"
MAN_DIR    = DATA_ROOT / "manifests"
MANIFEST   = MAN_DIR / "oa_manifest.csv"
RAW_PMCXML.mkdir(parents=True, exist_ok=True)
RAW_PUBXML.mkdir(parents=True, exist_ok=True)

def log(*a): print(*a, file=sys.stderr, flush=True)

# ───────── Auto-tuning per machine ─────────
CPU = os.cpu_count() or 8
# High but safe thread counts; override with env if needed
THREADS_ELINK  = int(os.getenv("THREADS_ELINK",  min(64, max(8, 2*CPU))))
THREADS_EFETCH = int(os.getenv("THREADS_EFETCH", min(64, max(8, 2*CPU))))
# Practical batch sizes (one HTTP call per batch)
BATCH_ELINK   = 250
BATCH_PUBMED  = 250        # PubMed efetch
BATCH_PMC     = 80         # PMC JATS are large; keep ≤100

# NCBI polite limits (token bucket):
DEFAULT_RPS_WITH_KEY = 10
DEFAULT_RPS_NO_KEY   = 3
MAX_RPS = float(os.getenv(
    "MAX_RPS",
    str(DEFAULT_RPS_WITH_KEY if Entrez.api_key else DEFAULT_RPS_NO_KEY)
))

class TokenBucket:
    """Simple thread-safe token bucket limiter."""
    def __init__(self, rate_per_sec: float, capacity: float | None = None):
        self.rate = max(0.5, rate_per_sec)
        self.capacity = capacity or self.rate
        self.tokens = self.capacity
        self.lock = threading.Lock()
        self.last = time.perf_counter()
    def acquire(self):
        while True:
            with self.lock:
                now = time.perf_counter()
                self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.rate)
                self.last = now
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
            time.sleep(0.002)

# One shared bucket for all HTTP calls:
BUCKET = TokenBucket(MAX_RPS)

# ───────── Manifest helpers ─────────
def read_manifest() -> List[Dict[str,str]]:
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST}")
    with MANIFEST.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def write_manifest(rows: List[Dict[str,str]]):
    cols = ["pmid","pmcid","doi","title","journal","publication_date","language",
            "url_source","publisher_url","open_access",
            "metadata_file","abstract_file","fulltext_file"]
    with MANIFEST.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            for c in cols: r.setdefault(c,"")
            w.writerow({c:r[c] for c in cols})

# ───────── Step 1 — PMID → PMCID via ELink ─────────
def elink_chunk(pmids: List[str]) -> Dict[str,str]:
    BUCKET.acquire()
    try:
        with Entrez.elink(dbfrom="pubmed", db="pmc", id=",".join(pmids), retmode="xml") as h:
            data = Entrez.read(h)
    except Exception as e:
        log("[elink error]", e); return {}
    out: Dict[str,str] = {}
    for ls in data:
        pid = str(ls.get("IdList",[None])[0]) if ls.get("IdList") else None
        if not pid: continue
        for db in ls.get("LinkSetDb", []) or []:
            if db.get("DbTo") == "pmc":
                for link in db.get("Link", []) or []:
                    out[pid] = "PMC" + str(link.get("Id")); break
    return out

def map_pmids_to_pmcids(pmids: List[str]) -> Dict[str,str]:
    pmids = [p for p in pmids if p]
    chunks = [pmids[i:i+BATCH_ELINK] for i in range(0, len(pmids), BATCH_ELINK)]
    results: Dict[str,str] = {}
    with ThreadPoolExecutor(max_workers=THREADS_ELINK) as ex:
        for fut in as_completed([ex.submit(elink_chunk, ch) for ch in chunks]):
            results.update(fut.result())
    return results

# ───────── Step 2 — metadata.xml (PubMed EFetch) ─────────
def parse_pubmed_articles(xml_bytes: bytes) -> List[ET._Element]:
    root = ET.fromstring(xml_bytes)
    return root.findall(".//PubmedArticle")

def pubmed_fetch_chunk(pmids: List[str]) -> Dict[str, bytes]:
    BUCKET.acquire()
    try:
        with Entrez.efetch(db="pubmed", id=",".join(pmids), retmode="xml") as h:
            data = h.read()
    except Exception as e:
        log("[efetch pubmed error]", e); return {}
    out: Dict[str, bytes] = {}
    for art in parse_pubmed_articles(data):
        pmid_el = art.find(".//PMID")
        pmid = pmid_el.text.strip() if pmid_el is not None else None
        if pmid:
            out[pmid] = ET.tostring(art, encoding="utf-8", xml_declaration=True)
    return out

def ensure_metadata_xml(rows: List[Dict[str,str]]):
    need = [r["pmid"] for r in rows if not r.get("metadata_file")]
    if not need: return
    chunks = [need[i:i+BATCH_PUBMED] for i in range(0, len(need), BATCH_PUBMED)]
    with ThreadPoolExecutor(max_workers=THREADS_EFETCH) as ex:
        for fut in as_completed([ex.submit(pubmed_fetch_chunk, ch) for ch in chunks]):
            for pmid, xml_b in fut.result().items():
                d = PMID_ROOT / f"PMID_{pmid}"; d.mkdir(parents=True, exist_ok=True)
                (RAW_PUBXML / f"PMID_{pmid}.xml").write_bytes(xml_b)  # cache
                (d / "metadata.xml").write_bytes(xml_b)

# ───────── Step 3 — fulltext.xml (PMC JATS) ─────────
PMCID_RE = re.compile(br"<article-id[^>]*pub-id-type=['\"]pmcid['\"][^>]*>PMC(\d+)</article-id>")
PMID_RE  = re.compile(br"<article-id[^>]*pub-id-type=['\"]pmid['\"][^>]*>(\d+)</article-id>")

def pmc_fetch_chunk(numeric_pmcids: List[str]) -> List[Tuple[str, bytes]]:
    BUCKET.acquire()
    try:
        with Entrez.efetch(db="pmc", id=",".join(numeric_pmcids), retmode="xml") as h:
            xml = h.read()
    except Exception as e:
        log("[efetch pmc error]", e); return []
    parser = ET.XMLParser(recover=True, huge_tree=True)
    root = ET.fromstring(xml, parser=parser)
    out: List[Tuple[str, bytes]] = []
    for art in root.xpath(".//*[local-name()='article']"):
        b = ET.tostring(art, encoding="utf-8", xml_declaration=True)
        m = PMCID_RE.search(b)
        if m:
            out.append(("PMC"+m.group(1).decode(), b))
    return out

def ensure_fulltext_xml(rows: List[Dict[str,str]]):
    needed: Dict[str,str] = {r["pmcid"]: r["pmid"]
                             for r in rows
                             if r.get("pmcid") and not r.get("fulltext_file","").endswith("fulltext.xml")}
    if not needed: return
    pmc_numeric = [re.sub(r"^PMC","", x) for x in needed.keys()]
    chunks = [pmc_numeric[i:i+BATCH_PMC] for i in range(0, len(pmc_numeric), BATCH_PMC)]
    with ThreadPoolExecutor(max_workers=THREADS_EFETCH) as ex:
        for fut in as_completed([ex.submit(pmc_fetch_chunk, ch) for ch in chunks]):
            for pmcid, xml_b in fut.result():
                pmid = needed.get(pmcid)
                if not pmid:
                    m2 = PMID_RE.search(xml_b)
                    if m2: pmid = m2.group(1).decode()
                if not pmid: continue
                d = PMID_ROOT / f"PMID_{pmid}"; d.mkdir(parents=True, exist_ok=True)
                (RAW_PMCXML / f"{pmcid}.xml").write_bytes(xml_b)  # cache
                (d / "fulltext.xml").write_bytes(xml_b)

# ───────── Orchestration ─────────
def main():
    rows = read_manifest()

    # 1) Map missing PMCIDs
    missing = [r["pmid"] for r in rows if not r.get("pmcid")]
    if missing:
        log(f"[ELink] {len(missing)} PMIDs → PMCIDs | threads={THREADS_ELINK} | rps={MAX_RPS}")
        mapping = map_pmids_to_pmcids(missing)
        for r in rows:
            if not r.get("pmcid") and r["pmid"] in mapping:
                r["pmcid"] = mapping[r["pmid"]]

    # 2) Ensure metadata.xml
    need_meta = [r for r in rows if not r.get("metadata_file") or not r["metadata_file"].endswith("metadata.xml")]
    if need_meta:
        log(f"[PubMed XML] {len(need_meta)} records → metadata.xml | threads={THREADS_EFETCH} | rps={MAX_RPS}")
        ensure_metadata_xml(rows)

    # 3) Ensure fulltext.xml
    need_full = [r for r in rows if r.get("pmcid") and not r.get("fulltext_file","").endswith("fulltext.xml")]
    if need_full:
        log(f"[PMC JATS] {len(need_full)} records → fulltext.xml | threads={THREADS_EFETCH} | rps={MAX_RPS}")
        ensure_fulltext_xml(rows)

    # 4) Normalize manifest paths
    for r in rows:
        pmid = r["pmid"]; d = PMID_ROOT / f"PMID_{pmid}"
        meta, abst, full = d/"metadata.xml", d/"abstract.txt", d/"fulltext.xml"
        if meta.exists(): r["metadata_file"] = str(meta.relative_to(DATA_ROOT))
        if abst.exists(): r["abstract_file"] = str(abst.relative_to(DATA_ROOT))
        if full.exists(): r["fulltext_file"]  = str(full.relative_to(DATA_ROOT))

    write_manifest(rows)
    log("[done] Manifest updated to metadata.xml / abstract.txt / fulltext.xml")
    log(f"[manifest] {MANIFEST}")

if __name__ == "__main__":
    main()
