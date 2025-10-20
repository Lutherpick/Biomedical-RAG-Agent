# data_pipeline/step_fulltext_from_manifest.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
from lxml import etree as ET
from Bio import Entrez
import os, sys, csv, re, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import certifi

# ───────── Paths ─────────
ROOT       = Path(__file__).resolve().parents[1]
DATA_ROOT  = ROOT / "data_pipeline" / "data"
PMID_ROOT  = DATA_ROOT / "pubmed_open_access"
RAW_PMCXML = DATA_ROOT / "raw" / "pmc_xml"
RAW_PUBXML = DATA_ROOT / "raw" / "pubmed_xml"
MAN_DIR    = DATA_ROOT / "manifests"
MANIFEST   = MAN_DIR / "oa_manifest.csv"
FAILED_URLS = MAN_DIR / "failed_urls.csv"

RAW_PMCXML.mkdir(parents=True, exist_ok=True)
RAW_PUBXML.mkdir(parents=True, exist_ok=True)
MAN_DIR.mkdir(parents=True, exist_ok=True)

def log(*a): print(*a, file=sys.stderr, flush=True)

# ───────── NCBI identity (per document) ─────────
# (Uses your environment if set; otherwise falls back to a stable default).
Entrez.email   = os.getenv("NCBI_EMAIL", "bazefi01@thu.de")
Entrez.tool    = os.getenv("NCBI_TOOL",  "biomed-rag-ingester")
Entrez.api_key = os.getenv("NCBI_API_KEY")

# ───────── Modes / limits (env switches) ─────────
# - MODE="full": full corpus pass (ELink + PMC + optional publisher)
# - MODE="pmc_working_set": 4k PMC-only pass (no ELink, no publisher)
MODE = os.getenv("MODE", "full")
ELINK_LIMIT = int(os.getenv("ELINK_LIMIT", "0"))  # 0 = no cap
MANIFEST_PATH = os.getenv("MANIFEST_PATH", str(MANIFEST))  # optional alt manifest

# ───────── Auto-tuning & limits ─────────
CPU = os.cpu_count() or 8
THREADS_ELINK      = int(os.getenv("THREADS_ELINK", "4"))
THREADS_EFETCH     = int(os.getenv("THREADS_EFETCH", "4"))
THREADS_PUBLISHER  = int(os.getenv("THREADS_PUBLISHER", "12"))

BATCH_ELINK, BATCH_PUBMED, BATCH_PMC = 200, 200, 50
DEFAULT_RPS_WITH_KEY = 3
DEFAULT_RPS_NO_KEY   = 1
MAX_RPS = float(os.getenv(
    "MAX_RPS",
    str(DEFAULT_RPS_WITH_KEY if Entrez.api_key else DEFAULT_RPS_NO_KEY)
))

# ───────── Token Bucket for fair-use global rate ─────────
class TokenBucket:
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

BUCKET = TokenBucket(MAX_RPS)

# ───────── Manifest helpers ─────────
COLS = [
    "pmid","pmcid","doi","title","journal","publication_date","language",
    "url_source","publisher_url","open_access",
    "metadata_file","abstract_file","fulltext_file"
]

def _strip_bom_quotes(s: str) -> str:
    if not isinstance(s, str):
        return s
    # Remove UTF-8 BOM and any surrounding quotes/whitespace
    return s.lstrip("\ufeff").strip().strip('"').strip("'")

def _normalize_row_keys(row: Dict[str, str]) -> Dict[str, str]:
    return {_strip_bom_quotes(k): v for k, v in row.items()}

def _normalize_values(row: Dict[str, str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in row.items():
        if isinstance(v, str):
            vv = v.strip().strip('"').strip("'")
            # Normalize PMCIDs to start with uppercase "PMC"
            if k == "pmcid" and vv and not vv.upper().startswith("PMC"):
                vv = "PMC" + vv
            out[k] = vv
        else:
            out[k] = v
    return out

def read_manifest() -> List[Dict[str,str]]:
    p = Path(MANIFEST_PATH)
    if not p.exists():
        raise FileNotFoundError(f"Manifest not found: {p}")
    # Use utf-8-sig to automatically drop BOM if present
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        rdr = csv.DictReader(f)
        rows = [_normalize_values(_normalize_row_keys(r)) for r in rdr]
    # Basic schema validation
    missing_cols = [c for c in ("pmid","pmcid") if c not in (rows[0].keys() if rows else [])]
    if missing_cols:
        raise KeyError(
            f"Manifest missing required columns: {missing_cols}. "
            f"Found columns: {list(rows[0].keys()) if rows else 'NONE'} in {p}"
        )
    return rows

def write_manifest(rows: List[Dict[str,str]]):
    p = Path(MANIFEST_PATH)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            for c in COLS:
                r.setdefault(c, "")
            w.writerow({c: r[c] for c in COLS})

def _summary(rows: List[Dict[str,str]]) -> str:
    n = len(rows)
    with_pmcid = sum(1 for r in rows if r.get("pmcid"))
    with_xml   = sum(1 for r in rows if str(r.get("fulltext_file","")).endswith("fulltext.xml"))
    with_pdf   = sum(1 for r in rows if str(r.get("fulltext_file","")).endswith("fulltext.pdf"))
    with_html  = sum(1 for r in rows if str(r.get("fulltext_file","")).endswith("fulltext.html"))
    return f"rows={n} pmcid={with_pmcid} xml={with_xml} pdf={with_pdf} html={with_html}"

# ───────── Step 1 — PMID → PMCID via ELink ─────────
def elink_chunk(pmids: List[str]) -> Dict[str,str]:
    BUCKET.acquire()
    try:
        with Entrez.elink(dbfrom="pubmed", db="pmc", id=",".join(pmids), retmode="xml") as h:
            data = Entrez.read(h)
    except Exception as e:
        log("[elink error]", e)
        return {}
    out: Dict[str,str] = {}
    for ls in data:
        pid = str(ls.get("IdList",[None])[0]) if ls.get("IdList") else None
        if not pid: continue
        for db in ls.get("LinkSetDb", []) or []:
            if db.get("DbTo") == "pmc":
                for link in db.get("Link", []) or []:
                    out[pid] = "PMC" + str(link.get("Id"))
                    break
    return out

def map_pmids_to_pmcids(pmids: List[str]) -> Dict[str,str]:
    pmids = [p for p in pmids if p]
    if not pmids: return {}
    chunks = [pmids[i:i+BATCH_ELINK] for i in range(0, len(pmids), BATCH_ELINK)]
    results: Dict[str,str] = {}
    with ThreadPoolExecutor(max_workers=THREADS_ELINK) as ex:
        for fut in as_completed([ex.submit(elink_chunk, ch) for ch in chunks]):
            results.update(fut.result())
    return results

# ───────── Step 2 — ensure metadata.xml ─────────
def pubmed_fetch_chunk(pmids: List[str]) -> Dict[str, bytes]:
    BUCKET.acquire()
    try:
        with Entrez.efetch(db="pubmed", id=",".join(pmids), retmode="xml") as h:
            data = h.read()
    except Exception as e:
        log("[efetch pubmed error]", e)
        return {}
    out: Dict[str, bytes] = {}
    root = ET.fromstring(data)
    for art in root.findall(".//PubmedArticle"):
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
                d = PMID_ROOT / f"PMID_{pmid}"
                d.mkdir(parents=True, exist_ok=True)
                (RAW_PUBXML / f"PMID_{pmid}.xml").write_bytes(xml_b)
                (d / "metadata.xml").write_bytes(xml_b)

# ───────── Step 3 — PMC JATS (preferred full text) ─────────
PMCID_RE = re.compile(br"<article-id[^>]*pub-id-type=['\"]pmcid['\"][^>]*>PMC(\d+)</article-id>")

def pmc_fetch_chunk(numeric_pmcids: List[str]) -> List[Tuple[str, bytes]]:
    BUCKET.acquire()
    try:
        with Entrez.efetch(db="pmc", id=",".join(numeric_pmcids), retmode="xml") as h:
            xml = h.read()
    except Exception as e:
        log("[efetch pmc error]", e)
        return []
    parser = ET.XMLParser(recover=True, huge_tree=True)
    root = ET.fromstring(xml, parser=parser)
    out: List[Tuple[str, bytes]] = []
    for art in root.xpath(".//*[local-name()='article']"):
        b = ET.tostring(art, encoding="utf-8", xml_declaration=True)
        m = PMCID_RE.search(b)
        if m:
            out.append(("PMC"+m.group(1).decode(), b))
    return out

def ensure_fulltext_via_pmc(rows: List[Dict[str,str]]):
    needed: Dict[str,str] = {
        r["pmcid"]: r["pmid"]
        for r in rows
        if r.get("pmcid") and not r.get("fulltext_file")
    }
    if not needed: return
    pmc_numeric = [re.sub(r"^PMC","", x.upper()) for x in needed.keys()]
    chunks = [pmc_numeric[i:i+BATCH_PMC] for i in range(0, len(pmc_numeric), BATCH_PMC)]
    with ThreadPoolExecutor(max_workers=THREADS_EFETCH) as ex:
        for fut in as_completed([ex.submit(pmc_fetch_chunk, ch) for ch in chunks]):
            for pmcid, xml_b in fut.result():
                pmid = needed.get(pmcid)
                if not pmid: continue
                d = PMID_ROOT / f"PMID_{pmid}"
                d.mkdir(parents=True, exist_ok=True)
                (RAW_PMCXML / f"{pmcid}.xml").write_bytes(xml_b)
                (d / "fulltext.xml").write_bytes(xml_b)

# ───────── Step 4 — Publisher PDF/HTML fallback (disabled in pmc_working_set) ─────────
UA = os.getenv("HTTP_UA", "Mozilla/5.0 (compatible; biomed-rag-ingester; +https://pubmed.ncbi.nlm.nih.gov)")
SESSION = requests.Session()
SESSION.verify = certifi.where()
retries = Retry(
    total=3,
    backoff_factor=0.5,
    status_forcelist=[429,500,502,503,504],
    allowed_methods=["GET","HEAD"],
    raise_on_status=False,
)
SESSION.mount("https://", HTTPAdapter(max_retries=retries))
SESSION.mount("http://",  HTTPAdapter(max_retries=retries))
SESSION.headers.update({"User-Agent": UA, "Accept": "*/*"})

BLOCKED_HOSTS = {"www.pagepressjournals.org", "www.ectrx.org"}

def try_publisher_download(doi_url: str, out_dir: Path) -> str:
    """Return relative path to saved fulltext.{pdf|html} or '' if not saved."""
    if not doi_url: return ""
    try:
        # Prefer direct PDF via content negotiation
        r = SESSION.get(
            doi_url, headers={"Accept": "application/pdf"},
            timeout=8, allow_redirects=True
        )
        ctype = (r.headers.get("Content-Type") or "").lower()
        if r.status_code == 200 and ("application/pdf" in ctype or r.url.lower().endswith(".pdf")) and r.content:
            p = out_dir / "fulltext.pdf"
            p.write_bytes(r.content)
            return str(p.relative_to(DATA_ROOT))
        # Fallback to HTML
        r_html = SESSION.get(
            doi_url,
            headers={"Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8"},
            timeout=25, allow_redirects=True
        )
        if r_html.text.strip():
            p = out_dir / "fulltext.html"
            p.write_text(r_html.text, encoding="utf-8", errors="ignore")
            return str(p.relative_to(DATA_ROOT))
    except requests.exceptions.SSLError:
        return ""
    except Exception as e:
        log("[publisher fallback]", e)
    return ""

def ensure_fulltext_via_publisher(rows: List[Dict[str,str]]):
    # Hard-disabled in pmc_working_set (PMC JATS only, per document)
    if MODE == "pmc_working_set":
        return
    todo = [r for r in rows if not r.get("pmcid") and not r.get("fulltext_file") and r.get("publisher_url")]
    if not todo: return

    failed = []
    total = len(todo)
    done = 0

    def worker(r: Dict[str,str]):
        nonlocal done
        pmid = r["pmid"]
        host = urlparse(r["publisher_url"]).netloc.lower()
        if host in BLOCKED_HOSTS:
            failed.append({"pmid": pmid, "url": r["publisher_url"], "reason": "blocked_host"})
            done += 1; return
        d = PMID_ROOT / f"PMID_{pmid}"
        d.mkdir(parents=True, exist_ok=True)
        saved = try_publisher_download(r["publisher_url"], d)
        if saved:
            r["fulltext_file"] = saved
        else:
            failed.append({"pmid": pmid, "url": r["publisher_url"], "reason": "download_failed"})
        done += 1
        if done % 500 == 0:
            log(f"[Publisher fallback] Progress: {done}/{total}")

    log(f"[Publisher fallback] {total} records without PMCID → PDF/HTML | threads={THREADS_PUBLISHER}")
    with ThreadPoolExecutor(max_workers=THREADS_PUBLISHER) as ex:
        list(ex.map(worker, todo))

    if failed:
        with FAILED_URLS.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["pmid","url","reason"])
            w.writeheader()
            for r in failed: w.writerow(r)
        log(f"[Publisher fallback] {len(failed)} failed URLs logged → {FAILED_URLS}")

# ───────── Orchestration ─────────
def main():
    rows = read_manifest()
    if not rows:
        log(f"[warn] Manifest has header but no data rows: {MANIFEST_PATH}")
        write_manifest(rows)  # keep structure but won't crash later
        log(f"[done] Manifest normalized with metadata.xml / abstract.txt / fulltext.(xml|pdf|html)")
        log(f"[manifest] {MANIFEST_PATH}")
        return

    # Summary (pre)
    log(f"[summary] pre: {_summary(rows)} | mode={MODE} | rps={MAX_RPS}")

    # 1) Map missing PMCIDs (skip entirely in pmc_working_set)
    missing = [r["pmid"] for r in rows if not r.get("pmcid")] if MODE != "pmc_working_set" else []
    if ELINK_LIMIT > 0 and missing:
        missing = missing[:ELINK_LIMIT]
    if missing:
        log(f"[ELink] {len(missing)} PMIDs → PMCIDs | threads={THREADS_ELINK} | rps={MAX_RPS}")
        mapping = map_pmids_to_pmcids(missing)
        for r in rows:
            if not r.get("pmcid"):
                pmcid = mapping.get(r["pmid"])
                if pmcid:
                    r["pmcid"] = pmcid

    # 2) Ensure metadata.xml
    if any(not r.get("metadata_file") for r in rows):
        log("[PubMed XML] backfilling metadata.xml where missing")
        ensure_metadata_xml(rows)

    # 3) PMC JATS full text (preferred, aligns with document)
    need_pmc = [r for r in rows if r.get("pmcid") and not r.get("fulltext_file")]
    if need_pmc:
        log(f"[PMC JATS] {len(need_pmc)} records → fulltext.xml | threads={THREADS_EFETCH} | rps={MAX_RPS}")
        ensure_fulltext_via_pmc(rows)

    # 4) Publisher fallback (disabled in pmc_working_set)
    ensure_fulltext_via_publisher(rows)

    # 5) Normalize manifest to actual files present on disk
    for r in rows:
        pmid = r["pmid"]
        d = PMID_ROOT / f"PMID_{pmid}"
        meta, abst = d/"metadata.xml", d/"abstract.txt"
        full_xml, full_pdf, full_html = d/"fulltext.xml", d/"fulltext.pdf", d/"fulltext.html"
        if meta.exists(): r["metadata_file"] = str(meta.relative_to(DATA_ROOT))
        if abst.exists(): r["abstract_file"] = str(abst.relative_to(DATA_ROOT))
        if full_xml.exists(): r["fulltext_file"] = str(full_xml.relative_to(DATA_ROOT))
        elif full_pdf.exists(): r["fulltext_file"] = str(full_pdf.relative_to(DATA_ROOT))
        elif full_html.exists(): r["fulltext_file"] = str(full_html.relative_to(DATA_ROOT))

    # Summary (post)
    log(f"[summary] post: {_summary(rows)}")

    write_manifest(rows)
    log("[done] Manifest normalized with metadata.xml / abstract.txt / fulltext.(xml|pdf|html)")
    log(f"[manifest] {MANIFEST_PATH}")

if __name__ == "__main__":
    main()
