# data_pipeline/build_open_access_corpus.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
from lxml import etree as ET
from Bio import Entrez
import csv, json, os, sys, time, datetime as dt
from multiprocessing import Pool, Lock, Manager

# ---------- NCBI identity ----------
Entrez.email   = os.getenv("NCBI_EMAIL",   "bazefi01@thu.de")
Entrez.tool    = os.getenv("NCBI_TOOL",    "biomed-rag-ingester")
Entrez.api_key = os.getenv("NCBI_API_KEY")

# ---------- Base Query ----------
QUERY_BASE = (
    '("Animal Experimentation"[MeSH] OR "Models, Animal"[MeSH] OR "Disease Models, Animal"[MeSH] '
    'OR "Preclinical Studies as Topic"[MeSH] OR "Drug Evaluation, Preclinical"[MeSH] '
    'OR "Toxicity Tests"[MeSH] OR "In Vivo Techniques"[MeSH]) '
    'AND (Animals[mh] NOT Humans[mh]) '
    'AND ("animal experiment*"[tiab] OR "animal model*"[tiab] OR "preclinical"[tiab] '
    'OR "in vivo"[tiab] OR mouse[tiab] OR mice[tiab] OR rat[tiab] OR rats[tiab] OR rabbit[tiab] '
    'OR rabbits[tiab] OR dog[tiab] OR dogs[tiab] OR canine[tiab] OR pig[tiab] OR pigs[tiab] '
    'OR swine[tiab] OR porcine[tiab] OR sheep[tiab] OR ovine[tiab] OR cattle[tiab] OR bovine[tiab] '
    'OR zebrafish[tiab] OR xenopus[tiab] OR "chick embryo"[tiab]) '
    'AND hasabstract[text] '
    'AND (english[lang] OR german[lang]) '
    'AND free full text[sb]'
)

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data_pipeline" / "data"
OUT_BASE  = DATA_ROOT / "pubmed_open_access"
MANIFESTS = DATA_ROOT / "manifests"
OUT_BASE.mkdir(parents=True, exist_ok=True)
MANIFESTS.mkdir(parents=True, exist_ok=True)
MANIFEST_CSV = MANIFESTS / "oa_manifest.csv"

def log(*a): print(*a, file=sys.stderr, flush=True)

# ---------- Utilities ----------
def esearch_ids(term: str, mindate: str, maxdate: str) -> List[str]:
    """Return all PMIDs for a given date slice."""
    ids, retstart = [], 0
    while True:
        try:
            with Entrez.esearch(
                    db="pubmed", term=term,
                    datetype="pdat", mindate=mindate, maxdate=maxdate,
                    retstart=retstart, retmax=10000, usehistory="n", retmode="xml"
            ) as h:
                data = Entrez.read(h)
            new = list(map(str, data.get("IdList", [])))
            if not new: break
            ids.extend(new)
            if len(new) < 10000: break
            retstart += 10000
            time.sleep(0.2)
        except Exception as e:
            log(f"[esearch error] {e}")
            time.sleep(2)
            continue
    return ids

def efetch_pubmed_batch(ids: List[str], retries: int = 3) -> bytes:
    """Fetch metadata+abstract XML for up to 10k PMIDs."""
    for _ in range(retries):
        try:
            with Entrez.efetch(db="pubmed", id=",".join(ids), retmode="xml") as h:
                return h.read()
        except Exception as e:
            log(f"[efetch error: {e}] retrying...")
            time.sleep(2)
    return b""

def extract_records(pubmed_xml: bytes) -> List[ET._Element]:
    if not pubmed_xml: return []
    root = ET.fromstring(pubmed_xml)
    return root.findall(".//PubmedArticle")

def text(node, xpath):
    n = node.find(xpath)
    return "" if n is None else "".join(n.itertext()).strip()

def list_text(nodes):
    return ["".join(n.itertext()).strip() for n in nodes]

# ---------- Parsing ----------
def parse_pubmed_article(art: ET._Element) -> Dict[str, Any]:
    pmid = text(art, ".//PMID")
    title = text(art, ".//ArticleTitle")
    journal = text(art, ".//Journal/Title")
    pub_year = text(art, ".//PubDate/Year") or text(art, ".//JournalIssue/PubDate/Year")
    lang = text(art, ".//Language") or "en"

    authors = []
    for a in art.findall(".//AuthorList/Author"):
        last, fore = text(a, "./LastName"), text(a, "./ForeName")
        full = " ".join([fore, last]).strip() or text(a, "./CollectiveName")
        if full: authors.append(full)

    doi, pmcid = "", ""
    for aid in art.findall(".//ArticleIdList/ArticleId"):
        t = (aid.get("IdType") or "").lower()
        if t == "doi": doi = aid.text or ""
        if t == "pmcid": pmcid = aid.text or ""

    mesh = [(mh.text or "").strip() for mh in art.findall(".//MeshHeading/DescriptorName") if mh.text]
    abs_paras = list_text(art.findall(".//Abstract/AbstractText"))
    abstract = "\n\n".join(abs_paras)
    publisher_url = f"https://doi.org/{doi}" if doi else ""

    return {
        "pmid": pmid,
        "pmcid": pmcid,
        "doi": doi,
        "title": title,
        "journal": journal,
        "publication_date": f"date_year_{pub_year}" if pub_year else "",
        "language": "language_german" if lang.lower().startswith("ger") else "language_english",
        "authors_full": authors,
        "mesh_terms": mesh,
        "url_source": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        "publisher_url": publisher_url,
        "kind": "doc_non-curated",
        "open_access": True
    }, abstract

# ---------- Saving ----------
def save_per_pmid(rec: Dict[str, Any], abstract: str) -> Dict[str, Any]:
    pmid = rec["pmid"]
    d = OUT_BASE / f"PMID_{pmid}"
    d.mkdir(parents=True, exist_ok=True)
    m = d / "metadata.json"
    a = d / "abstract.txt"
    m.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    a.write_text(abstract or "", encoding="utf-8")
    rec["metadata_file"] = str(m.relative_to(DATA_ROOT))
    rec["abstract_file"] = str(a.relative_to(DATA_ROOT))
    return rec

# ---------- Manifest ----------
def write_manifest(rows: List[Dict[str, Any]]):
    cols = [
        "pmid", "pmcid", "doi", "title", "journal", "publication_date", "language",
        "url_source", "publisher_url", "open_access",
        "metadata_file", "abstract_file", "fulltext_file"
    ]
    exists = MANIFEST_CSV.exists()
    with MANIFEST_CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if not exists:
            w.writeheader()
        for row in rows:
            for c in cols:
                row.setdefault(c, "")
            w.writerow({c: row[c] for c in cols})

# ---------- Worker ----------
def process_batch(batch):
    xml = efetch_pubmed_batch(batch)
    arts = extract_records(xml)
    out = []
    for art in arts:
        meta, abs_txt = parse_pubmed_article(art)
        meta = save_per_pmid(meta, abs_txt)
        out.append(meta)
    return out

# ---------- Main ----------
def run(target: int = 20000, start_date="2010/01/01", end_date="2025/12/31"):
    start = dt.datetime.strptime(start_date, "%Y/%m/%d").date()
    end = dt.datetime.strptime(end_date, "%Y/%m/%d").date()
    saved = 0

    for year in range(start.year, end.year + 1):
        mindate, maxdate = f"{year}/01/01", f"{year}/12/31"
        ids = esearch_ids(QUERY_BASE, mindate, maxdate)
        if not ids:
            continue
        log(f"[{year}] Found {len(ids)} PMIDs")

        chunks = [ids[i:i + 5000] for i in range(0, len(ids), 5000)]
        with Pool(processes=6) as pool:
            for result in pool.imap_unordered(process_batch, chunks):
                write_manifest(result)
                saved += len(result)
                if saved >= target:
                    break
        if saved >= target:
            break

    log(f"[done] Saved {saved} records → {OUT_BASE}")
    log(f"[manifest] {MANIFEST_CSV}")

# ---------- CLI ----------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=20000)
    ap.add_argument("--start_date", type=str, default="2010/01/01")
    ap.add_argument("--end_date", type=str, default="2025/12/31")
    args = ap.parse_args()
    run(target=args.target, start_date=args.start_date, end_date=args.end_date)
