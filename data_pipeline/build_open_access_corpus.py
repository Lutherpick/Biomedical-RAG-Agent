# data_pipeline/build_open_access_corpus.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
from lxml import etree as ET
from Bio import Entrez
import csv, os, sys, time, datetime as dt

# ---------- NCBI identity ----------
Entrez.email   = os.getenv("NCBI_EMAIL",   "bazefi01@thu.de")
Entrez.tool    = os.getenv("NCBI_TOOL",    "biomed-rag-ingester")
Entrez.api_key = os.getenv("NCBI_API_KEY")

# ---------- Base Query (as documented: OA, animals/preclinical) ----------
# build_open_access_corpus.py

QUERY_BASE = (
    '("Animal Experimentation"[MeSH] OR "Models, Animal"[MeSH] OR "Disease Models, Animal"[MeSH] '
    'OR "Preclinical Studies as Topic"[MeSH] OR "Drug Evaluation, Preclinical"[MeSH] '
    'OR "Toxicity Tests"[MeSH] OR "In Vivo Techniques"[MeSH]) '
    'AND (Animals[mh] NOT Humans[mh]) '
    # ▼ add the tiab/species block you pasted ▼
    'AND ('
    '"animal experiment*"[tiab] OR '
    '"animal model*"[tiab] OR '
    '"preclinical"[tiab] OR '
    '"in vivo"[tiab] OR '
    'mouse[tiab] OR mice[tiab] OR '
    'rat[tiab] OR rats[tiab] OR '
    'rabbit[tiab] OR rabbits[tiab] OR '
    'dog[tiab] OR dogs[tiab] OR canine[tiab] OR '
    'pig[tiab] OR pigs[tiab] OR swine[tiab] OR porcine[tiab] OR '
    'sheep[tiab] OR ovine[tiab] OR '
    'cattle[tiab] OR bovine[tiab] OR '
    'zebrafish[tiab] OR '
    'xenopus[tiab] OR '
    '"chick embryo"[tiab]'
    ') '
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
RUN_INFO_TXT = MANIFESTS / "oa_manifest_runinfo.txt"

def log(*a): print(*a, file=sys.stderr, flush=True)

COLS = [
    "pmid","pmcid","doi","title","journal","publication_date","language",
    "url_source","publisher_url","open_access",
    "metadata_file","abstract_file","fulltext_file"
]

def ensure_manifest():
    if not MANIFEST_CSV.exists():
        with MANIFEST_CSV.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=COLS).writeheader()

def append_manifest(rows: List[Dict[str, str]]):
    ensure_manifest()
    with MANIFEST_CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        for r in rows:
            for c in COLS: r.setdefault(c, "")
            w.writerow({c: r[c] for c in COLS})

def text(node: ET._Element, xpath: str) -> str:
    n = node.find(xpath)
    return "" if n is None else "".join(n.itertext()).strip()

def parse_one_pubmed_article(art: ET._Element) -> Dict[str, str]:
    pmid = text(art, ".//PMID")
    title = text(art, ".//ArticleTitle")
    journal = text(art, ".//Journal/Title")
    pub_year = text(art, ".//PubDate/Year") or text(art, ".//JournalIssue/PubDate/Year")
    lang = text(art, ".//Language") or "en"

    doi, pmcid = "", ""
    for aid in art.findall(".//ArticleIdList/ArticleId"):
        t = (aid.get("IdType") or "").lower()
        if t == "doi": doi = (aid.text or "").strip()
        if t == "pmcid": pmcid = (aid.text or "").strip()

    abstract_paras = ["".join(p.itertext()).strip() for p in art.findall(".//Abstract/AbstractText")]
    abstract = "\n\n".join([p for p in abstract_paras if p])

    return {
        "pmid": pmid,
        "pmcid": pmcid,
        "doi": doi,
        "title": title,
        "journal": journal,
        "publication_date": f"date_year_{pub_year}" if pub_year else "",
        "language": "language_german" if lang.lower().startswith("ger") else "language_english",
        "abstract": abstract,
        "publisher_url": f"https://doi.org/{doi}" if doi else "",
    }

def run(start_date="2010/01/01", end_date="2025/12/31"):
    # split by publication year to stay under API limits
    start_y = int(start_date.split("/")[0])
    end_y   = int(end_date.split("/")[0])
    BATCH   = 10000
    append_rows: List[Dict[str,str]] = []

    for year in range(start_y, end_y + 1):
        log(f"[esearch] {year} (usehistory=Y)")
        with Entrez.esearch(
                db="pubmed",
                term=QUERY_BASE,
                datetype="pdat",
                mindate=f"{year}/01/01",
                maxdate=f"{year}/12/31",
                usehistory="y",
                retmode="xml",
                retmax=0
        ) as h:
            data = Entrez.read(h)

        count  = int(data.get("Count", 0))
        webenv = data.get("WebEnv")
        qkey   = data.get("QueryKey")
        if count == 0 or not webenv or not qkey:
            log(f"[{year}] nothing found"); continue

        log(f"[{year}] Count={count} WebEnv={webenv} QueryKey={qkey}")

        fetched = 0
        while fetched < count:
            retmax = min(BATCH, count - fetched)
            log(f"[efetch] {year} {fetched}..{fetched+retmax-1} / {count}")
            tries = 0
            while True:
                try:
                    with Entrez.efetch(
                            db="pubmed",
                            query_key=qkey,
                            WebEnv=webenv,
                            rettype="xml",
                            retmode="xml",
                            retstart=fetched,
                            retmax=retmax,
                            api_key=Entrez.api_key,
                            email=Entrez.email
                    ) as h:
                        xml_bytes = h.read()
                    break
                except Exception as e:
                    tries += 1
                    log(f"[efetch error] {e} (retry {tries}/3)")
                    time.sleep(2)
                    if tries >= 3:
                        log(f"[efetch skipped range {fetched}..{fetched+retmax-1}]")
                        xml_bytes = b""
                        break

            if not xml_bytes:
                fetched += retmax
                continue

            root = ET.fromstring(xml_bytes)
            for art in root.findall(".//PubmedArticle"):
                pmid_el = art.find(".//PMID")
                pmid = pmid_el.text.strip() if pmid_el is not None else None
                if not pmid: continue
                d = OUT_BASE / f"PMID_{pmid}"
                d.mkdir(parents=True, exist_ok=True)
                (d / "metadata.xml").write_bytes(
                    ET.tostring(art, encoding="utf-8", xml_declaration=True)
                )
                rec = parse_one_pubmed_article(art)
                (d / "abstract.txt").write_text(rec["abstract"], encoding="utf-8")
                row = {
                    "pmid": rec["pmid"],
                    "pmcid": rec["pmcid"],
                    "doi": rec["doi"],
                    "title": rec["title"],
                    "journal": rec["journal"],
                    "publication_date": rec["publication_date"],
                    "language": rec["language"],
                    "url_source": f"https://pubmed.ncbi.nlm.nih.gov/{rec['pmid']}/",
                    "publisher_url": rec["publisher_url"],
                    "open_access": "True",
                    "metadata_file": str((d / "metadata.xml").relative_to(DATA_ROOT)),
                    "abstract_file": str((d / "abstract.txt").relative_to(DATA_ROOT)),
                    "fulltext_file": ""
                }
                append_rows.append(row)

            fetched += retmax
            time.sleep(0.3)

            if len(append_rows) >= 2000:
                append_manifest(append_rows)
                append_rows.clear()

        # flush yearly
        if append_rows:
            append_manifest(append_rows)
            append_rows.clear()
        log(f"[{year}] finished -> manifest updated")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--start_date", type=str, default="2010/01/01")
    ap.add_argument("--end_date", type=str, default="2025/12/31")
    args = ap.parse_args()
    run(start_date=args.start_date, end_date=args.end_date)
