# download_pdfs.py
# pip install biopython requests bs4
import os, json, time, random, re, requests
from pathlib import Path
from typing import Optional
from bs4 import BeautifulSoup
from Bio import Entrez

# ---- paths ----
MAP = Path("pubmed_export/pmid_pmcid_map.json")
PDFDIR = Path("data/raw/pdfs"); PDFDIR.mkdir(parents=True, exist_ok=True)

# ---- NCBI/HTTP identity ----
Entrez.email = "bazefi01@thu.de"
Entrez.api_key = os.getenv("NCBI_API_KEY")  # setx NCBI_API_KEY "YOUR_KEY"
Entrez.tool  = "biomed-rag-ingester"
UA = {"User-Agent": "biomed-rag-ingester (bazefi01@thu.de)"}

# ---- helpers ----
def nap(): time.sleep(0.12 + random.uniform(0, 0.08))
def is_pdf(b: bytes) -> bool: return isinstance(b, (bytes, bytearray)) and b.startswith(b"%PDF")

def fetch_entrez_pdf(pmcid: str) -> Optional[bytes]:
    """Try official E-utilities PDF. Returns bytes or None."""
    try:
        pid = pmcid.replace("PMC","")
        with Entrez.efetch(db="pmc", id=pid, rettype="pdf", retmode="binary") as h:
            data = h.read()
        return data if is_pdf(data) else None
    except Exception:
        return None

def parse_pdf_url_from_page(html: str) -> Optional[str]:
    """Find a PDF link on the article HTML page."""
    soup = BeautifulSoup(html, "html.parser")

    m = soup.find("meta", attrs={"name": "citation_pdf_url"})
    if m and m.get("content"):
        href = m["content"].strip()
        return href if href.startswith("http") else "https://pmc.ncbi.nlm.nih.gov" + href

    for link in soup.find_all("link", attrs={"rel": "alternate", "type": "application/pdf"}):
        href = (link.get("href") or "").strip()
        if href:
            return href if href.startswith("http") else "https://pmc.ncbi.nlm.nih.gov" + href

    for a in soup.select('a[href]'):
        href = (a.get("href") or "").strip()
        if href.endswith(".pdf") or "/pdf/" in href or re.search(r"\.pdf(\?|$)", href, re.I):
            return href if href.startswith("http") else "https://pmc.ncbi.nlm.nih.gov" + href

    return None

def fetch_http_pdf(pmcid: str) -> Optional[bytes]:
    """Try common PMC URL patterns, then parse the page for a PDF link. Returns bytes or None."""
    base = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"
    candidates = [
        "?pdf=1",
        "pdf/",
        "pdf",                       # some servers accept without slash
        f"pdf/{pmcid}.pdf",          # occasional explicit filename
        f"{pmcid}.pdf",              # rare, but cheap to try
    ]
    with requests.Session() as s:
        s.headers.update(UA)

        # fast paths
        for path in candidates:
            url = base + path
            r = s.get(url, timeout=60, allow_redirects=True)
            if is_pdf(r.content):
                return r.content
            # save first non-PDF response for inspection
            dbg = PDFDIR / f"__debug_{pmcid}_{path.replace('/','_')}.html"
            try:
                dbg.write_bytes(r.content if isinstance(r.content, (bytes, bytearray)) else str(r.content).encode("utf-8", "ignore"))
            except Exception:
                pass

        # parse the article page
        page = s.get(base, timeout=60)
        if page.status_code != 200:
            return None
        pdf_url = parse_pdf_url_from_page(page.text)
        if not pdf_url:
            # save page for inspection (HTML-only articles will land here)
            (PDFDIR / f"__debug_{pmcid}_page.html").write_text(page.text, encoding="utf-8", errors="ignore")
            return None

        r = s.get(pdf_url, timeout=90, allow_redirects=True, stream=True)
        content = r.content
        if is_pdf(content):
            return content
        # log unexpected content
        (PDFDIR / f"__debug_{pmcid}_parsed_target.html").write_bytes(content if isinstance(content, (bytes, bytearray)) else str(content).encode("utf-8","ignore"))
        return None

def main():
    m = json.loads(MAP.read_text(encoding="utf-8"))
    ok = fail = 0
    failed = []

    for pmid, pmcid in m.items():
        out = PDFDIR / f"{pmcid}__PMID{pmid}.pdf"
        if out.exists():
            ok += 1
            continue

        data = fetch_entrez_pdf(pmcid)
        if data is None:
            data = fetch_http_pdf(pmcid)

        if data is not None:
            out.write_bytes(data)
            ok += 1
        else:
            fail += 1
            failed.append((pmid, pmcid))

        nap()

    print(f"PDFs: {ok} downloaded, {fail} failed")
    if failed:
        (PDFDIR / "_failed_pmcids.json").write_text(json.dumps(failed, indent=2))
        print("Failed examples:", failed[:5])

if __name__ == "__main__":
    main()
