# pmc_chunker/src/download_xmls.py
from __future__ import annotations
import io, tarfile
from pathlib import Path
import pandas as pd
import requests
from lxml import etree
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

MANIFEST = Path("pmc_chunker/out/manifest_4000.csv")
XML_DIR  = Path("pmc_chunker/data/xml")
XML_DIR.mkdir(parents=True, exist_ok=True)

S = requests.Session()
S.headers.update({"User-Agent": "biomed-rag-agent/pmc-downloader"})

def _https(u: str) -> str:
    # NCBI serves the same path over HTTPS
    return u.replace("ftp://", "https://", 1) if u.startswith("ftp://") else u

@retry(stop=stop_after_attempt(5), wait=wait_exponential_jitter(1, 6))
def oa_record(pmcid: str) -> etree._Element:
    url = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
    r = S.get(url, params={"id": pmcid}, timeout=60)
    r.raise_for_status()
    return etree.fromstring(r.content)

def pick_xml_or_tgz(root: etree._Element):
    # prefer direct XML
    for link in root.xpath(".//link[@format='xml']/@href"):
        return ("xml", _https(link))
    for link in root.xpath(".//link[@format='tgz']/@href"):
        return ("tgz", _https(link))
    return (None, None)

@retry(stop=stop_after_attempt(5), wait=wait_exponential_jitter(1, 6))
def fetch_bytes(url: str) -> bytes:
    r = S.get(url, timeout=120)
    r.raise_for_status()
    return r.content

def save_from_tgz(pmcid: str, url: str) -> bool:
    data = fetch_bytes(url)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        members = [m for m in tf.getmembers() if m.name.lower().endswith((".nxml", ".xml"))]
        if not members:
            return False
        members.sort(key=lambda m: m.size, reverse=True)  # pick largest article file
        mem = members[0]
        with tf.extractfile(mem) as fh:
            (XML_DIR / f"{pmcid}.xml").write_bytes(fh.read())
    return True

def main():
    df = pd.read_csv(MANIFEST)
    for _, row in tqdm(df.iterrows(), total=len(df), desc="[download xml]"):
        pmcid = str(row["PMCID"])
        out = XML_DIR / f"{pmcid}.xml"
        if out.exists():
            continue

        root = oa_record(pmcid)
        kind, url = pick_xml_or_tgz(root)
        if not url:
            continue

        if kind == "xml":
            out.write_bytes(fetch_bytes(url))
        else:
            ok = save_from_tgz(pmcid, url)
            if not ok:
                print(f"no xml in tgz for {pmcid}")

    print("[done] XMLs in", XML_DIR)

if __name__ == "__main__":
    main()
