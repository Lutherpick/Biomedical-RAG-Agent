# data_pipeline/download_pmc_xml.py
from __future__ import annotations
from pathlib import Path
from typing import Optional
from lxml import etree as ET
from Bio import Entrez
import json, time, re, sys, os

# ---- NCBI identity ----
Entrez.email = "bazefi01@thu.de"
Entrez.tool  = "biomed-rag-ingester"
Entrez.api_key = os.getenv("NCBI_API_KEY")  # optional, speeds up

# ---- Paths (project-root aware) ----
ROOT = Path(__file__).resolve().parents[1]  # repo root

# NEW: prefer processed outputs from the new mapper
MAP_JSON  = ROOT / "data_pipeline" / "data" / "processed" / "pmid_pmcid_map_120k.json"
MAP_JSONL = ROOT / "data_pipeline" / "data" / "processed" / "pmid_pmcid_map_120k.jsonl"

OUTDIR = ROOT / "data_pipeline" / "data" / "raw" / "pmc_xml"
OUTDIR.mkdir(parents=True, exist_ok=True)

def _log(*a): print(*a, file=sys.stderr)

def load_pmcids() -> list[str]:
    pmcids = []
    if MAP_JSON.exists():
        m = json.loads(MAP_JSON.read_text(encoding="utf-8"))
        pmcids.extend(m.values())
        _log(f"[map] Loaded {len(m)} pairs from {MAP_JSON}")
    elif MAP_JSONL.exists():
        n = 0
        with MAP_JSONL.open(encoding="utf-8") as f:
            for ln in f:
                try:
                    r = json.loads(ln)
                    if r.get("pmcid"):
                        pmcids.append(str(r["pmcid"]))
                        n += 1
                except Exception:
                    pass
        _log(f"[map] Loaded {n} pairs from {MAP_JSONL}")
    else:
        _log(f"[error] No map found at {MAP_JSON} or {MAP_JSONL}")
        sys.exit(1)

    # de-duplicate, preserve order
    seen, uniq = set(), []
    for p in pmcids:
        if p and p not in seen:
            seen.add(p); uniq.append(p)
    _log(f"[map] Unique PMCIDs: {len(uniq)}")
    return uniq

def fetch_pmc_xml_batch(ids: list[str]) -> bytes:
    with Entrez.efetch(db="pmc", id=",".join(ids), retmode="xml") as h:
        return h.read()

def iter_articles(xml_bytes: bytes):
    parser = ET.XMLParser(recover=True, huge_tree=True)
    root = ET.fromstring(xml_bytes, parser=parser)
    for art in root.xpath(".//*[local-name()='article']"):
        yield ET.tostring(art, encoding="utf-8", xml_declaration=True)

def pmcid_from_article_bytes(b: bytes) -> Optional[str]:
    m = re.search(br"<article-id[^>]*pub-id-type=[\"']pmcid[\"'][^>]*>PMC(\d+)</article-id>", b)
    return f"PMC{m.group(1).decode()}" if m else None

def main():
    pmcids = load_pmcids()
    # efetch expects numeric IDs for PMC
    numeric_ids = [re.sub(r"^PMC", "", p) for p in pmcids]

    ok = fail = 0
    BATCH = 50

    _log(f"[run] Fetching XML for {len(numeric_ids)} PMCIDs → {OUTDIR}")

    for i in range(0, len(numeric_ids), BATCH):
        chunk = numeric_ids[i:i+BATCH]
        try:
            xml_bytes = fetch_pmc_xml_batch(chunk)
            got = 0
            for j, art_bytes in enumerate(iter_articles(xml_bytes)):
                pmcid = pmcid_from_article_bytes(art_bytes) or f"PMC_unknown_{i}_{j}"
                out = OUTDIR / f"{pmcid}_{i}_{j}.xml"
                out.write_bytes(art_bytes)
                ok += 1; got += 1
            _log(f"[batch] {i+len(chunk):>5}/{len(numeric_ids)} → saved {got} articles")
        except Exception as e:
            _log("[batch error]", e)
            fail += len(chunk)
        time.sleep(0.34)  # be polite

    print(f"XML downloaded: {ok} articles saved, {fail} failed")
    print("XML files on disk:", len(list(OUTDIR.glob('*.xml'))))

if __name__ == "__main__":
    main()
