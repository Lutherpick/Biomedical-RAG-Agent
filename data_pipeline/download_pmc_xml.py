# download_pmc_xml.py  (use E-utilities, not /xml/ URL)
# pip install biopython
from Bio import Entrez
from pathlib import Path
import json, time, re

Entrez.email = "bazefi01@thu.de"          # <-- set yours
Entrez.tool  = "biomed-rag-ingester"

MAP = Path("pubmed_export/pmid_pmcid_map.json")
OUTDIR = Path("data/raw/pmc_xml"); OUTDIR.mkdir(parents=True, exist_ok=True)

mapping = json.loads(MAP.read_text(encoding="utf-8"))
pmcid_list = [mapping[k] for k in mapping]          # e.g., ["PMC12345", ...]
# efetch wants the numeric id (or PMCID with PMC prefix works in practice, but be safe):
ids = [re.sub(r"^PMC", "", x) for x in pmcid_list]

ok = fail = 0
BATCH = 50
for i in range(0, len(ids), BATCH):
    chunk_ids = ids[i:i+BATCH]
    try:
        with Entrez.efetch(db="pmc", id=",".join(chunk_ids), retmode="xml") as h:
            xml_bytes = h.read()  # bytes
        # Split into individual articles
        # The response is <pmc-articleset>…<article>… structures.
        # Save per <article> using a simple split on closing tag.
        parts = xml_bytes.split(b"</article>")
        for part in parts:
            if b"<article" not in part:
                continue
            # grab the PMCID inside if present
            pmcid = None
            m = re.search(br"<article-id[^>]*pub-id-type=\"pmcid\"[^>]*>PMC(\d+)</article-id>", part)
            if m:
                pmcid = f"PMC{m.group(1).decode()}"
            # fallback to loop index if no pmcid tag
            fname = f"{pmcid or 'PMC_unknown'}_{i}.xml"
            out = OUTDIR / fname
            out.write_bytes(part + b"</article>")
            ok += 1
    except Exception:
        fail += len(chunk_ids)
    time.sleep(0.34)

print(f"XML downloaded (via efetch): {ok} articles saved, {fail} failed")

from pathlib import Path
from lxml import etree as ET

xml_dir = Path("data/raw/pmc_xml")
files = list(xml_dir.glob("PMC*.xml"))
print("XML files:", len(files))

root = ET.parse(str(files[0])).getroot()
body_text = " ".join("".join(p.itertext()).strip()
                     for p in root.findall(".//body//p"))[:800]
print("Preview:\n", body_text)
