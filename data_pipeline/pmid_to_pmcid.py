# pip install biopython
from Bio import Entrez; Entrez.email="bazefi01@thu.de"; Entrez.tool="biomed-rag"
from pathlib import Path; import json, time
IN = Path("pubmed_export/pubmed_first_1000.jsonl")
OUT = Path("pubmed_export/pmid_pmcid_map.json")

pmids=[]
with IN.open(encoding="utf-8") as f:
    for line in f:
        rec=json.loads(line);
        if rec.get("pmid"): pmids.append(rec["pmid"])

mapping={}
for i in range(0,len(pmids),200):
    chunk=pmids[i:i+200]
    with Entrez.elink(dbfrom="pubmed", db="pmc", id=",".join(chunk)) as h:
        data=Entrez.read(h)
    for ls in data:
        pid = ls["IdList"][0] if ls.get("IdList") else None
        for db in ls.get("LinkSetDb", []):
            if db.get("DbTo")=="pmc":
                for link in db.get("Link", []):
                    mapping[pid]=f"PMC{link['Id']}"
    time.sleep(0.34)

OUT.write_text(json.dumps(mapping, indent=2))
print(f"PMC links: {len(mapping)} saved → {OUT}")
