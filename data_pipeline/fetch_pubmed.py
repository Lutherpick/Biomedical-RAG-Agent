# pip install biopython
import os, random
from Bio import Entrez
import time, json, xml.etree.ElementTree as ET
import re
from pathlib import Path
from lxml import etree as ET


Entrez.email = "bazefi01@thu.de"
Entrez.api_key = os.getenv("NCBI_API_KEY")  # set via env
Entrez.tool  = "biomed-rag-ingester"

def _sleep(): time.sleep(0.12 + random.uniform(0, 0.05))
QUERY = '''
(
"Animal Experimentation"[MeSH] OR
"Models, Animal"[MeSH] OR
"Disease Models, Animal"[MeSH] OR
"Preclinical Studies as Topic"[MeSH] OR
"Drug Evaluation, Preclinical"[MeSH] OR
"Toxicity Tests"[MeSH] OR
"In Vivo Techniques"[MeSH]
)
AND (Animals[mh] NOT Humans[mh])
AND (
"animal experiment*"[tiab] OR
"animal model*"[tiab] OR
"preclinical"[tiab] OR
"in vivo"[tiab] OR
mouse[tiab] OR mice[tiab] OR
rat[tiab] OR rats[tiab] OR
rabbit[tiab] OR rabbits[tiab] OR
dog[tiab] OR dogs[tiab] OR canine[tiab] OR
pig[tiab] OR pigs[tiab] OR swine[tiab] OR porcine[tiab] OR
sheep[tiab] OR ovine[tiab] OR
cattle[tiab] OR bovine[tiab] OR
zebrafish[tiab] OR
xenopus[tiab] OR
"chick embryo"[tiab]
)
AND ("2010/01/01"[dp] : "2025/12/31"[dp])
AND hasabstract[text]
AND (english[lang] OR german[lang])

'''.strip()
#AND free full text[sb]
OUT_DIR = Path("pubmed_export"); OUT_DIR.mkdir(exist_ok=True)

def search_ids(query: str, n: int = 1000, step: int = 200):
    ids = []
    for retstart in range(0, n, step):
        with Entrez.esearch(db="pubmed", term=query, retmax=step, retstart=retstart) as h:
            rec = Entrez.read(h)
        ids.extend(rec.get("IdList", []))
        if len(rec.get("IdList", [])) < step:
            break
        time.sleep(0.34)  # <= 3 req/sec
    return ids[:n]

def fetch_xml_batched(pmids, batch=200):
    """Return concatenated PubMed XML as bytes, fetching in batches."""
    all_bytes = b'<?xml version="1.0" encoding="UTF-8"?><PubmedArticleSet>'
    for i in range(0, len(pmids), batch):
        chunk = pmids[i:i+batch]
        with Entrez.efetch(db="pubmed", id=",".join(chunk), rettype="xml", retmode="xml") as h:
            chunk_bytes = h.read()
            # 🔧 Remove any XML declarations with regex
            chunk_bytes = re.sub(br'<\?xml[^>]*\?>', b'', chunk_bytes)
            all_bytes += chunk_bytes
        time.sleep(0.34)
    all_bytes += b"</PubmedArticleSet>"
    return all_bytes

def xml_to_jsonl(xml_bytes: bytes, out_path: Path):
    # 🔧 Clean again before parsing
    xml_clean = re.sub(br'<\?xml[^>]*\?>', b'', xml_bytes).strip()
    parser = ET.XMLParser(recover=True)        # <-- ignores minor syntax errors
    root = ET.fromstring(xml_clean, parser=parser)
    out = open(out_path, "w", encoding="utf-8")

    for art in root.findall(".//PubmedArticle"):
        title_el = art.find(".//ArticleTitle")
        title = "".join(title_el.itertext()).strip() if title_el is not None else None

        abst_el = art.find(".//Abstract")
        abstract = " ".join("".join(x.itertext()).strip() for x in (abst_el.findall(".//AbstractText") if abst_el is not None else [])) or None

        journal_el = art.find(".//Journal/Title")
        journal = journal_el.text if journal_el is not None else None

        year_el = art.find(".//JournalIssue/PubDate/Year")
        year = int(year_el.text) if (year_el is not None and (year_el.text or "").isdigit()) else None

        doi = None
        for aid in art.findall(".//ArticleIdList/ArticleId"):
            if (aid.get("IdType", "") or "").lower() == "doi":
                doi = aid.text
                break

        authors = []
        for a in art.findall(".//AuthorList/Author"):
            last = (a.findtext("LastName") or "").strip()
            fore = (a.findtext("ForeName") or "").strip()
            name = (fore + " " + last).strip()
            if name:
                authors.append(name)

        pmid = art.findtext(".//PMID")
        langs = [l.text for l in art.findall(".//Language") if l is not None and l.text]

        rec = {
            "pmid": pmid,
            "title": title,
            "authors": authors,
            "journal": journal,
            "year": year,
            "doi": doi,
            "language": langs[0] if langs else None,
            "publication_date": None,
            "kind_doc": None,
            "abstract": abstract
        }
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    out.close()

if __name__ == "__main__":
    pmids = search_ids(QUERY, n=3000, step=200)
    print(f"Fetched {len(pmids)} PMIDs")

    xml_bytes = fetch_xml_batched(pmids, batch=200)

    # ✅ write bytes safely
    (OUT_DIR / "pubmed_first_1000.xml").write_bytes(xml_bytes)

    # Convert to JSONL
    xml_to_jsonl(xml_bytes, OUT_DIR / "pubmed_first_1000.jsonl")
    print("Saved: pubmed_export/pubmed_first_1000.xml and pubmed_first_1000.jsonl")
