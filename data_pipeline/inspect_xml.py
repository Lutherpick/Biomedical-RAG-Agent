# data_pipeline/inspect_xml.py
from __future__ import annotations
from pathlib import Path
from lxml import etree as ET

ROOT     = Path(__file__).resolve().parents[1]
XML_DIR  = ROOT / "data_pipeline" / "data" / "raw" / "pmc_xml"

def first_text(root, xp):
    n = root.xpath(xp)
    if not n: return ""
    if isinstance(n, list): n = n[0]
    return "".join(n.itertext()) if hasattr(n, "itertext") else str(n)

def label(h: str):
    h = (h or "").lower()
    for k in ["introduction","materials and methods","methods","results","discussion","results and discussion"]:
        if h.startswith(k): return k
    return None

def main():
    files = sorted(XML_DIR.glob("PMC*.xml"))
    if not files:
        print("No XML files found."); return
    p = files[0]
    parser = ET.XMLParser(recover=True, huge_tree=True)
    root = ET.parse(str(p), parser=parser).getroot()
    if root.tag.lower().endswith("pmc-articleset"):
        art = root.find(".//{*}article")
        if art is not None: root = art

    print("File:", p.name)
    print("PMCID:", first_text(root, ".//article-id[@pub-id-type='pmcid']/text()"))
    print("PMID:",  first_text(root, ".//article-id[@pub-id-type='pmid']/text()"))
    print("DOI:",   first_text(root, ".//article-id[@pub-id-type='doi']/text()"))
    print("Journal:", first_text(root, ".//journal-title"))
    print("Year:",    first_text(root, "(.//pub-date/year | .//pub-date[@pub-type='ppub']/year | .//pub-date[@pub-type='epub']/year)[1]"))
    print("Title:",   first_text(root, ".//article-title"))

    body = root.find(".//{*}body")
    if body is None:
        print("No <body>"); return

    secs = []
    for sec in body.findall(".//{*}sec"):
        t = first_text(sec, "./{*}title")
        lab = label(t)
        if lab: secs.append(lab)

    paras = body.xpath('.//*[local-name()="p"]')
    figs  = body.xpath('.//*[local-name()="fig"]')

    print("I/M/R/D sections:", secs[:12], f"(total {len(secs)})")
    print("Body paragraphs:", len(paras))
    print("Figures:", len(figs))

if __name__ == "__main__":
    main()
