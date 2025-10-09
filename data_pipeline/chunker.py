# pip install lxml
from pathlib import Path
from lxml import etree as ET
import json, re

IN_XML = Path("data/raw/pmc_xml")
IN_TXT = Path("data/processed/pdf_text")
OUT    = Path("data/processed/chunks"); OUT.mkdir(parents=True, exist_ok=True)
TARGET_TOKENS = 600; OVERLAP = 0.1

def tokenize(s): return re.findall(r"\w+|[^\w\s]", s)

def chunks(tokens, size, overlap):
    step = max(1, int(size*(1-overlap)))
    for i in range(0, len(tokens), step):
        yield tokens[i:i+size]

def xml_to_text(path: Path) -> str:
    # tolerant parser handles minor XML errors
    parser = ET.XMLParser(recover=True, huge_tree=True)
    try:
        tree = ET.parse(str(path), parser=parser)
    except ET.XMLSyntaxError:
        return ""
    root = tree.getroot()
    # if the file still contains a <pmc-articleset>, dive into the first <article>
    if root.tag.lower().endswith("pmc-articleset"):
        art = root.find(".//article")
        if art is not None:
            root = art
    # collect paragraphs from body
    texts = ["".join(p.itertext()).strip() for p in root.findall(".//body//p")]
    text = "\n\n".join(t for t in texts if t)
    return text

def emit(doc_id, text):
    if not text or text.strip() == "":
        print(f"SKIP empty text: {doc_id}")
        return
    toks = tokenize(text)
    if not toks:
        print(f"SKIP no tokens: {doc_id}")
        return
    total = 0
    for idx, ch in enumerate(chunks(toks, TARGET_TOKENS, OVERLAP)):
        total += 1
        rec = {
            "doc_id": doc_id,
            "chunk_index": idx,
            "chunk_total": None,  # can be backfilled later
            "text": " ".join(ch),
        }
        (OUT/f"{doc_id}__{idx:04d}.json").write_text(
            json.dumps(rec, ensure_ascii=False), encoding="utf-8"
        )
    print(f"OK {doc_id} -> {total} chunks")

# XML first
for x in IN_XML.glob("PMC*.xml"):
    emit(x.stem, xml_to_text(x))

# PDF text fallback
for t in IN_TXT.glob("*.txt"):
    emit(t.stem, t.read_text(encoding="utf-8"))

print("Chunking done.")
