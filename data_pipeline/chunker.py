# data_pipeline/chunker.py
# Requirements: pip install lxml regex
from __future__ import annotations
from pathlib import Path
from lxml import etree as ET
from lxml.etree import XPathEvalError
from typing import List, Dict, Iterable, Tuple, Optional
import json, re, unicodedata

# -------- Paths --------
ROOT      = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data_pipeline" / "data"
IN_XML    = DATA_ROOT / "raw" / "pmc_xml"
OUT_DIR   = DATA_ROOT / "processed" / "chunks"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSONL = OUT_DIR / "chunks.jsonl"

# -------- Chunking params --------
TARGET = 400         # ~400 ± 50 tokens, section-pure, no overlap
CAPTION_MAX = 300    # captions ≤300
MIN_CHUNK = 100      # drop tiny tails

# Section names (case-insensitive)
SEC_MAP = {
    "introduction": "introduction",
    "materials and methods": "methods",
    "methods": "methods",
    "results": "results",
    "discussion": "discussion",
    "results and discussion": "results_discussion",
}

token_pat = re.compile(r"\w+|[^\w\s]", re.UNICODE)

def tokenize(s: str) -> List[str]:
    return token_pat.findall(s or "")

def detokenize(tokens: List[str]) -> str:
    s = " ".join(tokens)
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    return s

def norm_text(s: str) -> str:
    if not s: return ""
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ---------- Safe XPath helpers (namespace-agnostic) ----------
def first_text_safe(root: ET._Element, xpaths: List[str]) -> str:
    for xp in xpaths:
        try:
            nodes = root.xpath(xp)
        except XPathEvalError:
            continue
        if not nodes:
            continue
        n = nodes[0]
        if isinstance(n, str):
            return norm_text(n)
        return norm_text("".join(n.itertext()))
    return ""

def all_nodes(root: ET._Element, xp: str) -> List[ET._Element]:
    try:
        out = root.xpath(xp)
        return [n for n in out if isinstance(n, ET._Element)]
    except XPathEvalError:
        return []

# ---------------- XML extraction ----------------
def meta_from_article(root: ET._Element) -> Dict[str, str]:
    pmid = first_text_safe(root, [".//*[local-name()='article-id' and @pub-id-type='pmid']/text()"])
    pmcid = first_text_safe(root, [".//*[local-name()='article-id' and @pub-id-type='pmcid']/text()"])
    doi = first_text_safe(root, [
        ".//*[local-name()='article-id' and @pub-id-type='doi']/text()",
        ".//*[local-name()='article-id' and @pub-id-type='DOI']/text()",
    ])
    journal = first_text_safe(root, [
        ".//*[local-name()='journal-title']",
        ".//*[local-name()='journal-title-group']/*[local-name()='journal-title']",
    ])
    year = first_text_safe(root, [
        ".//*[local-name()='pub-date']/*[local-name()='year']/text()",
        ".//*[local-name()='year']/text()",
    ])
    title = first_text_safe(root, [".//*[local-name()='article-title']"])
    return {"pmid": pmid, "pmcid": pmcid, "doi": doi, "journal": journal, "year": year, "title": title}

def get_body(root: ET._Element) -> Optional[ET._Element]:
    nodes = all_nodes(root, ".//*[local-name()='body']")
    return nodes[0] if nodes else None

def section_label(h: str) -> Optional[str]:
    if not h: return None
    low = h.lower()
    if low in SEC_MAP: return SEC_MAP[low]
    for k, v in SEC_MAP.items():
        if low.startswith(k): return v
    return None

def iter_sections(body: ET._Element) -> Iterable[Tuple[str, ET._Element, str]]:
    """Yield (section_type, section_node, path_hint) for I/M/R/D sections only."""
    secs = all_nodes(body, ".//*[local-name()='sec']")
    for idx, sec in enumerate(secs):
        heading = first_text_safe(sec, ["./*[local-name()='title'][1]"])
        lab = section_label(heading)
        if not lab:
            continue
        path_hint = f"body/sec[{idx}]"
        yield (lab, sec, path_hint)

def collect_paragraphs(sec: ET._Element) -> List[str]:
    ps = all_nodes(sec, ".//*[local-name()='p']")
    out = []
    for p in ps:
        t = norm_text("".join(p.itertext()))
        if t: out.append(t)
    return out

def collect_captions(body: ET._Element) -> List[str]:
    caps = []
    for fig in all_nodes(body, ".//*[local-name()='fig']"):
        cap = first_text_safe(fig, [".//*[local-name()='caption']"])
        if cap: caps.append(cap)
    return caps

# ---------------- Chunk emission ----------------
def chunk_stream(meta: Dict[str, str], section_type: str, section_path: str, text: str):
    toks = tokenize(text)
    if not toks: return
    size = TARGET
    for i in range(0, len(toks), size):
        ch = toks[i:i+size]
        if len(ch) < MIN_CHUNK and i != 0:
            break
        yield {
            "pmcid": meta["pmcid"],
            "pmid": meta["pmid"],
            "doi": meta["doi"],
            "journal": meta["journal"],
            "year": meta["year"],
            "title": meta["title"],
            "topic": "",  # optional placeholder
            "section_type": section_type,
            "section_path": section_path,
            "chunk_index": i // size,
            "token_count": len(ch),
            "text": detokenize(ch),
            "is_caption": False,
            "figure_ids": [],
            "image_paths": [],
        }

def main():
    if OUT_JSONL.exists():
        OUT_JSONL.unlink()

    files = sorted(IN_XML.glob("PMC*.xml"))
    total_docs = 0
    total_chunks = 0

    with OUT_JSONL.open("w", encoding="utf-8") as out:
        for x in files:
            try:
                parser = ET.XMLParser(recover=True, huge_tree=True)
                root = ET.parse(str(x), parser=parser).getroot()

                # If wrapped, descend into <article>
                if root.tag and root.tag.lower().endswith("pmc-articleset"):
                    arts = all_nodes(root, ".//*[local-name()='article']")
                    if arts: root = arts[0]

                meta = meta_from_article(root)
                if not meta.get("pmcid"):
                    m = re.findall(r"(PMC\d+)", x.stem)
                    if m: meta["pmcid"] = m[0]

                body = get_body(root)
                if body is None:
                    continue

                # Title
                if meta.get("title"):
                    for rec in chunk_stream(meta, "title", "front/article-title", meta["title"]):
                        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        total_chunks += 1

                # Abstract
                abstract = first_text_safe(root, [".//*[local-name()='abstract']"])
                if abstract:
                    for rec in chunk_stream(meta, "abstract", "front/abstract", abstract):
                        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        total_chunks += 1

                # I/M/R/D sections
                for sec_type, sec_node, sec_path in iter_sections(body):
                    paras = collect_paragraphs(sec_node)
                    if not paras:
                        continue
                    text = norm_text("\n\n".join(paras))
                    for rec in chunk_stream(meta, sec_type, sec_path, text):
                        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        total_chunks += 1

                # Figure captions (text only), each ≤ CAPTION_MAX tokens in a single chunk
                for cap in collect_captions(body):
                    toks = tokenize(cap)[:CAPTION_MAX]
                    if not toks:
                        continue
                    rec = {
                        "pmcid": meta["pmcid"],
                        "pmid": meta["pmid"],
                        "doi": meta["doi"],
                        "journal": meta["journal"],
                        "year": meta["year"],
                        "title": meta["title"],
                        "topic": "",
                        "section_type": "figure_caption",
                        "section_path": "body/fig",
                        "chunk_index": 0,
                        "token_count": len(toks),
                        "text": detokenize(toks),
                        "is_caption": True,
                        "figure_ids": [],
                        "image_paths": [],
                    }
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    total_chunks += 1

                total_docs += 1

            except Exception as e:
                print(f"SKIP {x.name}: {e}")

    print(f"OK chunked docs: {total_docs}")
    print(f"OK total chunks: {total_chunks}")
    print(f"Wrote: {OUT_JSONL}")

if __name__ == "__main__":
    main()
