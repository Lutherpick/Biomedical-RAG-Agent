#!/usr/bin/env python3
# xml_paragraph_chunker_paragraph_first.py
# Paragraph-first chunker for NIHMS-style XML converted from PDFs.
# - Primary unit: paragraph; do not merge across paragraphs.
# - Never cut inside a sentence; split only if paragraph exceeds token_limit.
# - Token limit is a fallback (default 800).
# - Respects section boundaries.
# - Figure handling: a figure and its caption become one chunk when present.
# - Furniture cleanup: remove "Author Manuscript", page numbers, journal headers.

import argparse, json, re, sys, glob
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Optional
import xml.etree.ElementTree as ET

# ----------------------------
# Optional tokenizer (tiktoken). Fallback: whitespace split.
# ----------------------------
def _build_tokenizer():
    try:
        import tiktoken  # type: ignore
        enc = tiktoken.get_encoding("cl100k_base")
        return lambda s: len(enc.encode(s))
    except Exception:
        return lambda s: len(s.split())

count_tokens = _build_tokenizer()

# ----------------------------
# Sentence splitting: conservative regex. Do not cut abbreviations crudely.
# ----------------------------
_SENT_BOUNDARY = re.compile(
    r'(?<!\b(?:Mr|Mrs|Ms|Dr|Prof|Fig|Figs|Eq|Eqs|Ref|Refs|et|al|No|Nos|Vol|vs|St|Mt)\.)'
    r'(?<=[\.!?])(?=\s+["”’)\]]*\s*)'
)

def split_sentences(text: str) -> List[str]:
    normalized = re.sub(r'[ \t]*\n[ \t]*', ' ', text.strip())
    if count_tokens(normalized) <= 800:
        return [normalized]
    parts, last = [], 0
    for m in _SENT_BOUNDARY.finditer(normalized):
        i = m.end()
        seg = normalized[last:i].strip()
        if seg:
            parts.append(seg)
        last = i
    tail = normalized[last:].strip()
    if tail:
        parts.append(tail)
    return parts or [normalized]

# ----------------------------
# Page furniture cleanup
# Only removes clearly irrelevant artifacts.
# ----------------------------
_RE_PAGE = re.compile(r'^\s*Page\s+\d+(?:\s+of\s+\d+)?\s*$', re.IGNORECASE)
_RE_RUNNING_HEAD = re.compile(r'^\s*(?:This article is a US Government work.*|All rights reserved.*)\s*$', re.IGNORECASE)

def likely_letter_staircase(text: str) -> bool:
    # True if a "A\nu\nt\nh\no\nr\n ..." staircase dominates.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False
    singles = sum(1 for ln in lines if len(ln) <= 2 and ln.isalpha())
    return singles >= max(8, int(0.6 * len(lines)))  # many single-letter lines

def strip_furniture(text: str, journal_hint: Optional[str] = None) -> str:
    # Remove obvious page furniture while preserving scientific content.
    # Steps:
    # 1) Drop lines that are pure page numbers or standard running heads.
    # 2) Drop journal banner lines if they match the journal attribute exactly.
    # 3) Remove staircase artifacts like "Author Manuscript".
    lines = text.splitlines()
    out = []
    for ln in lines:
        s = ln.strip()
        if not s:
            out.append(ln)
            continue
        if _RE_PAGE.match(s):
            continue
        if _RE_RUNNING_HEAD.match(s):
            continue
        if journal_hint and s == journal_hint:
            continue
        out.append(ln)
    cleaned = "\n".join(out)
    if likely_letter_staircase(cleaned):
        # Most of the paragraph is a staircase; drop it entirely.
        return ""
    # Also remove explicit "Author Manuscript" even if repeated
    compact = re.sub(r'\s+', '', cleaned).lower()
    if "authormanuscript" in compact and len(compact) <= 200:
        return ""
    return cleaned

# ----------------------------
# Paragraph segmentation: primary unit.
# ----------------------------
_BLANKS = re.compile(r'\n\s*\n+', re.MULTILINE)

def extract_paragraphs(block_text: str) -> List[str]:
    if not block_text:
        return []
    if _BLANKS.search(block_text):
        paras = [p for p in _BLANKS.split(block_text) if p.strip()]
    else:
        paras = [block_text] if block_text.strip() else []
    return paras

# ----------------------------
# Chunk a single paragraph with token fallback (never mid-sentence).
# ----------------------------
def chunk_paragraph(paragraph_text: str, token_limit: int) -> List[str]:
    if not paragraph_text.strip():
        return []
    tok = count_tokens(paragraph_text)
    if tok <= token_limit:
        return [paragraph_text]
    sents = split_sentences(paragraph_text)
    chunks, buf, buf_tok = [], [], 0
    for s in sents:
        st = count_tokens(s)
        if buf and buf_tok + st > token_limit:
            chunks.append(" ".join(buf).strip())
            buf, buf_tok = [s], st
        else:
            buf.append(s)
            buf_tok += st
    if buf:
        chunks.append(" ".join(buf).strip())
    return [c for c in chunks if c.strip()]

# ----------------------------
# XML helpers: NIHMS-like structure and generic <fig>/<caption>.
# ----------------------------
def read_header(root: ET.Element) -> Dict[str, Optional[str]]:
    return {
        "pmcid": root.attrib.get("pmcid"),
        "pmid": root.attrib.get("pmid"),
        "nihmsid": root.attrib.get("nihmsid"),
        "title": root.attrib.get("title"),
        "journal": root.attrib.get("journal"),
        "year": root.attrib.get("year"),
        "source_file": root.attrib.get("source_file"),
        "schema_version": root.attrib.get("schema_version"),
        "generated_from": root.attrib.get("generated_from"),
    }

def iter_blocks_and_figs(root: ET.Element) -> Iterable[Tuple[Dict[str, str], str, str]]:
    """
    Yields tuples: (meta, text, kind)
      kind in {"paragraph_block", "figure_caption"}
      For paragraph_block: text comes from <block><text>
      For figure_caption: tries <fig>/<caption> and packs together as one unit.
    """
    # NIHMS-converted custom structure
    sections = root.find("./sections")
    if sections is not None:
        for sec in sections.findall("./section"):
            sec_name = sec.attrib.get("name", "")
            sec_idx = sec.attrib.get("index", "")
            for sub in sec.findall("./subsection"):
                sub_name = sub.attrib.get("name", "")
                sub_idx = sub.attrib.get("index", "")
                # Paragraph-like blocks
                for blk in sub.findall("./block"):
                    meta = {
                        "section": sec_name,
                        "section_index": sec_idx,
                        "subsection": sub_name,
                        "subsection_index": sub_idx,
                    }
                    inner_sec = blk.findtext("./section")
                    inner_sub = blk.findtext("./subsection")
                    if inner_sec: meta["section"] = inner_sec
                    if inner_sub is not None: meta["subsection"] = inner_sub
                    txt = blk.findtext("./text") or ""
                    yield meta, txt, "paragraph_block"
                # Figure + caption as single unit if present in this subsection
                for fig in sub.findall(".//fig"):
                    # collect label + caption
                    label = (fig.findtext("./label") or "").strip()
                    cap = (fig.findtext("./caption") or "").strip()
                    # Some conversions wrap caption under <caption><p>...</p></caption>
                    if not cap:
                        cap = " ".join(p.text.strip() for p in fig.findall("./caption//p") if (p.text or "").strip())
                    text = ((label + " ") if label else "") + cap
                    if text.strip():
                        meta = {
                            "section": sec_name,
                            "section_index": sec_idx,
                            "subsection": sub_name,
                            "subsection_index": sub_idx,
                            "figure_label": label,
                        }
                        yield meta, text, "figure_caption"

    # Generic fallback: plain JATS-like
    # Captions
    for fig in root.findall(".//fig"):
        label = (fig.findtext("./label") or "").strip()
        cap = (fig.findtext("./caption") or "").strip()
        if not cap:
            cap = " ".join(p.text.strip() for p in fig.findall("./caption//p") if (p.text or "").strip())
        text = ((label + " ") if label else "") + cap
        if text.strip():
            meta = {
                "section": "",
                "section_index": "",
                "subsection": "",
                "subsection_index": "",
                "figure_label": label,
            }
            yield meta, text, "figure_caption"

# ----------------------------
# Main per-file processing
# ----------------------------
def process_file(xml_path: Path, token_limit: int, drop_furniture: bool = True) -> Iterable[Dict]:
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    hdr = read_header(root)
    journal_hint = hdr.get("journal") or None
    file_meta = {"file_path": str(xml_path)}
    para_counter = 0

    for meta, raw_text, kind in iter_blocks_and_figs(root):
        if kind == "figure_caption":
            # One chunk per figure caption (+label). Do not split unless huge.
            txt = raw_text
            if drop_furniture:
                txt = strip_furniture(txt, journal_hint)
            txt = txt.strip()
            if not txt:
                continue
            # If a caption is extremely long, split on sentences with fallback
            chunks = chunk_paragraph(txt, token_limit)
            for i, ch in enumerate(chunks):
                rec = {
                    **hdr, **file_meta,
                    "section": meta.get("section", ""),
                    "subsection": meta.get("subsection", ""),
                    "section_index": meta.get("section_index", ""),
                    "subsection_index": meta.get("subsection_index", ""),
                    "section_path": "/".join([s for s in [meta.get("section",""), meta.get("subsection","")] if s]),
                    "figure_label": meta.get("figure_label", ""),
                    "paragraph_index_global": None,
                    "paragraph_index_in_block": None,
                    "chunk_index_in_paragraph": i,
                    "chunk_total_in_paragraph": len(chunks),
                    "type": "figure_caption",
                    "text": ch,
                    "token_count": count_tokens(ch),
                }
                yield rec
            continue

        # kind == paragraph_block
        text = raw_text if not drop_furniture else strip_furniture(raw_text, journal_hint)
        paras = extract_paragraphs(text)
        for p_idx, para in enumerate(paras):
            p = para.strip()
            if not p:
                continue
            para_counter += 1
            para_chunks = chunk_paragraph(p, token_limit)
            for c_idx, chunk in enumerate(para_chunks):
                rec = {
                    **hdr, **file_meta,
                    "section": meta.get("section", ""),
                    "subsection": meta.get("subsection", ""),
                    "section_index": meta.get("section_index", ""),
                    "subsection_index": meta.get("subsection_index", ""),
                    "section_path": "/".join([s for s in [meta.get("section",""), meta.get("subsection","")] if s]),
                    "paragraph_index_global": para_counter,
                    "paragraph_index_in_block": p_idx,
                    "chunk_index_in_paragraph": c_idx,
                    "chunk_total_in_paragraph": len(para_chunks),
                    "type": "paragraph",
                    "text": chunk,
                    "token_count": count_tokens(chunk),
                }
                yield rec

# ----------------------------
# CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Paragraph-first chunker for converted NIHMS/JATS XML.")
    ap.add_argument("--inp", required=True,
                    help="Input XML path or glob. Example: src\\PMC*.xml")
    ap.add_argument("--out", required=True,
                    help="Output JSONL path.")
    ap.add_argument("--max-tokens", type=int, default=800,
                    help="Token hard limit per chunk (fallback for long paragraphs).")
    ap.add_argument("--keep-furniture", action="store_true",
                    help="Do NOT remove page furniture like 'Author Manuscript' or page numbers.")
    args = ap.parse_args()

    # Expand paths
    paths: List[Path]
    p = Path(args.inp)
    if any(ch in args.inp for ch in ["*", "?", "["]):
        paths = [Path(x) for x in glob.glob(args.inp)]
    elif p.is_dir():
        paths = sorted(list(p.glob("*.xml")))
    else:
        paths = [p]

    if not paths:
        print(f"No input files matched: {args.inp}", file=sys.stderr)
        sys.exit(2)

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    with outp.open("w", encoding="utf-8") as f:
        for xml_path in paths:
            for rec in process_file(
                    xml_path,
                    token_limit=args.max_tokens,
                    drop_furniture=not args.keep_furniture
            ):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
