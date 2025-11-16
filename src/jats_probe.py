# jats_probe.py — print ordered full text with true paragraph boundaries
# Usage (PowerShell):
#   python .\jats_probe.py .\PMC9107928.xml --full > PMC9107928_full.txt
#   python .\jats_probe.py .\pmc_chunker\out\PMC8627534_out.xml --full --section "^Introduction"

import sys, glob, re
from pathlib import Path
import argparse
import xml.etree.ElementTree as ET

# Reuse the exact cleaners and paragraph splitter from chunker.py
from chunker import clean_furniture, split_paragraphs_clean

def ln(tag: str) -> str:
    return tag.split("}", 1)[1] if tag and tag.startswith("{") else (tag or "")

def iter_local(e, name):
    for n in e.iter():
        if ln(n.tag) == name:
            yield n

def text_join(node) -> str:
    # Keep a single paragraph from JATS <p>; collapse intra-para whitespace only.
    return " ".join(" ".join(node.itertext()).split())

def pick_xml(path_or_dir: str) -> Path:
    p = Path(path_or_dir)
    if p.is_file(): return p
    if p.is_dir():
        m = sorted(Path(p).glob("*.xml"))
        if m: return m[0]
    g = glob.glob(path_or_dir)
    if g: return Path(g[0])
    raise SystemExit(f"No XML found: {path_or_dir}")

def load_jats_records(root):
    """Return cleaned paragraphs from JATS (<abstract>, <sec>/<p>)."""
    recs = []
    # Abstract first (if present)
    abs_elt = next(iter_local(root, "abstract"), None)
    if abs_elt is not None:
        ps = list(iter_local(abs_elt, "p"))
        if ps:
            for k, p in enumerate(ps):
                txt = clean_furniture(text_join(p))
                if txt:
                    recs.append(("Abstract", "", txt, -1, k))
        else:
            txt = clean_furniture(text_join(abs_elt))
            if txt:
                recs.append(("Abstract", "", txt, -1, 0))

    # Body sections
    sec_idx = -1
    for sec in iter_local(root, "sec"):
        sec_idx += 1
        # pick first <title> under sec
        title = ""
        for t in iter_local(sec, "title"):
            title = text_join(t); break
        sub_idx = -1
        for p in iter_local(sec, "p"):
            sub_idx += 1
            txt = clean_furniture(text_join(p))
            if txt:
                recs.append((title or "sec", "", txt, sec_idx, sub_idx))
    return recs

def load_nihms_records(root):
    """Read <flat_records>/<record>. Split each record into true paragraphs by blank lines."""
    frs = list(iter_local(root, "flat_records"))
    if not frs: return []
    out = []
    for r in iter_local(frs[0], "record"):
        fields = {ln(c.tag): (c.text or "") for c in list(r)}
        section = (fields.get("section") or "").strip()
        subsection = (fields.get("subsection") or "").strip()
        if subsection.lower() == "author manuscript":  # drop NIHMS header line entirely
            continue
        # Remove “Page N” pseudo-subsections
        if re.match(r"^Page\s+\d+(?:\s*of\s+\d+)?$", subsection, flags=re.I):
            subsection = ""
        raw_text = fields.get("text") or ""
        paras = split_paragraphs_clean(raw_text)  # <— preserves PDF paragraph breaks
        sidx = int((fields.get("section_index") or "0") or 0)
        ssidx = int((fields.get("subsection_index") or "0") or 0)
        for k, para in enumerate(paras):
            out.append((section, subsection, para, sidx, ssidx + k))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xml", help="XML file or dir or glob")
    ap.add_argument("--section", default="", help="Regex filter on section name, e.g. '^Introduction'")
    ap.add_argument("--limit", type=int, default=0, help="Max paragraphs to print")
    ap.add_argument("--full", action="store_true", help="Print only paragraph text")
    args = ap.parse_args()

    src = pick_xml(args.xml)
    root = ET.parse(src).getroot()

    is_jats = any(ln(t.tag) in {"article", "front", "body", "sec"} for t in root.iter())
    rows = load_jats_records(root) if is_jats else load_nihms_records(root)

    if args.section:
        rx = re.compile(args.section)
        rows = [r for r in rows if rx.search(r[0] or "")]

    # sort by section/subsection order if available; JATS abstract uses -1
    rows.sort(key=lambda r: (r[3], r[4]))

    if args.limit > 0:
        rows = rows[:args.limit]

    if not rows:
        print("No paragraphs found."); return

    if args.full:
        for _, _, txt, _, _ in rows:
            if txt: print(txt)
        return

    # Probe-style with headers
    for i, (sec, sub, txt, _, _) in enumerate(rows, 1):
        path = f"{sec or 'section'}/{sub or 'p'}"
        print(f"[{i:02}] body_paragraph :: {path}")
        print(txt)
        print()

if __name__ == "__main__":
    main()
