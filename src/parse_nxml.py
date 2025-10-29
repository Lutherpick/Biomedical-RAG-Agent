#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import Optional, Iterable, Tuple, List, Dict
from lxml import etree

# ---------- text utils ----------
def _norm_text(s: str) -> str:
    return " ".join(s.split())

def _all_text(el: Optional[etree._Element]) -> str:
    if el is None:
        return ""
    return _norm_text("".join(el.itertext()))

# ---------- path helpers (namespace-agnostic, robust) ----------
def _lname(el: etree._Element) -> str:
    """
    Return a safe local-name for any node. Handles elements, comments, PIs.
    """
    tag = el.tag
    # Normal element tag -> string or '{ns}local'
    if isinstance(tag, str):
        if "}" in tag:
            return tag.split("}", 1)[1]
        return tag
    # lxml uses cython singletons for comments/PIs; provide readable labels
    try:
        name = getattr(tag, "__name__", None)
        if name:
            return name.lower()
    except Exception:
        pass
    return str(tag).lower()

def _sib_index(el: etree._Element) -> int:
    """1-based index among siblings with same local-name()."""
    parent = el.getparent()
    if parent is None:
        return 1
    name = _lname(el)
    idx = 0
    for sib in parent.iterchildren():  # elements only
        if _lname(sib) == name:
            idx += 1
        if sib is el:
            return idx
    return max(1, idx)

def _path_to(el: Optional[etree._Element]) -> str:
    """Build a JATS-like, namespace-free path, e.g., /body/sec[1]/sec[2]/p[3]."""
    if el is None:
        return "/"
    parts: List[str] = []
    cur = el
    # Stop at root <article>, ignore comments/PIs gracefully
    while cur is not None:
        name = _lname(cur)
        if name == "article":
            break
        parts.append(f"{name}[{_sib_index(cur)}]")
        cur = cur.getparent()
    return "/" + "/".join(reversed(parts))

def _closest_sec_path(el: etree._Element) -> str:
    cur = el
    while cur is not None and _lname(cur) != "sec":
        cur = cur.getparent()
    return _path_to(cur)

# ---------- extraction ----------
def _iter_elements(root: etree._Element) -> Iterable[Tuple[str, etree._Element]]:
    """
    Yield tuples of (type, element) for:
      - paragraphs: //sec//p
      - fig captions: //fig//caption
      - table captions: //table-wrap//caption
      - selected article-meta children
    """
    # paragraphs
    for p in root.xpath(".//*[local-name()='sec']//*[local-name()='p']"):
        yield ("paragraph", p)

    # figure captions
    for cap in root.xpath(".//*[local-name()='fig']//*[local-name()='caption']"):
        yield ("fig_caption", cap)

    # table captions
    for cap in root.xpath(".//*[local-name()='table-wrap']//*[local-name()='caption']"):
        yield ("table_caption", cap)

    # article-meta/* (subset)
    am = root.xpath(".//*[local-name()='article-meta']")
    if am:
        meta_root = am[0]
        for tag, tname in [
            (".//*[local-name()='article-title']", "article_meta"),
            (".//*[local-name()='journal-title']", "article_meta"),
            (".//*[local-name()='year']", "article_meta"),
            (".//*[local-name()='article-id' and @pub-id-type='doi']", "article_meta"),
        ]:
            for el in meta_root.xpath(tag):
                yield (tname, el)

def _pmcid_from_tree(root: etree._Element) -> Optional[str]:
    pmc = root.xpath(".//*[local-name()='article-id' and @pub-id-type='pmc']/text()")
    if pmc:
        v = pmc[0].strip()
        return v if v.startswith("PMC") else f"PMC{v}"
    return None

def parse_one(xml_path: Path) -> List[Dict]:
    with xml_path.open("rb") as f:
        tree = etree.parse(f)
    root = tree.getroot()
    pmcid = _pmcid_from_tree(root) or xml_path.stem

    rows: List[Dict] = []
    for typ, el in _iter_elements(root):
        txt = _all_text(el)
        if not txt:
            continue
        rec = {
            "pmcid": pmcid,
            "type": typ,
            "section_path": _closest_sec_path(el) if typ != "article_meta" else "/front/article-meta",
            "element_path": _path_to(el),
            "text": txt,
        }
        rows.append(rec)
    return rows

# ---------- io ----------
def _read_pmcids_from_manifest(manifest: Path) -> List[str]:
    import pandas as pd
    df = pd.read_csv(manifest)
    col = "PMCID" if "PMCID" in df.columns else ("pmcid" if "pmcid" in df.columns else None)
    if not col:
        return []
    return [str(x) for x in df[col].tolist()]

# ---------- cli ----------
def main():
    ap = argparse.ArgumentParser(description="Parse JATS .xml to JSONL with hierarchy.")
    ap.add_argument("--xml-dir", required=True, help="Directory containing PMC JATS XMLs")
    ap.add_argument("--manifest", help="Optional CSV with PMCID filter column {PMCID|pmcid}")
    ap.add_argument("--out", default="pmc_chunker/out/parsed_nxml.jsonl")
    args = ap.parse_args()

    xml_dir = Path(args.xml_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    allow: Optional[set[str]] = None
    if args.manifest:
        allow = set(_read_pmcids_from_manifest(Path(args.manifest)))

    n_written = 0
    with out.open("w", encoding="utf-8") as fh:
        for p in sorted(xml_dir.glob("*.xml")):
            if allow and p.stem not in allow and f"PMC{p.stem}" not in allow:
                continue
            try:
                rows = parse_one(p)
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                    n_written += 1
            except Exception as e:
                print(f"[warn] parse failed: {p.name}: {e}", file=sys.stderr)

    print(f"[done] {n_written} elements -> {out}")

if __name__ == "__main__":
    main()
