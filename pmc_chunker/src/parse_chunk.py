#!/usr/bin/env python3
from __future__ import annotations
import re, json, unicodedata, argparse
from pathlib import Path
from typing import Iterable, Dict, Any, List, Set, Tuple
import pandas as pd
from lxml import etree

# --- Defaults (4k scope; can be overridden by CLI) ---
HERE = Path(__file__).resolve().parent                     # pmc_chunker/src
PKG  = HERE / "pmc_chunker"
DEF_XML_DIR   = PKG / "data" / "xml"
DEF_OUT_DIR   = PKG / "out"
DEF_MANIFEST4 = DEF_OUT_DIR / "manifest_4000.csv"
OUT_JSON      = DEF_OUT_DIR / "chunks.jsonl"               # embed imports this

# --- Spec ---
TARGET_TOK   = 400
CAPTION_MAX  = 300
OVERLAP      = 0                                           # section-pure
SEC_MAP = {
    "introduction": "introduction",
    "materials and methods": "methods",
    "methods": "methods",
    "results": "results",
    "discussion": "discussion",
}

PMC_RE = re.compile(r"PMC(\d+)")
WS_RE  = re.compile(r"\s+")

def _norm(t: str) -> str:
    t = unicodedata.normalize("NFC", t or "")
    return WS_RE.sub(" ", t).strip()

def _tok(t: str) -> List[str]: return t.split()
def _chunks(tokens: List[str], max_len: int) -> Iterable[List[str]]:
    step = max_len - OVERLAP
    for i in range(0, len(tokens), step):
        yield tokens[i:i+max_len]

def _sec_type(title: str) -> str:
    tl = (title or "").lower()
    for k, v in SEC_MAP.items():
        if k in tl: return v
    return "other"

def _text(e: etree._Element) -> str: return " ".join(e.itertext())

def _ids(root: etree._Element) -> Tuple[str|None, str|None]:
    pmid = doi = None
    for a in root.xpath(".//article-id"):
        t = (a.get("pub-id-type") or a.get("pubid-type") or "").lower()
        if t == "pmid": pmid = (a.text or "").strip()
        if t == "doi":  doi  = (a.text or "").strip()
    return pmid, doi

def _journal_year(root: etree._Element) -> Tuple[str|None, int|None]:
    j = root.findtext(".//journal-title")
    y = root.findtext(".//pub-date/year")
    return (j.strip() if j else None, int(y) if y and y.isdigit() else None)

def _pmcid_from_filename(p: Path) -> str|None:
    m = PMC_RE.search(p.stem)
    return f"PMC{m.group(1)}" if m else None

def load_allowed_pmcids(manifest: Path) -> Set[str]:
    df = pd.read_csv(manifest)
    # accept any column whose name contains 'pmcid'
    col = next((c for c in df.columns if "pmcid" in c.lower()), None)
    assert col, "PMCID column not found in manifest_4000.csv"
    vals = (df[col].dropna().astype(str)
            .str.replace(r"^\s*", "", regex=True)
            .str.replace(r"^PMC?", "PMC", regex=True)
            .str.strip())
    allow = set(vals.tolist())
    assert 3600 <= len(allow) <= 4400, f"manifest size {len(allow)} != ~4000"
    return allow

def iter_records(xml_path: Path) -> Iterable[Dict[str, Any]]:
    root = etree.parse(str(xml_path)).getroot()

    title = _norm(root.findtext(".//article-title") or "")
    if title: yield {"section_type": "title", "text": title}

    for ab in root.findall(".//abstract"):
        t = _norm(_text(ab))
        if t: yield {"section_type": "abstract", "text": t}

    for sec in root.findall(".//body//sec"):
        head = sec.findtext("./title") or ""
        stype = _sec_type(head)
        t = _norm(_text(sec))
        if t: yield {"section_type": stype, "text": t}

    for cap in root.findall(".//fig//caption"):
        t = _norm(_text(cap))
        if t:
            toks = _tok(t)[:CAPTION_MAX]
            yield {"section_type": "figure_caption", "text": " ".join(toks)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml_dir", default=str(DEF_XML_DIR))
    ap.add_argument("--manifest", default=str(DEF_MANIFEST4))
    ap.add_argument("--out", default=str(OUT_JSON))
    args = ap.parse_args()

    xml_dir  = Path(args.xml_dir)
    manifest = Path(args.manifest)
    out_json = Path(args.out)

    assert xml_dir.exists(), f"missing {xml_dir}"
    allow = load_allowed_pmcids(manifest)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with out_json.open("w", encoding="utf-8") as w:
        for xml in sorted(xml_dir.glob("*.xml")):
            pmcid = _pmcid_from_filename(xml)
            if not pmcid or pmcid not in allow:
                continue

            root = etree.parse(str(xml)).getroot()
            pmid, doi = _ids(root)
            journal, year = _journal_year(root)

            cidx = 0
            for rec in iter_records(xml):
                toks = _tok(rec["text"])
                max_len = TARGET_TOK if rec["section_type"] != "figure_caption" else min(CAPTION_MAX, TARGET_TOK)
                for chunk in _chunks(toks, max_len):
                    obj = {
                        "text": " ".join(chunk),
                        "section_type": rec["section_type"],
                        "token_count": len(chunk),
                        "chunk_index": cidx,
                        "pmcid": pmcid, "pmid": pmid, "doi": doi,
                        "journal": journal, "year": year,
                    }
                    w.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    written += 1
                    cidx += 1

    print(f"[done] chunks={written} -> {out_json}  | allow={len(allow)} from {manifest}")

if __name__ == "__main__":
    main()
