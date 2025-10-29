#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
from lxml import etree
import re
import json

# =========================
# Tokenization utilities
# =========================
def _get_tokenizer():
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return lambda s: enc.encode(s)
    except Exception:
        return lambda s: s.split()

_TOKENIZE = _get_tokenizer()

def count_tokens(text: str) -> int:
    return len(_TOKENIZE(text))

# =========================
# Furniture stripping
# =========================
FURNITURE_INLINE = (
    r"(?:^\s*\d+\s*$)"
    r"|(?:^Author Manuscript.*$)"
    r"|(?:^bioRxiv.*$)"
    r"|(?:^\s*Page\s+\d+.*$)"
    r"|(?:^\s*Figure\s+\d+.*$)"
    r"|(?:^\s*Table\s+\d+.*$)"
)

def strip_furniture(text: str) -> str:
    lines = [
        ln for ln in text.splitlines()
        if not re.match(FURNITURE_INLINE, ln.strip(), flags=re.IGNORECASE)
    ]
    out = "\n".join(lines)
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return out

# =========================
# Token chunking
# =========================
def split_by_tokens(text: str, max_tokens: int, min_tokens: int = 100, overlap: int = 50) -> List[str]:
    toks = _TOKENIZE(text)
    n = len(toks)
    if n == 0:
        return []
    if min_tokens <= n <= max_tokens:
        return [text]

    out = []
    i = 0
    while i < n:
        j = min(i + max_tokens, n)
        piece = toks[i:j]
        if len(piece) >= min_tokens or (i == 0 and j == n):
            out.append(piece)
        nxt = j - overlap
        i = nxt if nxt > i else j

    # detokenize
    if isinstance(toks[0] if toks else "", str):
        return [" ".join(p) for p in out]

    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return [enc.decode(p) for p in out]
    except Exception:
        return [" ".join(p) for p in out]

# =========================
# XML helpers
# =========================
def _text(el) -> str:
    if el is None:
        return ""
    return " ".join("".join(el.itertext()).split())

def extract_sections(xml_path: Path) -> List[Tuple[str, str, str]]:
    """
    Return list of (section_type, text, section_path).
    Abstracts whole; top-level body <sec> only.
    """
    with xml_path.open("rb") as f:
        root = etree.parse(f).getroot()

    out: List[Tuple[str, str, str]] = []

    # Abstracts
    for i, abs_el in enumerate(root.xpath(".//*[local-name()='abstract']")):
        txt = _text(abs_el)
        if txt:
            out.append(("Abstract", txt, f"/abstract[{i+1}]"))

    # Body sections
    secs = root.xpath(".//*[local-name()='body']/*[local-name()='sec' and not(ancestor::*[local-name()='sec'])]")
    for s_idx, sec in enumerate(secs, start=1):
        title_nodes = sec.xpath("./*[local-name()='title']")
        title = (_text(title_nodes[0]) if title_nodes else "").strip()
        sec_txt = _text(sec)
        if not sec_txt:
            continue

        name = title.lower()
        if "method" in name or "materials" in name:
            sec_name = "Materials and methods"
        elif "result" in name:
            sec_name = "Results"
        elif "discussion" in name:
            sec_name = "Discussion"
        elif "conclusion" in name:
            sec_name = "Conclusions"
        elif "introduction" in name or "background" in name:
            sec_name = "Introduction"
        else:
            sec_name = title or "Section"

        out.append((sec_name, sec_txt, f"/body[1]/sec[{s_idx}]"))

    return out

# =========================
# Chunk builder
# =========================
def build_chunks_for_article(
        pmcid: str,
        xml_dir: Path,
        version: str,
        max_tokens: int,
        min_tokens: int,
        overlap: int,
        meta: Dict,
) -> List[Dict]:
    xml_path = xml_dir / f"{pmcid}.xml"
    if not xml_path.is_file():
        return []

    sections = extract_sections(xml_path)
    rows: List[Dict] = []
    chunk_idx = 0

    for sec_name, sec_text, sec_path in sections:
        sec_text = strip_furniture(sec_text)
        if not sec_text:
            continue

        # Abstracts unchunked
        if sec_name == "Abstract":
            parts = [sec_text]
        elif version == "v1":
            parts = [sec_text]
        else:
            parts = split_by_tokens(sec_text, max_tokens=max_tokens, min_tokens=min_tokens, overlap=overlap)

        for p in parts:
            if not p:
                continue
            rows.append({
                "pmcid": pmcid,
                "pmid": meta.get("pmid"),
                "doi": meta.get("doi"),
                "year": meta.get("year"),
                "journal": meta.get("journal"),
                "article_title": meta.get("title"),
                "topic": meta.get("topic"),
                "license": meta.get("license"),
                "version": version,
                "section_type": sec_name,
                "section_title": sec_name,
                "section_path": sec_path,
                "chunk_index": chunk_idx,
                "chunk_text": p,
                "token_count": count_tokens(p),
                "tokenizer_name": "tiktoken:cl100k_base",
                "source": "pmc_xml",
                "xml_path": str(xml_path),
            })
            chunk_idx += 1

    return rows

# =========================
# Pipeline runner
# =========================
def run(
        manifest_path: Path,
        xml_dir: Path,
        version: str,
        max_tokens: int,
        min_tokens: int,
        overlap: int,
        out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(manifest_path)

    cols = {c.lower(): c for c in df.columns}
    pmc_col = cols.get("pmcid") or cols.get("pmc id") or "PMCID"
    if pmc_col not in df.columns:
        raise ValueError("Manifest must contain PMCID/pmcid")

    def meta_for(row: pd.Series) -> Dict:
        return {
            "pmid": row.get(cols.get("pmid")),
            "doi": row.get(cols.get("doi")),
            "year": row.get(cols.get("year")),
            "journal": row.get(cols.get("journal")),
            "title": row.get(cols.get("title")),
            "topic": row.get(cols.get("topic")),
            "license": row.get(cols.get("license")),
        }

    rows: List[Dict] = []
    for _, r in df.iterrows():
        pmcid = str(r[pmc_col])
        rows.extend(
            build_chunks_for_article(
                pmcid=pmcid,
                xml_dir=xml_dir,
                version=version,
                max_tokens=max_tokens,
                min_tokens=min_tokens,
                overlap=overlap,
                meta=meta_for(r),
            )
        )

    out_path = out_dir / f"chunks_{version}.jsonl"
    pd.DataFrame(rows).to_json(out_path, orient="records", lines=True, force_ascii=False)
    print(f"[write] {len(rows)} chunks -> {out_path}")

# =========================
# CLI
# =========================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--version", choices=["v1", "v2", "v3"], required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--xml-dir", required=True)
    p.add_argument("--max-tokens", type=int, default=800)
    p.add_argument("--min-tokens", type=int, default=100)
    p.add_argument("--overlap", type=int, default=50)
    p.add_argument("--out-dir", default="pmc_chunker/out")
    return p.parse_args()

# =========================
# Entry point
# =========================
if __name__ == "__main__":
    args = parse_args()
    if args.version == "v1":
        max_t, min_t, ov = 10_000, 0, 0
    elif args.version == "v2":
        max_t, min_t, ov = args.max_tokens, args.min_tokens, args.overlap
    else:
        max_t, min_t, ov = min(args.max_tokens, 400), args.min_tokens, args.overlap

    run(
        manifest_path=Path(args.manifest),
        xml_dir=Path(args.xml_dir),
        version=args.version,
        max_tokens=max_t,
        min_tokens=min_t,
        overlap=ov,
        out_dir=Path(args.out_dir),
    )
