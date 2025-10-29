from __future__ import annotations
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import re
import argparse
import json

import pandas as pd
from lxml import etree
from huggingface_hub import snapshot_download
from langchain_experimental import text_splitter
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


# --------------------
# Token utilities
# --------------------
def _get_tokenizer():
    """
    Prefer tiktoken cl100k_base. Fallback to whitespace split.
    Returns a callable that maps text -> list[int|str].
    """
    try:
        import tiktoken  # type: ignore
        enc = tiktoken.get_encoding("cl100k_base")
        return lambda s: enc.encode(s)
    except Exception:
        return lambda s: s.split()


_TOKENIZE = _get_tokenizer()


def count_tokens(text: str) -> int:
    return len(_TOKENIZE(text))


def split_by_tokens(
        text: str,
        max_tokens: int,
        min_tokens: int = 100,
        overlap: int = 50,
) -> List[str]:
    """
    Split text into slices by token count. Never cross section boundaries if you
    call it per-section. Keeps overlap tokens between slices. Drops sub-100 token
    fragments unless the whole text is < min_tokens.
    """
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
        import tiktoken  # type: ignore
        enc = tiktoken.get_encoding("cl100k_base")
        return [enc.decode(p) for p in out]
    except Exception:
        return [" ".join(p) for p in out]


# --------------------
# Furniture stripping
# --------------------
FURNITURE_INLINE = (
    r"(?:^\s*\d+\s*$)"                # bare page numbers
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


# --------------------
# Fixed-size chunker
# --------------------
def getFixedChunker(chunk_size: int, chunkCountSymbol: str = " ") -> CharacterTextSplitter:
    """
    Character-based splitter with small overlap. Use only for quick tests.
    """
    return CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size * 0.08),
        separator=chunkCountSymbol,
        strip_whitespace=False,
    )


# --------------------
# Semantic chunker model (LangChain SemanticChunker)
# --------------------
def loadModel(
        modelName: str,
        modelPath: str,
        minChunkSize: int = 400,
) -> text_splitter.SemanticChunker:
    """
    Returns a LangChain SemanticChunker configured with HF embeddings.
    """
    model_kwargs = {"device": "cpu", "trust_remote_code": True}
    encode_kwargs = {"normalize_embeddings": False}
    hf = HuggingFaceEmbeddings(
        model_name=(modelPath + modelName),
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    return text_splitter.SemanticChunker(
        embeddings=hf,
        buffer_size=1,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=93,
        min_chunk_size=minChunkSize,
    )


def downloadModel(modelName: str, folderPath: str = "./models/") -> None:
    target = Path(folderPath) / modelName
    if not target.is_dir():
        snapshot_download(repo_id=modelName, local_dir=str(target))


def getModel(modelName: str, minChunkSize: int = 400) -> text_splitter.SemanticChunker:
    downloadModel(modelName)
    return loadModel(modelName=modelName, modelPath="./models/", minChunkSize=minChunkSize)


# --------------------
# Embeddings handle
# --------------------
def getEmbeddings(modelPath: str, modelName: str) -> HuggingFaceEmbeddings:
    model_kwargs = {"device": "cpu", "trust_remote_code": True}
    encode_kwargs = {"normalize_embeddings": False}
    return HuggingFaceEmbeddings(
        model_name=(modelPath + modelName),
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )


# --------------------
# High-level helpers
# --------------------
def splitText(splitterModel, text: str) -> List[str]:
    cleaned = strip_furniture(text)
    return splitterModel.split_text(cleaned)


# ====================
# JATS XML parsing
# ====================
NS = {"x": "http://www.ncbi.nlm.nih.gov/JATS1"}

def _xpath(root: etree._Element, path: str) -> List[etree._Element]:
    # Namespace-agnostic using local-name()
    return root.xpath(path)

def _text(el: Optional[etree._Element]) -> str:
    if el is None:
        return ""
    return " ".join("".join(el.itertext()).split())

def extract_sections(xml_path: Path) -> List[Tuple[str, str]]:
    """
    Return list of (section_type, text) respecting section boundaries.
    Includes: Abstract, Introduction, Methods, Results, Discussion, Conclusions.
    Falls back to all top-level <sec>.
    """
    with xml_path.open("rb") as f:
        tree = etree.parse(f)
    root = tree.getroot()

    out: List[Tuple[str, str]] = []

    # Abstracts
    for abs_el in _xpath(root, ".//*[local-name()='abstract']"):
        txt = _text(abs_el)
        if txt:
            out.append(("Abstract", txt))

    # Top-level sections
    for sec in _xpath(root, ".//*[local-name()='body']/*[local-name()='sec' and not(ancestor::*[local-name()='sec'])]"):
        title_nodes = sec.xpath(".//*[local-name()='title']")
        title = (_text(title_nodes[0]) if title_nodes else "").strip()
        sec_txt = _text(sec)
        if not sec_txt:
            continue

        # Normalize common names
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

        out.append((sec_name, sec_txt))

    return out


# ====================
# Pipeline
# ====================
def build_chunks_for_article(
        pmcid: str,
        xml_dir: Path,
        version: str,
        max_tokens: int,
        min_tokens: int,
        overlap: int,
) -> List[Dict]:
    xml_path = xml_dir / f"{pmcid}.xml"
    if not xml_path.is_file():
        return []

    sections = extract_sections(xml_path)
    rows: List[Dict] = []
    chunk_idx = 0

    for sec_name, sec_text in sections:
        sec_text = strip_furniture(sec_text)
        if not sec_text:
            continue

        if version == "v1":
            parts = [sec_text]
        else:
            parts = split_by_tokens(sec_text, max_tokens=max_tokens, min_tokens=min_tokens, overlap=overlap)

        for p in parts:
            if not p:
                continue
            rows.append(
                {
                    "pmcid": pmcid,
                    "section_type": sec_name,
                    "section_title": sec_name,
                    "chunk_index": chunk_idx,
                    "chunk_text": p,
                    "token_count": count_tokens(p),
                    "version": version,
                }
            )
            chunk_idx += 1

    return rows


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

    df_manifest = pd.read_csv(manifest_path)
    # Accept either 'PMCID' or 'pmcid'
    pmc_col = "PMCID" if "PMCID" in df_manifest.columns else ("pmcid" if "pmcid" in df_manifest.columns else None)
    if pmc_col is None:
        raise ValueError("Manifest must contain a PMCID/pmcid column")

    rows: List[Dict] = []
    for pmcid in df_manifest[pmc_col].astype(str):
        rows.extend(
            build_chunks_for_article(
                pmcid=pmcid,
                xml_dir=xml_dir,
                version=version,
                max_tokens=max_tokens,
                min_tokens=min_tokens,
                overlap=overlap,
            )
        )

    if not rows:
        # Still write an empty file to make failures obvious
        (out_dir / "chunks.json").write_text("", encoding="utf-8")
        print("[write] 0 chunks ->", out_dir / "chunks.json")
        return

    df = pd.DataFrame(rows)
    df.to_json(out_dir / "chunks.json", orient="records", lines=True, force_ascii=False)
    print(f"[write] {len(df)} chunks -> {(out_dir / 'chunks.json')}")


# ====================
# CLI
# ====================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--version", choices=["v1", "v2", "v3"], required=True)
    p.add_argument("--manifest", required=True, help="Path to manifest CSV")
    p.add_argument("--xml-dir", required=True, help="Directory with PMC XMLs")
    p.add_argument("--max-tokens", type=int, default=800)
    p.add_argument("--min-tokens", type=int, default=100)
    p.add_argument("--overlap", type=int, default=50)
    p.add_argument("--out-dir", default="pmc_chunker/out")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ver = args.version
    if ver == "v1":
        # one chunk per section
        max_t, min_t, ov = 10_000, 0, 0
    elif ver == "v2":
        max_t, min_t, ov = args.max_tokens, args.min_tokens, args.overlap  # 800 default
    else:
        # v3
        max_t, min_t, ov = args.max_tokens, args.min_tokens, args.overlap  # 400 via CLI

    run(
        manifest_path=Path(args.manifest),
        xml_dir=Path(args.xml-dir) if hasattr(args, "xml-dir") else Path(args.xml_dir),  # guard against hyphen edge
        version=ver,
        max_tokens=max_t,
        min_tokens=min_t,
        overlap=ov,
        out_dir=Path(args.out_dir),
    )
