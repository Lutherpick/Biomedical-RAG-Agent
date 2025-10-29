#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys, re
from pathlib import Path
from typing import List, Dict, Iterable, Tuple

# -------- sentence splitter with robust fallbacks --------
def _get_sentence_splitter():
    # 1) Fast path: blingfire
    try:
        import blingfire as bf  # type: ignore
        return lambda s: [t for t in bf.text_to_sentences(s).split("\n") if t.strip()]
    except Exception:
        pass
    # 2) NLTK (handle punkt + punkt_tab on newer versions)
    try:
        import nltk  # type: ignore
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
        # Newer NLTK separates language tables:
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            try:
                nltk.download("punkt_tab", quiet=True)
            except Exception:
                # ok to continue; sent_tokenize works without tables for en
                pass
        from nltk.tokenize import sent_tokenize  # type: ignore
        return lambda s: [t for t in sent_tokenize(s) if t.strip()]
    except Exception:
        pass
    # 3) Regex fallback with a few biomedical abbreviations guarded
    ABBR = r"(?:e\.g|i\.e|et al|Fig|Figs|Dr|Mr|Ms|Prof|Ref|ca|vs|No|Fig\.|Eq|Eq\.|cf)"
    rx = re.compile(rf"(?<!{ABBR})\.(?=\s+[A-Z])|(?<=[!?])\s+")
    def _split(s: str) -> List[str]:
        parts, start = [], 0
        for m in rx.finditer(s):
            end = m.end()
            seg = s[start:end].strip()
            if seg:
                parts.append(seg)
            start = end
        tail = s[start:].strip()
        if tail:
            parts.append(tail)
        return parts
    return _split

_SENT_SPLIT = _get_sentence_splitter()

# -------- tokenizer with fallback --------
def _get_tokenizer():
    try:
        import tiktoken  # type: ignore
        enc = tiktoken.get_encoding("cl100k_base")
        return enc.encode, enc.decode
    except Exception:
        return (lambda s: s.split()), (lambda toks: " ".join(toks))

_ENCODE, _DECODE = _get_tokenizer()

def _count_tokens(text: str) -> int:
    return len(_ENCODE(text))

# -------- section grouping helpers --------
def _section_root_from_path(element_path: str, section_path: str) -> str:
    return section_path or "/"

def _key(rec: Dict) -> Tuple[str, str]:
    return (rec["pmcid"], _section_root_from_path(rec.get("element_path",""), rec.get("section_path","/")))

# -------- chunking logic --------
def chunk_sentences_to_token_windows(sents: List[str], max_tokens: int, min_tokens: int, overlap_ratio: float) -> List[str]:
    chunks: List[str] = []
    window: List[str] = []
    window_tokens = 0
    ov = max(0, int(max_tokens * overlap_ratio))

    def flush():
        nonlocal window, window_tokens
        if not window:
            return
        text = " ".join(window).strip()
        if text and (_count_tokens(text) >= min_tokens or not chunks):
            chunks.append(text)
        window = []
        window_tokens = 0

    for s in sents:
        t = _count_tokens(s)
        if t > max_tokens:
            flush()
            toks = _ENCODE(s)
            i = 0
            while i < len(toks):
                j = min(i + max_tokens, len(toks))
                piece = _DECODE(toks[i:j])
                if _count_tokens(piece) >= min_tokens or not chunks:
                    chunks.append(piece)
                i = j - ov if j - ov > i else j
            continue

        if window_tokens + t <= max_tokens:
            window.append(s); window_tokens += t
        else:
            flush()
            window = [s]; window_tokens = t

    flush()

    if ov > 0 and len(chunks) > 1:
        out: List[str] = []
        prev_toks = None
        for idx, ch in enumerate(chunks):
            if idx == 0 or prev_toks is None:
                out.append(ch)
            else:
                tail = prev_toks[-ov:] if len(prev_toks) > ov else prev_toks
                merged = (_DECODE(tail) + " " + ch).strip()
                out.append(merged)
            prev_toks = _ENCODE(ch)
        chunks = out
    return chunks

# -------- io --------
def read_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue

# -------- cli --------
def main():
    ap = argparse.ArgumentParser(description="Chunk parsed_nxml.jsonl into token-bounded windows per section.")
    ap.add_argument("--parsed", required=True, help="Path to parsed_nxml.jsonl from parse_nxml.py")
    ap.add_argument("--out", default="pmc_chunker/out/chunks.jsonl")
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--min-tokens", type=int, default=100)
    ap.add_argument("--overlap", type=float, default=0.10, help="Proportional overlap fraction, e.g., 0.1 for 10%")
    ap.add_argument("--include-types", default="paragraph,fig_caption,table_caption,article_meta",
                    help="Comma list of element types to include before grouping.")
    args = ap.parse_args()

    include = set(t.strip() for t in args.include_types.split(",") if t.strip())
    parsed = Path(args.parsed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    buckets: Dict[Tuple[str,str], List[Dict]] = {}
    for rec in read_jsonl(parsed):
        if rec.get("type") not in include:
            continue
        k = _key(rec)
        buckets.setdefault(k, []).append(rec)

    for k in buckets:
        buckets[k].sort(key=lambda r: r.get("element_path",""))

    n_chunks = 0
    with out.open("w", encoding="utf-8") as fh:
        for (pmcid, section_path), items in buckets.items():
            sents: List[str] = []
            for r in items:
                sents.extend(_SENT_SPLIT(r["text"]))
            if not sents:
                continue

            chunks = chunk_sentences_to_token_windows(
                sents, max_tokens=args.max_tokens, min_tokens=args.min_tokens, overlap_ratio=args.overlap
            )

            for ci, ch in enumerate(chunks):
                rec_out = {
                    "PMCID": pmcid,
                    "section_path": section_path,
                    "chunk_id": ci,
                    "text": ch,
                    "token_count": _count_tokens(ch),
                }
                fh.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
                n_chunks += 1

    print(f"[done] {n_chunks} chunks -> {out}")

if __name__ == "__main__":
    main()
