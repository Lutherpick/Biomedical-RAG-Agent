#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, random, sys
from pathlib import Path

def load_chunks(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def resolve_path(args) -> Path:
    if args.file:
        return Path(args.file)
    base = Path(args.in_dir)
    if not args.version:
        print("Provide --file or --version {v1|v2|v3}", file=sys.stderr)
        sys.exit(2)
    return base / f"chunks_{args.version}.jsonl"

def main():
    ap = argparse.ArgumentParser(description="Peek chunked JSONL")
    ap.add_argument("--file", help="Explicit JSONL path")
    ap.add_argument("--version", choices=["v1", "v2", "v3"], help="Auto-pick chunks_{version}.jsonl")
    ap.add_argument("--in-dir", default="src/pmc_chunker/out", help="Directory containing chunks_*.jsonl")
    ap.add_argument("--pmcid", help="Filter by PMCID")
    ap.add_argument("--section", help="Filter by section_type (e.g., Results)")
    ap.add_argument("--n", type=int, default=5, help="How many to show")
    ap.add_argument("--width", type=int, default=800, help="Max characters per chunk to print")
    ap.add_argument("--stats", action="store_true", help="Only print counts per section")
    args = ap.parse_args()

    path = resolve_path(args)
    if not path.is_file():
        print(f"Not found: {path}", file=sys.stderr)
        sys.exit(1)

    chunks = list(load_chunks(path))

    if args.pmcid:
        chunks = [c for c in chunks if str(c.get("pmcid")) == args.pmcid]
    if args.section:
        chunks = [c for c in chunks if str(c.get("section_type")) == args.section]

    if not chunks:
        print("No chunks matched filters.")
        return

    if args.stats:
        from collections import Counter
        sec_counts = Counter(c.get("section_type","") for c in chunks)
        total = len(chunks)
        print(f"[stats] total={total}")
        for sec, cnt in sec_counts.most_common():
            print(f"{sec or 'Unknown'}: {cnt}")
        return

    sample = random.sample(chunks, min(args.n, len(chunks)))
    for c in sample:
        print("=" * 100)
        print(f"PMCID: {c.get('pmcid')} | Section: {c.get('section_type')} | Title: {c.get('section_title','')}")
        print(f"Index: {c.get('chunk_index')} | Tokens: {c.get('token_count')} | Version: {c.get('version')}")
        print("-" * 100)
        txt = (c.get("chunk_text") or "").strip()
        print(txt[:args.width])

if __name__ == "__main__":
    main()
