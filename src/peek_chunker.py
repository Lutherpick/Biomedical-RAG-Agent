#!/usr/bin/env python3
import argparse, json, random, sys
from pathlib import Path

def load_jsonl(p: Path):
    with p.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def show(rec, width=800):
    print("="*90)
    print(f"PMCID: {rec.get('pmcid')}")
    print(f"Section: {rec.get('section_type')} | Title: {rec.get('section_title','')}")
    print(f"Chunk index: {rec.get('chunk_index')} | Tokens: {rec.get('token_count')}")
    print("-"*90)
    txt = rec.get("chunk_text","").strip()
    print(txt[:width])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="path to chunks_v*.jsonl")
    ap.add_argument("--n", type=int, default=5, help="number of samples")
    ap.add_argument("--pmcid", default=None)
    ap.add_argument("--section", default=None)
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--width", type=int, default=800)
    args = ap.parse_args()

    p = Path(args.file)
    rows = list(load_jsonl(p))

    if args.pmcid:
        rows = [r for r in rows if r.get("pmcid")==args.pmcid]
    if args.section:
        rows = [r for r in rows if str(r.get("section_type","")).lower()==args.section.lower()]
    if args.index is not None:
        rows = [r for r in rows if r.get("chunk_index")==args.index]

    if not rows:
        print("no rows matched")
        sys.exit(0)

    sample = rows if len(rows) <= args.n else random.sample(rows, args.n)
    for r in sample:
        show(r, width=args.width)

if __name__ == "__main__":
    main()
