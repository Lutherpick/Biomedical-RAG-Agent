#!/usr/bin/env python3
# merge_jsonl_with_manifest.py
from __future__ import annotations
import argparse, json, pandas as pd
from pathlib import Path

def main():
    ap = argparse.ArgumentParser("Join paragraph JSONL with manifest on PMID")
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    man = pd.read_csv(args.manifest, dtype=str).fillna("")
    by_pmid = {str(r.PMID): r for _, r in man.iterrows()}

    out = Path(args.out).open("w", encoding="utf-8")
    with open(args.jsonl, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            pmid = str(r.get("pmid") or "")
            m = by_pmid.get(pmid)
            if m:
                r["pmcid"]  = r.get("pmcid")  or m.PMCID
                r["doi"]    = r.get("doi")    or m.doi
                r["year"]   = r.get("year")   or m.year
                r["journal"]= r.get("journal")or m.journal
                r["topic"]  = r.get("topic")  or m.topic
                r["license"]= r.get("license")or m.license
                r["title"]  = r.get("title")  or m.title
            # derive section_path
            sec = (r.get("section") or r.get("section_type") or r.get("type") or "").strip()
            sub = (r.get("subsection") or r.get("section_title") or "").strip()
            r["section_type"]  = sec or r.get("section_type") or ""
            r["section_title"] = r.get("section_title") or sub
            r["section_path"]  = f"{sec}/{sub}" if sec and sub else (sec or sub or "")
            out.write(json.dumps(r, ensure_ascii=False) + "\n")

    out.close()
    print("[merge] wrote", args.out)

if __name__ == "__main__":
    main()
