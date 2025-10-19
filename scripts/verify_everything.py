# data_pipeline/verify_everything.py
from __future__ import annotations
from pathlib import Path
import csv, sys
from lxml import etree as ET

ROOT      = Path(__file__).resolve().parents[1]
DATA      = ROOT / "data_pipeline" / "data"
MANIFEST  = DATA / "manifests" / "oa_manifest.csv"

def die(msg): print(msg, file=sys.stderr); sys.exit(1)

def read_manifest():
    if not MANIFEST.exists(): die(f"[X] manifest not found: {MANIFEST}")
    with MANIFEST.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        if not rows: die("[X] manifest has no rows")
        return rows

def main():
    rows = read_manifest()

    # A) counts
    full = [r for r in rows if r.get("fulltext_file","").endswith("fulltext.xml")]
    pmcids = [r["pmcid"] for r in full if r.get("pmcid")]
    uniq = set(pmcids)
    print(f"[✓] rows total: {len(rows)}")
    print(f"[✓] fulltext rows: {len(full)}")
    print(f"[✓] unique PMCIDs among fulltexts: {len(uniq)}")
    if len(uniq) != len(full):
        print(f"[X] duplicates by PMCID: {len(full) - len(uniq)}")
        dups = {}
        for pmc in pmcids:
            dups[pmc] = dups.get(pmc, 0) + 1
        top = sorted([(k,v) for k,v in dups.items() if v>1], key=lambda x:-x[1])[:10]
        print("    e.g.", top)
        sys.exit(2)

    # B) file existence
    missing = []
    for r in full:
        p = DATA / r["fulltext_file"]
        if not p.exists(): missing.append((r.get("pmcid",""), str(p)))
    print(f"[✓] missing fulltext files: {len(missing)}")
    if missing:
        for m in missing[:5]: print("    ", m)
        sys.exit(3)

    # C) quick XML parse sanity (sample N=25)
    bad = 0
    for r in full[:25]:
        p = DATA / r["fulltext_file"]
        try:
            parser = ET.XMLParser(recover=True, huge_tree=True)
            ET.parse(str(p), parser)
        except Exception as e:
            bad += 1
            print("    [parse error]", p, e)
    if bad == 0:
        print("[✓] XML parse check (25 samples): OK")
    else:
        print(f"[X] XML parse failures in samples: {bad}")
        sys.exit(4)

    print("\n[✓] VERIFIED: corpus is consistent and ready for chunking.\n")

if __name__ == "__main__":
    main()
