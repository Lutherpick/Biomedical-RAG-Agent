# data_pipeline/dedupe_manifest_by_pmcid.py
from __future__ import annotations
from pathlib import Path
import csv, sys

ROOT       = Path(__file__).resolve().parents[1]
DATA_ROOT  = ROOT / "data_pipeline" / "data"
MAN_DIR    = DATA_ROOT / "manifests"
SRC        = MAN_DIR / "oa_manifest.csv"
BAK        = MAN_DIR / "oa_manifest.backup.csv"
OUT        = MAN_DIR / "oa_manifest.csv"   # overwrite after backup

COLS = ["pmid","pmcid","doi","title","journal","publication_date","language",
        "url_source","publisher_url","open_access",
        "metadata_file","abstract_file","fulltext_file"]

def read_rows():
    with SRC.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def write_rows(rows):
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS); w.writeheader()
        for r in rows:
            for c in COLS: r.setdefault(c, "")
            w.writerow({c:r[c] for c in COLS})

def score(row):
    # Prefer rows that look most "complete"
    s = 0
    if row.get("fulltext_file","").endswith("fulltext.xml"): s += 4
    if row.get("metadata_file","").endswith("metadata.xml"): s += 3
    pmid = row.get("pmid","")
    if pmid and not pmid.startswith("PMCID_"): s += 2  # real PMID better than placeholder
    if row.get("title"): s += 1
    return s

def main():
    rows = read_rows()

    # backup first
    OUT.write_text(SRC.read_text(encoding="utf-8"), encoding="utf-8")
    BAK.write_text(SRC.read_text(encoding="utf-8"), encoding="utf-8")

    by_pmcid = {}
    no_pmcid = []
    for r in rows:
        pmcid = r.get("pmcid","").strip()
        if pmcid:
            by_pmcid.setdefault(pmcid, []).append(r)
        else:
            no_pmcid.append(r)

    kept, dropped = [], 0
    for pmcid, group in by_pmcid.items():
        # choose best row by score (stable: keep first if tie)
        best = max(group, key=score)
        kept.append(best)
        dropped += (len(group) - 1)

    # keep all rows without pmcid (they don't collide by definition)
    kept.extend(no_pmcid)

    write_rows(kept)

    # stats
    fulltexts = [r for r in kept if r.get("fulltext_file","").endswith("fulltext.xml")]
    uniques   = {r.get("pmcid") for r in kept if r.get("pmcid")}
    print(f"[dedupe] input rows: {len(rows)}")
    print(f"[dedupe] kept rows:  {len(kept)} (dropped {dropped} duplicate rows by PMCID)")
    print(f"[dedupe] fulltext rows: {len(fulltexts)}")
    print(f"[dedupe] unique PMCIDs: {len(uniques)}")

if __name__ == "__main__":
    main()
