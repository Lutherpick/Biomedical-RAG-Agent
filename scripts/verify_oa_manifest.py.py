# data_pipeline/verify_oa_manifest_fast.py
from pathlib import Path
import csv, json, os, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data_pipeline" / "data"
OUT_BASE = DATA_ROOT / "pubmed_open_access"
MANIFEST = DATA_ROOT / "manifests" / "oa_manifest.csv"

def stat_nonempty(p: Path) -> bool:
    try:
        return p.exists() and os.stat(p).st_size > 0
    except Exception:
        return False

def check_row(r, mode: str, sample_fraction: float):
    pmid = r["pmid"]
    meta_path = DATA_ROOT / r["metadata_file"]
    abs_path  = DATA_ROOT / r["abstract_file"]

    issues = []
    if not stat_nonempty(meta_path):
        issues.append(("missing_metadata", pmid))
    if not stat_nonempty(abs_path):
        issues.append(("missing_abstract", pmid))

    # Optional deeper checks
    if mode == "full":
        # parse JSON for all; otherwise only sample a fraction
        parse_json = True
    else:
        # fast mode: sample a fraction (e.g., 0.1 == 10%)
        parse_json = (hash(pmid) % 10_000) / 10_000.0 < sample_fraction

    if parse_json and meta_path.exists():
        try:
            with meta_path.open("r", encoding="utf-8") as f:
                json.load(f)
        except Exception:
            issues.append(("corrupted_metadata", pmid))

    return issues

def main(mode: str, workers: int, sample_fraction: float):
    if not MANIFEST.exists():
        print(f"❌ Manifest not found: {MANIFEST}")
        return

    with MANIFEST.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"🔍 Checking {len(rows):,} manifest rows in {mode} mode "
          f"(workers={workers}, sample={sample_fraction})")

    issues = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(check_row, r, mode, sample_fraction) for r in rows]
        for fut in as_completed(futs):
            issues.extend(fut.result())

    total_folders = sum(1 for _ in OUT_BASE.glob("PMID_*"))

    print(f"✅ Manifest rows: {len(rows):,}")
    print(f"📦 PMID folders: {total_folders:,}")

    if not issues:
        print("🎉 All good!")
    else:
        miss_meta = [p for t,p in issues if t == "missing_metadata"]
        miss_abs  = [p for t,p in issues if t == "missing_abstract"]
        bad_meta  = [p for t,p in issues if t == "corrupted_metadata"]

        if miss_meta: print(f"  - Missing metadata.json: {len(miss_meta)} (e.g., {miss_meta[:5]})")
        if miss_abs:  print(f"  - Missing/empty abstract.txt: {len(miss_abs)} (e.g., {miss_abs[:5]})")
        if bad_meta:  print(f"  - Corrupted metadata.json: {len(bad_meta)} (e.g., {bad_meta[:5]})")

        report = DATA_ROOT / "manifests" / "oa_validation_report.csv"
        with report.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["pmid","issue_type"])
            for t,p in issues: w.writerow([p,t])
        print(f"🧾 Report → {report}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fast","full"], default="fast")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 8)
    ap.add_argument("--sample", type=float, default=0.0,
                    help="In fast mode, fraction of rows to JSON-parse (0.0–1.0).")
    args = ap.parse_args()
    main(args.mode, args.workers, max(0.0, min(1.0, args.sample)))
