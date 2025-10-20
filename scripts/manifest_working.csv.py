# scripts/fix_manifest_from_local_pmcxml.py
# Rebuild PMCID mapping from local PMC XMLs and patch the manifest.
# Outputs:
#   data_pipeline/data/manifests/oa_manifest_fixed.csv
#   data_pipeline/data/manifests/manifest_working.csv  (only rows with valid PMCID + local fulltext xml)

import os, re, csv, sys
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]  # repo_root/data_pipeline/...
DATA_ROOT = ROOT / "data_pipeline" / "data"
XML_DIR = DATA_ROOT / "raw" / "pmc_xml"
MANIFEST = DATA_ROOT / "manifests" / "oa_manifest.csv"
OUT_FIXED = DATA_ROOT / "manifests" / "oa_manifest_fixed.csv"
OUT_WORK  = DATA_ROOT / "manifests" / "manifest_working.csv"

def find_pmid_pmcid(xml_path: Path):
    """Parse minimal <article-id> tags to get pmid and pmcid."""
    try:
        tree = ET.parse(str(xml_path))
        root = tree.getroot()
    except Exception:
        return None, None
    pmid = None
    pmcid = None
    for el in root.findall(".//article-id"):
        t = el.attrib.get("pub-id-type", "").lower()
        txt = (el.text or "").strip()
        if t == "pmid" and txt.isdigit():
            pmid = txt
        elif t in ("pmcid", "pmc"):
            # normalize forms like "PMC123456"
            m = re.search(r"(PMC\d+)", txt.upper())
            if m:
                pmcid = m.group(1)
    return pmid, pmcid

def scan_local_pmcxml():
    """Return dicts from local PMC xmls:
       - pmid -> pmcid
       - pmcid -> xml_relpath (relative to repo root)
    """
    pmid_to_pmcid = {}
    pmcid_to_xml = {}
    if not XML_DIR.exists():
        return pmid_to_pmcid, pmcid_to_xml

    for p in XML_DIR.rglob("*.xml"):
        pmid, pmcid = find_pmid_pmcid(p)
        if pmcid:
            rel = p.relative_to(ROOT)
            pmcid_to_xml[pmcid] = str(rel)
        if pmid and pmcid:
            pmid_to_pmcid[pmid] = pmcid
    return pmid_to_pmcid, pmcid_to_xml

def read_manifest(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        rows = [dict(row) for row in r]
        fieldnames = r.fieldnames or []
    return rows, fieldnames

def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)

def main():
    if not MANIFEST.exists():
        print(f"[ERR] Manifest not found: {MANIFEST}")
        sys.exit(1)

    print("[scan] building local pmid/pmcid map from PMC XML …")
    pmid_to_pmcid, pmcid_to_xml = scan_local_pmcxml()
    print(f"[scan] pmid→pmcid: {len(pmid_to_pmcid)} | pmcid→xml: {len(pmcid_to_xml)}")

    rows, fields = read_manifest(MANIFEST)

    # Ensure required columns exist
    if "pmid" not in fields:
        # Try common variants
        for k in ("PMID", "id", "pubmed_id"):
            if k in fields:
                for row in rows:
                    row["pmid"] = row.get(k, "").strip()
                fields.append("pmid")
                break
    if "pmcid" not in fields:
        fields.append("pmcid")
        for row in rows:
            row["pmcid"] = ""

    # Also add a local xml path column (if not present)
    xml_col = "pmc_fulltext_xml"
    if xml_col not in fields:
        fields.append(xml_col)
        for row in rows:
            row[xml_col] = ""

    # Patch pmcid from local map where missing/empty
    patched = 0
    with_xml = 0
    for row in rows:
        pmid = (row.get("pmid") or "").strip()
        pmcid = (row.get("pmcid") or "").strip().upper()
        if not pmcid and pmid in pmid_to_pmcid:
            row["pmcid"] = pmid_to_pmcid[pmid]
            patched += 1
            pmcid = row["pmcid"]  # update local

        if pmcid in pmcid_to_xml:
            row[xml_col] = pmcid_to_xml[pmcid]
            with_xml += 1

    print(f"[patch] pmcid filled from local XMLs: {patched}")
    print(f"[patch] rows with local xml path: {with_xml}")

    # Write fixed manifest
    write_csv(OUT_FIXED, rows, fields)
    print(f"[out] {OUT_FIXED}")

    # Working subset: has PMCID and existing xml file on disk
    work = []
    for row in rows:
        pmcid = (row.get("pmcid") or "").strip().upper()
        xrel = row.get(xml_col, "").strip()
        if not (pmcid and re.match(r"^PMC\d+$", pmcid) and xrel):
            continue
        xabs = ROOT / xrel
        if xabs.exists():
            work.append(row)

    write_csv(OUT_WORK, work, fields)
    print(f"[out] {OUT_WORK} (rows={len(work)})")

if __name__ == "__main__":
    main()
