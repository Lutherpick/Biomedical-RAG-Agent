# data_pipeline/make_manifest_4000.py
from __future__ import annotations
from pathlib import Path
import csv, re, random
from lxml import etree as ET

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data_pipeline" / "data"
MAN_DIR = DATA_ROOT / "manifests"

# Try common input manifests (auto-detect the first that exists)
CANDIDATE_INPUTS = [
    MAN_DIR / "manifest_working_merged.csv",
    MAN_DIR / "manifest_working.csv",
    MAN_DIR / "oa_manifest_fixed.csv",
    MAN_DIR / "oa_manifest.csv",
    ]

OUT_MAN = MAN_DIR / "manifest_4000.csv"

random.seed(42)

# Topic buckets per document standard (balanced 4k)
TOPIC_REGEX = {
    "Oncology":         re.compile(r"cancer|carcinoma|tumou?r|oncolog", re.I),
    "Neuroscience":     re.compile(r"neuro|brain|spinal|synap|cortex|hippocamp", re.I),
    "Infectious":       re.compile(r"virus|viral|bacteria|bacterial|fung|infect|covid|influenza", re.I),
    "Immunology":       re.compile(r"immun|cytokin|t cell|b cell|antigen|antibody|th\d", re.I),
    "CardioMetabolic":  re.compile(r"cardio|heart|myocard|metabolic|diabet|obes|lipid", re.I),
    "GeneticsGenomics": re.compile(r"genom|genetic|transcriptom|sequenc|crispr|knockout|mutation", re.I),
}
BUCKETS = list(TOPIC_REGEX.keys()) + ["Misc"]  # include Misc for leftover/top-up

def pick_col(fieldnames, *cands):
    """Case-insensitive column picker."""
    lookup = {f.lower(): f for f in fieldnames if f}
    for c in cands:
        if c is None: continue
        k = c.lower()
        if k in lookup:
            return lookup[k]
    return None

def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
        fields = rows[0].keys() if rows else []
    return rows, list(fields)

def write_csv(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

def xml_year_lang(xml_path: Path):
    """Return (year, lang_str) from a JATS xml file. Tolerant parsing."""
    try:
        root = ET.parse(str(xml_path), ET.XMLParser(recover=True, huge_tree=True)).getroot()
    except Exception:
        return None, ""
    # Year
    y = None
    y_el = root.find(".//pub-date/year")
    if y_el is not None and y_el.text:
        txt = y_el.text.strip()
        if txt.isdigit():
            y = int(txt)
        else:
            m = re.search(r"(\d{4})", txt)
            if m:
                y = int(m.group(1))
    # Language: xml:lang on <article> or on <article-meta>
    lang = (root.get("{http://www.w3.org/XML/1998/namespace}lang") or "").lower()
    if not lang:
        am = root.find(".//article-meta")
        if am is not None:
            lang = (am.get("{http://www.w3.org/XML/1998/namespace}lang") or "").lower()
    return y, lang

def assign_topic(title: str) -> str:
    t = title or ""
    for name, rx in TOPIC_REGEX.items():
        if rx.search(t):
            return name
    return "Misc"

def resolve_xml_path(r: dict, fields, pmcid: str) -> Path | None:
    """Resolve the on-disk JATS XML for a given row/PMCID.
       Priority: fulltext_file -> pmc_fulltext_xml -> glob by PMCID in raw/pmc_xml.
    """
    col_xmlpath = pick_col(fields, "fulltext_file", "fulltext_xml", "xml_path", "fulltext_path")
    col_pmc_xml = pick_col(fields, "pmc_fulltext_xml")
    xml_abs = None

    # 1) explicit fulltext_file if present
    xml_rel_full = (r.get(col_xmlpath) or "").strip() if col_xmlpath else ""
    if xml_rel_full:
        cand = ROOT / xml_rel_full
        if cand.exists():
            xml_abs = cand
        else:
            # try path as-is (absolute or cwd-relative)
            p = Path(xml_rel_full)
            if p.exists():
                xml_abs = p

    # 2) pmc_fulltext_xml column as fallback
    if xml_abs is None and col_pmc_xml:
        xml_rel_pmc = (r.get(col_pmc_xml) or "").strip()
        if xml_rel_pmc:
            cand = ROOT / xml_rel_pmc
            if cand.exists():
                xml_abs = cand
            else:
                p = Path(xml_rel_pmc)
                if p.exists():
                    xml_abs = p

    # 3) glob by PMCID (handles suffixes like _0_14)
    if xml_abs is None and pmcid:
        raw_dir = ROOT / "data_pipeline" / "data" / "raw" / "pmc_xml"
        hits = list(raw_dir.glob(f"{pmcid}*.xml")) + list(raw_dir.glob(f"{pmcid}*.nxml"))
        if hits:
            # pick the largest (usually the full article vs. small fragments)
            xml_abs = max(hits, key=lambda x: x.stat().st_size)

    # Sanity check
    if xml_abs and xml_abs.exists() and xml_abs.suffix.lower() in [".xml", ".nxml"]:
        return xml_abs
    return None

def main():
    # Choose input manifest automatically
    in_path = next((p for p in CANDIDATE_INPUTS if p.exists()), None)
    if not in_path:
        print("[err] No manifest found. Expected one of:")
        for p in CANDIDATE_INPUTS:
            print("  -", p)
        return
    print(f"[in] {in_path}")

    rows, fields = read_csv(in_path)
    if not rows:
        print("[err] Input manifest is empty.")
        return

    # Detect columns
    col_pmcid = pick_col(fields, "pmcid")
    col_title = pick_col(fields, "title", "article_title")
    col_pubdate = pick_col(fields, "publication_date", "pub_date", "date")
    col_year = pick_col(fields, "year", "pub_year")
    col_lang = pick_col(fields, "language", "lang")

    # Diagnostics
    total = len(rows)
    miss_pmcid = no_xml = bad_year = non_en = 0

    # Filtering per document standard (English, 2018–2025, PMC JATS XML)
    filtered = []
    for r in rows:
        pmcid = (r.get(col_pmcid) or "").strip().upper() if col_pmcid else ""
        if not pmcid or not pmcid.startswith("PMC"):
            miss_pmcid += 1
            continue

        xml_abs = resolve_xml_path(r, fields, pmcid)
        if xml_abs is None:
            no_xml += 1
            continue

        # Year
        y = None
        if col_year and (r.get(col_year) or "").strip().isdigit():
            y = int((r.get(col_year) or "0").strip())
        if not y and col_pubdate:
            m = re.search(r"(\d{4})", (r.get(col_pubdate) or ""))
            if m:
                y = int(m.group(1))
        if not y:
            y, _ = xml_year_lang(xml_abs)
        if not y or y < 2018 or y > 2025:
            bad_year += 1
            continue

        # Language
        lang = (r.get(col_lang) or "").lower() if col_lang else ""
        if not lang:
            _, lang = xml_year_lang(xml_abs)
        l = lang.strip().lower()
        if l.startswith("de") or "german" in l:
            non_en += 1
            continue

        # Topic
        title = r.get(col_title, "")
        topic = assign_topic(title)

        rr = dict(r)
        rr["topic"] = topic
        rr["year"] = str(y)
        rr["language"] = l or "en"  # default if missing
        rr["fulltext_file"] = str(xml_abs)  # explicit resolved path
        filtered.append(rr)

    print(f"[diag] total={total} | selected_after_filters={len(filtered)}")
    print(f"[diag] dropped: miss_pmcid={miss_pmcid}, no_xml={no_xml}, bad_year(<!2018 or >2025)={bad_year}, non_english={non_en}")

    if not filtered:
        print("[warn] 0 rows after filtering. Check diagnostics/paths.")
        return

    # Balanced sampling to 4000
    buckets = {b: [] for b in BUCKETS}
    for r in filtered:
        buckets.setdefault(r["topic"], []).append(r)

    per_bucket = max(1, 4000 // len(BUCKETS))
    selected = []
    for b in BUCKETS[:-1]:  # main buckets (exclude Misc for now)
        arr = buckets.get(b, [])
        random.shuffle(arr)
        selected.extend(arr[:per_bucket])

    if len(selected) < 4000:
        leftovers = []
        for b in BUCKETS[:-1]:
            leftovers.extend(buckets.get(b, [])[per_bucket:])
        leftovers.extend(buckets.get("Misc", []))
        random.shuffle(leftovers)
        need = 4000 - len(selected)
        selected.extend(leftovers[:need])

    # Emit manifest_4000.csv (≤4000 rows if not enough data yet)
    out_fields = [
        "pmid","pmcid","doi","title","journal","publication_date","year","language",
        "url_source","publisher_url","open_access",
        "metadata_file","abstract_file","fulltext_file",
        "topic"
    ]
    write_csv(OUT_MAN, selected[:min(4000, len(selected))], out_fields)
    print(f"[out] {OUT_MAN} rows={min(4000, len(selected))}")

if __name__ == "__main__":
    main()
