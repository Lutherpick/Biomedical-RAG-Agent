# data_pipeline/search_pubmed_sliced.py
from __future__ import annotations
from pathlib import Path
from Bio import Entrez
import argparse, json, sys, time, os, random, datetime as dt

# --- Identify yourself to NCBI (keep! add API key for higher limits) ---
Entrez.email   = "bazefi01@thu.de"
Entrez.tool    = "biomed-rag-ingester"
Entrez.api_key = os.getenv("NCBI_API_KEY")  # optional but recommended

# Base query WITHOUT dates; we will inject dates via esearch params (mindate/maxdate)
QUERY_BASE = (
    '("Animal Experimentation"[MeSH] OR "Models, Animal"[MeSH] OR "Disease Models, Animal"[MeSH] '
    'OR "Preclinical Studies as Topic"[MeSH] OR "Drug Evaluation, Preclinical"[MeSH] '
    'OR "Toxicity Tests"[MeSH] OR "In Vivo Techniques"[MeSH]) '
    'AND (Animals[mh] NOT Humans[mh]) '
    'AND ("animal experiment*"[tiab] OR "animal model*"[tiab] OR "preclinical"[tiab] '
    'OR "in vivo"[tiab] OR mouse[tiab] OR mice[tiab] OR rat[tiab] OR rats[tiab] OR rabbit[tiab] '
    'OR rabbits[tiab] OR dog[tiab] OR dogs[tiab] OR canine[tiab] OR pig[tiab] OR pigs[tiab] '
    'OR swine[tiab] OR porcine[tiab] OR sheep[tiab] OR ovine[tiab] OR cattle[tiab] OR bovine[tiab] '
    'OR zebrafish[tiab] OR xenopus[tiab] OR "chick embryo"[tiab]) '
    'AND hasabstract[text] '
    'AND (english[lang] OR german[lang]) '
    'AND free full text[sb]'
)

def esearch_count(term: str, mindate: str, maxdate: str, retries: int = 5) -> int:
    delay = 0.6
    for attempt in range(1, retries + 1):
        try:
            with Entrez.esearch(
                    db="pubmed", term=term, retmax=0, retstart=0,
                    datetype="pdat", mindate=mindate, maxdate=maxdate, usehistory="n", retmode="xml"
            ) as h:
                data = Entrez.read(h)
            return int(data.get("Count", 0))
        except Exception as e:
            print(f"[count] attempt {attempt}/{retries} failed ({mindate}..{maxdate}): {e}", file=sys.stderr)
            time.sleep(delay + random.uniform(0, 0.5))
            delay = min(delay * 2, 8)
    print(f"[count] giving up on {mindate}..{maxdate}", file=sys.stderr)
    return 0

def esearch_page(term: str, retmax: int, retstart: int, mindate: str, maxdate: str, retries: int = 6) -> list[str]:
    """Robust esearch page fetch with date constraints."""
    delay = 0.5
    for attempt in range(1, retries + 1):
        try:
            with Entrez.esearch(
                    db="pubmed",
                    term=term,
                    retmax=retmax,
                    retstart=retstart,
                    datetype="pdat",
                    mindate=mindate,
                    maxdate=maxdate,
                    usehistory="n",
                    sort="relevance",
                    retmode="xml",
            ) as h:
                data = Entrez.read(h)
            return list(map(str, data.get("IdList", [])))
        except Exception as e:
            print(f"[esearch] attempt {attempt}/{retries} failed retstart={retstart} size={retmax} "
                  f"({mindate}..{maxdate}): {e}", file=sys.stderr)
            time.sleep(delay + random.uniform(0, 0.5))
            delay = min(delay * 2, 8)
    print(f"[esearch] giving up page retstart={retstart} size={retmax} ({mindate}..{maxdate})", file=sys.stderr)
    return []

def daterange_slices(start: dt.date, end: dt.date, term: str, max_per_bin: int = 9000) -> list[tuple[str, str]]:
    """
    Recursively split [start, end] by date until each slice count ≤ max_per_bin.
    Returns list of (mindate_str, maxdate_str) in ISO 'YYYY/MM/DD'.
    """
    def iso(d: dt.date) -> str: return d.strftime("%Y/%m/%d")
    count = esearch_count(term, iso(start), iso(end))
    if count <= max_per_bin:
        return [(iso(start), iso(end))]
    # split in half
    mid = start + (end - start) / 2
    mid = dt.date.fromordinal(int(mid.toordinal()))  # ensure date
    left  = daterange_slices(start, mid, term, max_per_bin)
    right = daterange_slices(mid + dt.timedelta(days=1), end, term, max_per_bin)
    return left + right

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=120000, help="Total PMIDs desired (will stop after this many)")
    ap.add_argument("--page", type=int, default=1000, help="Pagination size per request (≤ 5000 recommended)")
    ap.add_argument("--out", type=str, default="pubmed_120k.jsonl")
    ap.add_argument("--start_date", type=str, default="2010-01-01")
    ap.add_argument("--end_date", type=str,   default="2025-12-31")
    ap.add_argument("--query", type=str, default=QUERY_BASE)
    ap.add_argument("--bin_max", type=int, default=9000, help="Max results allowed per date-bin before splitting")
    args = ap.parse_args()

    out_path = Path(__file__).resolve().parent / args.out

    # Resume (dedupe)
    seen: set[str] = set()
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    seen.add(json.loads(line)["pmid"])
                except Exception:
                    pass
        print(f"[resume] found {len(seen)} existing PMIDs in {out_path}", file=sys.stderr)

    # Build date bins that each satisfy PubMed's 9,999 cap
    sd = dt.datetime.strptime(args.start_date, "%Y-%m-%d").date()
    ed = dt.datetime.strptime(args.end_date, "%Y-%m-%d").date()
    bins = daterange_slices(sd, ed, args.query, max_per_bin=args.bin_max)
    print(f"[bins] {len(bins)} date bins generated", file=sys.stderr)

    wrote = len(seen)
    # Iterate bins in chronological order (or reverse if you prefer recent-first)
    with out_path.open("a", encoding="utf-8") as out:
        for mindate, maxdate in bins:
            if wrote >= args.target:
                break
            # Count again (cheap) for progress info
            cnt = esearch_count(args.query, mindate, maxdate)
            print(f"[bin] {mindate} .. {maxdate} ≈ {cnt} results", file=sys.stderr)

            # Paginate within the bin
            retstart = 0
            while retstart < cnt and wrote < args.target:
                size = min(args.page, args.target - wrote, 5000)
                ids = esearch_page(args.query, retmax=size, retstart=retstart, mindate=mindate, maxdate=maxdate)
                if not ids:
                    # failed page; skip ahead to avoid infinite loop
                    retstart += size
                    continue
                new = 0
                for pmid in ids:
                    if pmid in seen:
                        continue
                    out.write(json.dumps({"pmid": pmid}) + "\n")
                    seen.add(pmid)
                    wrote += 1
                    new += 1
                    if wrote >= args.target:
                        break
                print(f"Fetched PMIDs: {wrote}/{args.target} (+{new}) [{mindate}..{maxdate} @ {retstart}]", file=sys.stderr)
                retstart += size
                # polite: with API key we can do up to ~3 req/sec; keep it conservative
                time.sleep(0.35)

    print(f"Saved {wrote} PMIDs → {out_path}")

if __name__ == "__main__":
    main()
