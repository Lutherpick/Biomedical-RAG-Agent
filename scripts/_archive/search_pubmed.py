from __future__ import annotations
import argparse
import math
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
from data_pipeline.config import load_config
from data_pipeline.ncbi import esearch, efetch_batch

"""
This script implements the doc's 'Document retrieval' phase up to saving all XML:
1) ESearch (usehistory=y, retmax=0) to get Count, WebEnv, QueryKey.
2) EFetch in batches of 10,000 (retmode=xml, rettype=xml).
3) Save raw XML pages under data/pubmed_xml/page_00001.xml, ... (no parsing yet).
"""

def main():
    load_dotenv()
    cfg = load_config()

    ap = argparse.ArgumentParser(description="PubMed ESearch+EFetch (doc-compliant).")
    ap.add_argument("--query", required=True, help="EXACT query string from the document.")
    ap.add_argument("--out", default="data/data/pubmed_xml", help="Output directory for XML pages.")
    ap.add_argument("--batch", type=int, default=10000, help="EFetch batch size (doc uses 10000).")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) ESearch
    print("[ESearch] contacting PubMed…")
    total, webenv, qk = esearch(cfg.base, cfg.email, cfg.api_key, args.query)
    print(f"[ESearch] count={total:,} | QueryKey={qk} | WebEnv(len)={len(webenv)}")

    # 2) EFetch in batches of 10,000
    pages = math.ceil(total / args.batch) if total else 0
    if pages == 0:
        print("[EFetch] No records to fetch (count=0). Check your query.")
        return

    for i in tqdm(range(pages), desc="[EFetch] downloading XML pages"):
        retstart = i * args.batch
        xml_text = efetch_batch(cfg.base, cfg.email, cfg.api_key, webenv, qk, retstart, args.batch)
        out_file = out_dir / f"page_{i+1:05d}.xml"
        out_file.write_text(xml_text, encoding="utf-8")

    # 3) Done — we now have XML batches locally
    print(f"[done] Saved {pages} XML page file(s) to: {out_dir.resolve()}")
    print("Next step (per doc): iterate each record and download the full texts "
          "(JATS from PMC if PMCID exists, else OA PDF/HTML from publisher).")

if __name__ == "__main__":
    main()
