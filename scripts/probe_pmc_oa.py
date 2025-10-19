from pathlib import Path
import sys, requests, xml.etree.ElementTree as ET

PMCID = sys.argv[1] if len(sys.argv) > 1 else "PMC11601800"
OA_API = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"

def log(*a): print(*a, flush=True)

def main():
    log(f"[OA] query {PMCID}")
    r = requests.get(OA_API, params={"id": PMCID}, timeout=30,
                     headers={"User-Agent": "biomed-rag-agent/1.0"})
    log("status:", r.status_code, r.headers.get("Content-Type"))
    if r.status_code != 200:
        log("!! OA API failed"); return
    Path("oa_dump.xml").write_text(r.text, encoding="utf-8")

    root = ET.fromstring(r.text)
    # The schema returns <record> with <link> (tgz/pdf) and sometimes <files><file .../></files>
    # Try the <file> list first (best: direct per-asset URLs), else fall back to TGZ package.
    files = []
    for f in root.findall(".//file"):
        href = f.get("href") or ""
        fmt  = f.get("format") or ""
        if href and (fmt == "img" or href.lower().endswith((".jpg",".jpeg",".png",".gif",".tif",".tiff",".svg"))):
            files.append(href)

    log(f"files(img): {len(files)}")
    for u in files[:20]:
        log("  -", u)

    if not files:
        tgz = None
        for l in root.findall(".//link"):
            if (l.get("format") or "").lower() == "tgz":
                tgz = l.get("href")
        if tgz:
            log("No per-file URLs; package is available:")
            log("  tgz:", tgz)
        else:
            log("No OA files returned (article may not be in the OA subset).")

if __name__ == "__main__":
    main()
