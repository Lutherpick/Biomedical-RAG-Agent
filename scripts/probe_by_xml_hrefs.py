from pathlib import Path
import re, sys, xml.etree.ElementTree as ET, requests, imghdr

BASE   = "https://pmc.ncbi.nlm.nih.gov"
UA     = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
          "(KHTML, like Gecko) Chrome/120.0 Safari/537.36)")
ROOT   = Path(__file__).resolve().parents[1] / "data_pipeline" / "data"
XMLDIR = ROOT / "raw" / "pmc_xml"

PMCID  = sys.argv[1] if len(sys.argv) > 1 else "PMC11601800"
XML    = next(XMLDIR.glob(f"{PMCID}_*.xml"), None)
if not XML:
    print(f"No XML found for {PMCID} in {XMLDIR}")
    sys.exit(1)

DBG = ROOT / "processed" / "debug" / PMCID
DBG.mkdir(parents=True, exist_ok=True)

def sniff(b: bytes) -> str:
    if not b: return ""
    k = imghdr.what(None, b)
    if k: return k
    head = b.lstrip()[:64].lower()
    if head.startswith(b"<svg") or head.startswith(b"<?xml"): return "svg"
    return ""

def hrefs_from_xml(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    XLINK = "{http://www.w3.org/1999/xlink}href"
    hrefs = []
    for tag in ("graphic", "inline-graphic", "media"):
        for g in root.findall(f".//{{*}}{tag}"):
            h = g.get(XLINK) or g.get("href")
            if h: hrefs.append(h)
    return hrefs

def candidates(pmcid: str, href: str):
    if href.startswith("http"):
        return [href, href + ("&download=1" if "?" in href else "?download=1")]
    base = f"{BASE}/articles/{pmcid}"
    return [
        f"{base}/bin/{href}",
        f"{base}/bin/{href}?download=1",
        f"{base}/pdf/{href}",
        f"{base}/pdf/{href}?download=1",
    ]

print(f"[XML] {XML.name}")
hrefs = hrefs_from_xml(XML)
print(f"Found {len(hrefs)} href(s) in XML")
for i,h in enumerate(hrefs[:10],1):
    print(f"  {i}. {h}")

s = requests.Session()
s.headers.update({"User-Agent": UA, "Referer": f"{BASE}/articles/{PMCID}/"})

# prime cookies by visiting article page once
art = s.get(f"{BASE}/articles/{PMCID}/", timeout=30)
open(DBG/"article.html","w",encoding="utf-8").write(art.text)
print(f"[article] {art.status_code} {art.headers.get('Content-Type')}")

n = 0
for h in hrefs:
    for url in candidates(PMCID, h):
        try:
            r = s.get(url, timeout=30, allow_redirects=True,
                      headers={"Accept":"image/avif,image/webp,image/apng,image/*,*/*;q=0.8"})
        except Exception as e:
            print("    ->", url, "ERR", e)
            continue
        ct = (r.headers.get("Content-Type") or "").lower()
        kind = sniff(r.content)
        first = r.content[:12].hex(" ")
        print(f"    -> {url} :: {r.status_code} ct={ct} sniff={kind} bytes={first}")
        name = url.split("/")[-1].split("?")[0]
        out  = DBG / f"{n:03d}-{name}{'' if kind else '.html'}"
        out.write_bytes(r.content)
        n += 1

print(f"[done] saved to {DBG}")
