from pathlib import Path
import re, sys, requests

PMCID = sys.argv[1] if len(sys.argv) > 1 else "PMC11601800"
BASE  = "https://pmc.ncbi.nlm.nih.gov"
UA    = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
         "(KHTML, like Gecko) Chrome/120.0 Safari/537.36)")

# Find figure detail pages (F1, F2, ...)
FIG_PAGE = re.compile(r'/articles/%s/figure/([A-Za-z0-9_-]+)/' % PMCID, re.I)
# Extract <img src=...> from those pages
IMG_SRC  = re.compile(r'src=["\'](https://[^"\']+\.(?:jpg|png|tif|svg))["\']', re.I)

sess = requests.Session()
sess.headers.update({"User-Agent": UA})

print(f"[+] Checking article: {BASE}/articles/{PMCID}/")
art = sess.get(f"{BASE}/articles/{PMCID}/", timeout=30)
figs = sorted(set(FIG_PAGE.findall(art.text)))
print(f"    Found {len(figs)} figure pages:", figs)

if not figs:
    print("    No <figure/> pages detected in main HTML.")
    sys.exit(0)

for tag in figs:
    url = f"{BASE}/articles/{PMCID}/figure/{tag}/"
    r = sess.get(url, timeout=30)
    imgs = IMG_SRC.findall(r.text)
    print(f"  {tag}: {len(imgs)} image(s)")
    for img in imgs:
        print("     ", img)
