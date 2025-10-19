# scripts/verify_figures.py
from pathlib import Path

# Project-root aware
ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = [
    ROOT / "data_pipeline" / "data" / "processed" / "figures",
    ROOT / "data" / "processed" / "figures",
    ]
FIG = next((p for p in CANDIDATES if p.exists()), CANDIDATES[0])

MAGIC = {
    b"\xFF\xD8\xFF": "jpg",
    b"\x89PNG\r\n\x1A\n": "png",
    b"GIF87a": "gif",
    b"GIF89a": "gif",
    b"II*\x00": "tif",
    b"MM\x00*": "tif",
}
def sniff(p: Path):
    b = p.read_bytes()[:32]
    for sig, kind in MAGIC.items():
        if b.startswith(sig): return kind
    if b.lstrip().startswith(b"<svg") or b.lstrip().startswith(b"<?xml"):
        return "svg"
    return "unknown"

print("FIG dir:", FIG)
all_files = [f for f in FIG.rglob("*") if f.is_file()]
print("Files total:", len(all_files))

bad = []
by_kind = {"jpg":0,"png":0,"gif":0,"tif":0,"svg":0,"unknown":0}
for f in all_files:
    kind = sniff(f)
    by_kind[kind] = by_kind.get(kind,0)+1
    if kind == "unknown":
        bad.append(f)

print("By kind:", by_kind)
print("Non-image / wrong files:", len(bad))
for x in bad[:25]:
    print(" -", x)
