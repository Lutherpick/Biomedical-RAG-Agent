# scripts/extract_figures_pmc.py
from pathlib import Path
import io, json, re, tarfile, time, xml.etree.ElementTree as ET
import requests

# ---------- Paths ----------
DATA_ROOT = Path(__file__).resolve().parents[1] / "data_pipeline" / "data"
XML_DIR   = DATA_ROOT / "raw" / "pmc_xml"
OUT_DIR   = DATA_ROOT / "processed" / "figures"
OUT_META  = DATA_ROOT / "processed" / "figures_meta.jsonl"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Config ----------
OA_API   = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
UA       = "biomed-rag-agent/1.0 (+github.com/Lutherpick/Biomedical-RAG-Agent)"
IMG_EXTS = (".jpg", ".jpeg", ".png", ".gif", ".tif", ".tiff", ".svg", ".bmp", ".webp")
SLEEP_BETWEEN = 0.10  # seconds, be polite to PMC

# ---------- Optional TIFF→JPEG ----------
try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

def convert_tiff_to_jpeg(path: Path) -> Path:
    if not PIL_AVAILABLE:
        return path
    if path.suffix.lower() not in (".tif", ".tiff"):
        return path
    try:
        img = Image.open(path)
        out = path.with_suffix(".jpg")
        img.convert("RGB").save(out, quality=92, optimize=True)
        path.unlink(missing_ok=True)
        return out
    except Exception:
        return path

# ---------- Utils ----------
def log(msg: str) -> None:
    print(msg, flush=True)

def pmcid_from_filename(name: str):
    m = re.match(r"(PMC\d+)", name)
    return m.group(1) if m else None

def all_xml_files(d: Path):
    return sorted({p for p in d.glob("*.xml")})

def fetch_oa_record(pmcid: str):
    try:
        r = requests.get(OA_API, params={"id": pmcid}, timeout=60,
                         headers={"User-Agent": UA})
        if r.status_code != 200 or not r.text:
            return None
        return ET.fromstring(r.text)
    except Exception:
        return None

def collect_img_urls(oa_root):
    urls = []
    for f in oa_root.findall(".//file"):
        href = (f.get("href") or "").strip()
        fmt  = (f.get("format") or "").lower()
        if href and (fmt == "img" or href.lower().endswith(IMG_EXTS)):
            urls.append(href)
    # de-duplicate while preserving order
    seen, uniq = set(), []
    for u in urls:
        if u not in seen:
            uniq.append(u); seen.add(u)
    return uniq

def find_tgz_link(oa_root):
    for l in oa_root.findall(".//link"):
        fmt = (l.get("format") or "").lower()
        href = (l.get("href") or "").strip()
        if fmt == "tgz" and href:
            if href.startswith("ftp://"):
                href = href.replace("ftp://", "https://", 1)
            return href
    return None

def save_bytes(dir_path: Path, fname: str, blob: bytes) -> Path:
    dir_path.mkdir(parents=True, exist_ok=True)
    fname = fname.split("?")[0]
    out = dir_path / fname
    # resume-safe skip
    if out.exists() and out.stat().st_size > 0:
        return out
    out.write_bytes(blob)
    return out

# ---------- Captions from local XML ----------
def text_of(n):
    return "" if n is None else "".join(n.itertext()).strip()

def build_caption_map(pmcid: str):
    """
    Returns two dicts:
      name_map[basename] -> (fig_id, caption)
      stem_map[stem_lower] -> (fig_id, caption)  (fallback: ignore extension/case)
    """
    name_map, stem_map = {}, {}
    xml_path = next(XML_DIR.glob(f"{pmcid}_*.xml"), None)
    if not xml_path:
        return name_map, stem_map
    try:
        root = ET.parse(xml_path).getroot()
    except Exception:
        return name_map, stem_map

    XLINK = "{http://www.w3.org/1999/xlink}href"
    figs = root.findall(".//{*}fig")
    for idx, fig in enumerate(figs, 1):
        fid = fig.get("id") or f"fig{idx}"
        cap = text_of(fig.find(".//{*}caption"))
        for tag in ("graphic", "inline-graphic", "media"):
            for g in fig.findall(f".//{{*}}{tag}"):
                href = g.get(XLINK) or g.get("href")
                if not href:
                    continue
                base = Path(href).name
                stem = Path(base).stem.lower()
                if base and base not in name_map:
                    name_map[base] = (fid, cap)
                if stem and stem not in stem_map:
                    stem_map[stem] = (fid, cap)
    return name_map, stem_map

def lookup_caption(name_map, stem_map, saved_fname: str):
    base = Path(saved_fname).name
    stem = Path(base).stem.lower()
    return name_map.get(base) or stem_map.get(stem) or (None, None)

# ---------- Main ----------
def main():
    log(f"DATA_ROOT: {DATA_ROOT}")
    log(f"XML_DIR:   {XML_DIR}")
    files = all_xml_files(XML_DIR)
    log(f"Found {len(files)} XML files")

    n_images = 0

    with OUT_META.open("a", encoding="utf-8") as meta_out:
        for xf in files:
            pmcid = pmcid_from_filename(xf.name)
            if not pmcid:
                log(f"[{xf.name}] PMCID=? (skip)"); continue

            cap_name_map, cap_stem_map = build_caption_map(pmcid)

            oa = fetch_oa_record(pmcid)
            if oa is None:
                log(f"[{pmcid}] OA query failed"); continue

            paper_dir = OUT_DIR / pmcid
            saved_any = False

            # 1) Direct per-file URLs (best path)
            img_urls = collect_img_urls(oa)
            if img_urls:
                log(f"[{pmcid}] OA per-file images: {len(img_urls)}")
                s = requests.Session()
                s.headers.update({"User-Agent": UA})
                for u in img_urls:
                    try:
                        r = s.get(u, timeout=90)
                        if r.status_code == 200 and r.content:
                            fname = u.split("/")[-1]
                            out = save_bytes(paper_dir, fname, r.content)
                            out = convert_tiff_to_jpeg(out)
                            fid, caption = lookup_caption(cap_name_map, cap_stem_map, out.name)
                            meta_out.write(json.dumps({
                                "pmcid": pmcid,
                                "fig_id": fid,
                                "href": u,
                                "url_used": u,
                                "saved_as": str(out.relative_to(DATA_ROOT)),
                                "caption": caption
                            }, ensure_ascii=False) + "\n")
                            n_images += 1
                            saved_any = True
                        time.sleep(SLEEP_BETWEEN)
                    except Exception:
                        continue

            # 2) TGZ fallback (extract images only)
            if not saved_any:
                tgz = find_tgz_link(oa)
                if tgz:
                    log(f"[{pmcid}] downloading TGZ package…")
                    try:
                        r = requests.get(tgz, timeout=180, headers={"User-Agent": UA})
                        if r.status_code == 200 and r.content:
                            with tarfile.open(fileobj=io.BytesIO(r.content), mode="r:gz") as tar:
                                for m in tar.getmembers():
                                    if not (m.isfile() and m.name.lower().endswith(IMG_EXTS)):
                                        continue
                                    data = tar.extractfile(m).read()
                                    out_name = Path(m.name).name
                                    out = save_bytes(paper_dir, out_name, data)
                                    out = convert_tiff_to_jpeg(out)
                                    fid, caption = lookup_caption(cap_name_map, cap_stem_map, out.name)
                                    meta_out.write(json.dumps({
                                        "pmcid": pmcid,
                                        "fig_id": fid,
                                        "href": f"{tgz}::{m.name}",
                                        "url_used": tgz,
                                        "saved_as": str(out.relative_to(DATA_ROOT)),
                                        "caption": caption
                                    }, ensure_ascii=False) + "\n")
                                    n_images += 1
                                    saved_any = True
                        log(f"[{pmcid}] images extracted from TGZ.")
                    except Exception as e:
                        log(f"[{pmcid}] TGZ fetch/unpack error: {e}")

            if not saved_any:
                log(f"[{pmcid}] no images found via OA (likely not in OA subset).")

    log(f"Images downloaded: {n_images}")

if __name__ == "__main__":
    main()
