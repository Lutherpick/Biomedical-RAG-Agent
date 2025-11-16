#!/usr/bin/env python3
r"""
Paragraph-first chunker for NIHMS-PDF XML and JATS XML.
Each paragraph = one chunk. No cross-paragraph merges.
If a paragraph exceeds --max-tokens, split only at sentence boundaries.

Output JSONL fields:
{pmcid, pmid, nihmsid, title, journal, year, source_file,
 section, subsection, text, type,
 section_index, subsection_index, para_local_index, split_index, chunk_index}

Examples (PowerShell):
  python .\parse_chunk.py --xml .\pmc_chunker\out\PMC9107928_out.xml --out .\pmc_chunker\out\paragraph_chunks.jsonl
  python .\parse_chunk.py --xml-dir .\pmc_chunker\data\xml --out .\pmc_chunker\out\paragraph_chunks_4000.jsonl
  python .\parse_chunk.py --xml "pmc_chunker\data\xml\*.xml" --out "pmc_chunker\out\paragraph_chunks_4000.jsonl" --per-file
"""

import argparse, json, re, sys, glob, os
from pathlib import Path
import xml.etree.ElementTree as ET

from chunker import token_count, split_overlong_paragraph, split_paragraphs_clean

# heuristics
_SENT_TOKEN = re.compile(r"\b(is|are|was|were|be|being|been|of|to|in|and|for|with|that|which)\b", re.I)
_FIG_LEAD = re.compile(r"^\s*(?:Figure|Fig\.)\s*\d+[A-Za-z]?(?:\s*[:.,)\-–])?", re.I)

def _ln(tag: str) -> str:
    return tag.split("}", 1)[1] if tag and tag[0] == "{" else (tag or "")

def find_child_text_any(e: ET.Element, local: str) -> str:
    for c in list(e):
        if _ln(c.tag) == local:
            return c.text or ""
    return ""

def find_attr_any(e: ET.Element, name: str, default: str = "") -> str:
    return e.get(name, default)

def iter_local(e: ET.Element, local: str):
    for n in e.iter():
        if _ln(n.tag) == local:
            yield n

def _norm_section(s: str) -> str:
    if not s: return ""
    s = s.lower().strip()
    s = re.sub(r'^\s*(\d+(?:\.\d+)*[\.\)])\s*', '', s)  # drop "1.2" prefixes
    s = re.sub(r'^\s*(section|chapter|part)\s+\d+[:\.\)]\s*', '', s)
    return s

def classify_paragraph(section: str, subsection: str, text: str) -> str:
    if (subsection or "").strip().lower() == "keywords":
        return "keywords"
    if _FIG_LEAD.match((subsection or "")) or _FIG_LEAD.match((text or "")):
        return "figure_caption"
    s = _norm_section(section)
    if s.startswith("abstract"): return "abstract"
    if s.startswith("introduction"): return "introduction"
    if s.startswith("materials and methods") or s.startswith("methods") or s == "materials":
        return "step" if re.match(r"^\s*\d+[\).\s]", text or "") else "method_paragraph"
    if s.startswith("results"): return "results_paragraph"
    if s.startswith("discussion"): return "discussion_paragraph"
    if s.startswith("references"): return "reference_entry"
    return "paragraph"

def looks_like_sentence_fragment(sub: str, nxt: str) -> bool:
    """
    Detect NIHMS 'subsection' lines that are just the beginning of the next paragraph.
    Merge them into text if so.
    """
    if not sub: return False
    s = sub.strip()
    if not s or s.lower().startswith("page "): return False
    if s.lower() in {"keywords", "acknowledgments", "acknowledgements"}: return False
    if s.endswith((".", ":", "?", "!", ";")): return False
    long_enough = len(s) >= 40 or (len(s.split()) >= 6 and _SENT_TOKEN.search(s))
    nxt_stripped = (nxt or "").lstrip()
    nxt_starts_lower_or_punct = (nxt_stripped[:1].islower() or (nxt_stripped and not nxt_stripped[0].isalnum()))
    return bool(long_enough and nxt_starts_lower_or_punct)

# ---------- NIHMS-PDF XML ----------

def parse_flat_records(root: ET.Element):
    """
    NIHMS converter outputs either:
      <flat_records><record>…<section>…<subsection>…<text>…</record>…
    or a hierarchical <sections><section><subsection><block><text>…</text></block>…
    """
    recs = []
    flat_records = list(iter_local(root, "flat_records"))
    if flat_records:
        flat = flat_records[0]
        for r in iter_local(flat, "record"):
            recs.append({
                "pmcid":       find_child_text_any(r, "pmcid"),
                "pmid":        find_child_text_any(r, "pmid"),
                "nihmsid":     find_child_text_any(r, "nihmsid"),
                "title":       find_child_text_any(r, "title"),
                "journal":     find_child_text_any(r, "journal"),
                "year":        find_child_text_any(r, "year"),
                "source_file": find_child_text_any(r, "source_file"),
                "section":     find_child_text_any(r, "section"),
                "subsection":  find_child_text_any(r, "subsection"),
                "text":        find_child_text_any(r, "text"),
                "section_index":     int((find_child_text_any(r, "section_index") or "0") or 0),
                "subsection_index":  int((find_child_text_any(r, "subsection_index") or "0") or 0),
                "char_len":          int((find_child_text_any(r, "char_len") or "0") or 0),
                "figure_id":         find_child_text_any(r, "figure_id") or "",
            })
        return recs

    sections = list(iter_local(root, "sections"))
    if not sections: return recs
    sections = sections[0]
    for sec in iter_local(sections, "section"):
        section_name = find_attr_any(sec, "name", "")
        sec_idx = int(find_attr_any(sec, "index", "0") or 0)
        for sub in iter_local(sec, "subsection"):
            subsection_name = find_attr_any(sub, "name", "")
            sub_idx = int(find_attr_any(sub, "index", "0") or 0)
            for blk in iter_local(sub, "block"):
                txt = find_child_text_any(blk, "text")
                char_len = find_attr_any(blk, "char_len", "")
                recs.append({
                    "pmcid":       find_attr_any(root, "pmcid", ""),
                    "pmid":        find_attr_any(root, "pmid", ""),
                    "nihmsid":     find_attr_any(root, "nihmsid", ""),
                    "title":       find_attr_any(root, "title", ""),
                    "journal":     find_attr_any(root, "journal", ""),
                    "year":        find_attr_any(root, "year", ""),
                    "source_file": find_attr_any(root, "source_file", ""),
                    "section":     section_name,
                    "subsection":  subsection_name,
                    "text":        txt or "",
                    "section_index":    sec_idx,
                    "subsection_index": sub_idx,
                    "char_len":         int(char_len or len(txt or "")),
                    "figure_id":        find_attr_any(blk, "figure_id", ""),
                })
    return recs

def _repair_and_clean(rec):
    """Drop useless 'Page N' subsections and merge subsection fragments into text when needed."""
    sub = rec.get("subsection") or ""
    txt = rec.get("text") or ""
    out = dict(rec)
    if re.match(r"^Page\s+\d+(?:\s*of\s+\d+)?$", sub, flags=re.I):
        out["subsection"] = ""
    if looks_like_sentence_fragment(sub, txt):
        merged_text = (sub.strip() + " " + txt.lstrip()).strip()
        out["raw_subsection"] = sub
        out["subsection"] = ""
        out["text"] = merged_text
        return out
    out["text"] = txt
    return out

# ---------- JATS XML ----------

def jats_iter_paragraph_records(root: ET.Element):
    """Yield records in the same dict shape as NIHMS for easy unification."""
    meta = {
        "pmcid": "",
        "pmid": "",
        "nihmsid": "",
        "title": "",
        "journal": "",
        "year": "",
        "source_file": "",
    }
    # front matter
    art = next(iter_local(root, "article"), None)
    if art is not None:
        meta["source_file"] = find_attr_any(art, "xlink:href", "") or ""
    front = next(iter_local(root, "front"), None)
    if front is not None:
        for t in iter_local(front, "article-title"):
            meta["title"] = " ".join(" ".join(t.itertext()).split()); break
        for j in iter_local(front, "journal-title"):
            meta["journal"] = " ".join(" ".join(j.itertext()).split()); break
        for a in iter_local(front, "article-id"):
            t = (a.text or "").strip()
            if a.get("pub-id-type") == "pmid": meta["pmid"] = t
            if a.get("pub-id-type") == "pmcid": meta["pmcid"] = t
            if a.get("pub-id-type") == "nihmsid": meta["nihmsid"] = t

    # abstract
    abs_elt = next(iter_local(root, "abstract"), None)
    if abs_elt is not None:
        ps = [p for p in iter_local(abs_elt, "p")]
        if ps:
            for k, p in enumerate(ps):
                txt = " ".join(" ".join(p.itertext()).split())
                for para in split_paragraphs_clean(txt):
                    yield dict(meta, section="Abstract", subsection="", text=para,
                               section_index=-1, subsection_index=k, figure_id="")
        else:
            txt = " ".join(" ".join(abs_elt.itertext()).split())
            for para in split_paragraphs_clean(txt):
                yield dict(meta, section="Abstract", subsection="", text=para,
                           section_index=-1, subsection_index=0, figure_id="")

    # body
    sec_idx = -1
    for sec in iter_local(root, "sec"):
        sec_idx += 1
        title = ""
        for t in iter_local(sec, "title"):
            title = " ".join(" ".join(t.itertext()).split()); break
        sub_idx = -1
        for p in iter_local(sec, "p"):
            sub_idx += 1
            txt = " ".join(" ".join(p.itertext()).split())
            for para in split_paragraphs_clean(txt):
                yield dict(meta, section=title or "sec", subsection="", text=para,
                           section_index=sec_idx, subsection_index=sub_idx, figure_id="")
        for fig in iter_local(sec, "fig"):
            cap = next(iter_local(fig, "caption"), None)
            if cap is None: continue
            cap_txt = " ".join(" ".join(cap.itertext()).split())
            if not cap_txt: continue
            yield dict(meta, section=title or "sec",
                       subsection="", text=cap_txt, section_index=sec_idx,
                       subsection_index=9999, figure_id=(fig.get("id") or ""),)

# ---------- core processing ----------

def process_one_xml(xml_path: Path, max_tokens: int, overlap_sentences: int):
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError as e:
        print(f"ParseError: {xml_path} :: {e}", file=sys.stderr, flush=True)
        return

    is_jats = any(_ln(t.tag) in {"article", "front", "body", "sec"} for t in root.iter())
    raw = list(jats_iter_paragraph_records(root)) if is_jats else parse_flat_records(root)

    if not raw:
        yield {"pmcid": "", "source_file": str(xml_path.name), "section": "", "subsection": "", "text": "",
               "type": "empty_document", "para_local_index": 0, "split_index": 0}
        return

    para_local_index = 0
    for rec in raw:
        rec = _repair_and_clean(rec) if not is_jats else rec
        rec_type = classify_paragraph(rec.get("section",""), rec.get("subsection",""), rec.get("text",""))

        if rec_type == "figure_caption":
            out = dict(rec); out["type"] = "figure_caption"
            out["para_local_index"] = para_local_index; out["split_index"] = 0
            yield out; para_local_index += 1
            continue

        paragraphs = split_paragraphs_clean(rec.get("text",""))
        if not paragraphs:
            para_local_index += 1
            continue

        for k, para in enumerate(paragraphs):
            if token_count(para) > max_tokens:
                pieces = split_overlong_paragraph(para, max_tokens, overlap_sentences=overlap_sentences)
                for j, piece in enumerate(pieces):
                    out = dict(rec)
                    out["text"] = piece
                    out["type"] = classify_paragraph(out["section"], out["subsection"], piece)
                    out["para_local_index"] = para_local_index
                    out["split_index"] = j
                    yield out
            else:
                out = dict(rec)
                out["text"] = para
                out["type"] = classify_paragraph(out["section"], out["subsection"], para)
                out["para_local_index"] = para_local_index
                out["split_index"] = 0
                yield out
            para_local_index += 1

def write_jsonl_stream(out_path: Path, rows_iter, per_doc_counter: dict):
    with out_path.open("a", encoding="utf-8") as f:
        for r in rows_iter:
            pmcid = r.get("pmcid") or "UNKNOWN"
            idx = per_doc_counter.get(pmcid, 0)
            r2 = dict(r); r2["chunk_index"] = idx
            per_doc_counter[pmcid] = idx + 1
            f.write(json.dumps(r2, ensure_ascii=False) + "\n")

def write_per_file_jsonl(path: Path, rows):
    counters = {}
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            pmcid = r.get("pmcid") or "UNKNOWN"
            idx = counters.get(pmcid, 0)
            r2 = dict(r); r2["chunk_index"] = idx
            counters[pmcid] = idx + 1
            f.write(json.dumps(r2, ensure_ascii=False) + "\n")

def expand_inputs(xml_patterns, xml_dir):
    files = []
    if xml_dir:
        base = Path(xml_dir)
        if base.exists():
            for name in os.listdir(base):
                if name.lower().endswith(".xml"):
                    files.append(str((base / name).resolve()))
    for pat in xml_patterns:
        matched = glob.glob(pat); files.extend(matched if matched else [pat])
    seen, uniq = set(), []
    for f in files:
        if f not in seen:
            uniq.append(f); seen.add(f)
    return [Path(f) for f in uniq]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", nargs="*", default=[], help="NIHMS/JATS XML file patterns")
    ap.add_argument("--xml-dir", default="", help="Directory with .xml files")
    ap.add_argument("--out", required=True, help="Merged JSONL output path")
    ap.add_argument("--per-file", action="store_true", help="Also write per-file *_out.jsonl")
    ap.add_argument("--max-tokens", type=int, default=800, help="Guardrail per paragraph")
    ap.add_argument("--overlap-sentences", type=int, default=1, help="Overlap when splitting overlong paragraphs")
    ap.add_argument("--progress-every", type=int, default=50, help="Log every N files")
    args = ap.parse_args()

    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("", encoding="utf-8")

    per_doc_counter = {}
    file_list = expand_inputs(args.xml, args.xml_dir)
    if not file_list:
        print("No XML inputs resolved.", file=sys.stderr); sys.exit(1)

    total = len(file_list)
    for i, p in enumerate(file_list, 1):
        if not p.exists():
            print(f"Not found: {p}", file=sys.stderr, flush=True)
            continue

        # Parse ONCE per file
        rows = list(process_one_xml(p, max_tokens=args.max_tokens, overlap_sentences=args.overlap_sentences))
        write_jsonl_stream(out_path, iter(rows), per_doc_counter)
        if args.per_file:
            write_per_file_jsonl(p.with_name(f"{p.stem}_out.jsonl"), rows)

        if args.progress_every and i % args.progress_every == 0:
            print(f"[progress] {i}/{total} files", flush=True)

    print(f"[done] processed {total} files → {out_path}", flush=True)

if __name__ == "__main__":
    main()
