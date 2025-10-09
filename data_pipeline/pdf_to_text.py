# pip install pymupdf
import fitz, os
from pathlib import Path
PDFDIR=Path("data/raw/pdfs"); TXTDIR=Path("data/processed/pdf_text"); TXTDIR.mkdir(parents=True, exist_ok=True)
for pdf in PDFDIR.glob("*.pdf"):
    txt=TXTDIR/(pdf.stem+".txt")
    if txt.exists(): continue
    text=[]
    with fitz.open(pdf) as doc:
        for p in doc: text.append(p.get_text())
    txt.write_text("\n".join(text),encoding="utf-8")
print("PDF → text done.")
