## Biomedical RAG – Text-Only Pipeline (No Images)

This branch provides a **reproducible, lightweight** pipeline to build a local corpus of **open-access PMC full-text XMLs**, chunk them, and index in **Qdrant**.  
Images are **not** downloaded; figure **captions (text)** are included.

### Prereqs
- Python 3.10+
- Docker (for Qdrant)
- (Recommended) NCBI API key

```bash
# Windows PowerShell – create venv & install basics
python -m venv .venv && .\.venv\Scripts\activate
pip install lxml tqdm regex sentence-transformers qdrant-client fastapi uvicorn scikit-learn
0) (optional) Rate & identity
bash
Copy code
# higher rate limits if you have a key
$env:NCBI_API_KEY = "<your_key>"
# overall request rate used by the downloader
$env:MAX_RPS = 4
1) Build Open-Access Manifest (metadata + abstracts)
bash
Copy code
python data_pipeline/build_open_access_corpus.py --target 4000 --start_date 2018/01/01 --end_date 2025/12/31
Outputs per-PMID folder (metadata.json, abstract.txt) and updates:

bash
Copy code
data_pipeline/data/manifests/oa_manifest.csv
2) Download full-text XML only from PMC (fills missing PMCIDs)
bash
Copy code
python data_pipeline/step_fulltext_from_manifest.py
Writes:

swift
Copy code
data_pipeline/data/raw/pmc_xml/PMC*.xml
data_pipeline/data/pubmed_open_access/PMID_XXXX/fulltext.xml
3) Chunk XML into section-pure text (~400 tokens)
bash
Copy code
python data_pipeline/chunker.py
Creates:

swift
Copy code
data_pipeline/data/processed/chunks/chunks.jsonl
4) Run Qdrant & upload chunks
bash
Copy code
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -v ${PWD}\qdrant_storage:/qdrant/storage qdrant/qdrant:latest
python data_pipeline/uploader_qdrant.py