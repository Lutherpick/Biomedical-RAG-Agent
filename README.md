# 🧬 Biomedical-RAG-Agent

An end-to-end **Retrieval-Augmented Generation (RAG)** pipeline for biomedical literature.

It fetches PubMed/PMC papers, extracts full text (XML + optional PDFs), chunks the content, embeds it, and stores vectors in **Qdrant** for semantic search. A FastAPI service exposes **natural-language search** and returns **top‑5 reranked passages with citations**.

---

## 🚀 Quick Start

### 1) Clone & set up environment
```powershell
git clone https://github.com/Lutherpick/Biomedical-RAG-Agent.git
cd Biomedical-RAG-Agent

# Create virtual env (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# (optional) keep pip fresh
python -m pip install --upgrade pip
```
> macOS/Linux users can use `python3 -m venv .venv && source .venv/bin/activate`.

### 2) Configure NCBI (higher request limits)
Create an API key at https://www.ncbi.nlm.nih.gov/account/ and set it:
```powershell
setx NCBI_API_KEY "your_api_key_here"
```
*(Restart your shell so the env var is available to scripts.)*

---

## 🧱 Data Pipeline

Run the following in order. Each step builds on the previous outputs.

```powershell
# (1) Fetch PubMed search results (XML + JSONL)
python data_pipeline/fetch_pubmed.py
# -> pubmed_export/pubmed_first_1000.xml
# -> pubmed_export/pubmed_first_1000.jsonl

# (2) Map PubMed IDs -> PMC IDs
python data_pipeline/pmid_to_pmcid.py
# -> pubmed_export/pmid_pmcid_map.json

# (3) Download PMC full‑text XMLs
python data_pipeline/download_pmc_xml.py
# -> data/raw/pmc_xml/*.xml

# (4) (Optional) Try downloading PDFs
python data_pipeline/download_pdfs.py
# -> data/raw/pdfs/*.pdf  (not all articles have PDFs; that’s fine)

# (5) (Optional) Convert PDFs to text
python data_pipeline/pdf_to_text.py
# -> data/processed/pdf_text/*.txt

# (6) Chunk XML/TXT into ~600‑token overlapping chunks
python data_pipeline/chunker.py
# -> data/processed/chunks/*.json
```

---

## 🗃️ Start Qdrant (Vector DB)

Use Docker with a persistent volume so data survives restarts:
```powershell
# Requires Docker Desktop running
docker rm -f qdrant 2>$null
docker run --name qdrant -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant:latest
# Dashboard: http://localhost:6333/dashboard
```

**Upload chunks to Qdrant:**
```powershell
python data_pipeline/uploader_qdrant.py
# -> "Done. Points in collection: <N>"  (N should be > 0)
```

---

## 🧪 Sprint 2 — Semantic Retrieval API (User Stories 3 & 4)

### What’s included
- **User Story 3**: Natural‑language semantic search  
  ✅ Same embedding model for queries & docs (`sentence-transformers/all-MiniLM-L6-v2`)  
  ✅ FastAPI endpoint `/search`  
  ✅ Top‑k ranking by cosine (Qdrant)  
  ✅ Logging + simple visualization

- **User Story 4**: Trustable evidence with citations  
  ✅ Retriever + **Cross‑Encoder** reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`)  
  ✅ **Top‑5** passages returned with citation metadata (`pmcid`, `url`, `doc_id`, `chunk_index`)  
  ✅ Structured JSON response  
  ✅ Validation on **10 queries**

### Run the API
```powershell
# From repo root, with .venv activated
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
# Health:  http://127.0.0.1:8000/
# Docs:    http://127.0.0.1:8000/docs
```

**Environment overrides via `.env` (optional):**
```
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=biomed_chunks
EMB_MODEL=sentence-transformers/all-MiniLM-L6-v2
RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
LOG_DIR=logs
```

### Reranker dependency (PyTorch CPU)
```powershell
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install --upgrade sentence-transformers transformers
```

### Example requests

**Default usage (k=5):**
```powershell
Invoke-RestMethod http://127.0.0.1:8000/search -Method POST -ContentType "application/json" `
  -Body (@{ query="glioblastoma treatment with temozolomide"; k=5 } | ConvertTo-Json)
```

**Improve recall before rerank (retrieve 30 → rerank → top‑5):**
```powershell
Invoke-RestMethod http://127.0.0.1:8000/search -Method POST -ContentType "application/json" `
  -Body (@{ query="preclinical mouse model of Parkinson's disease"; k=5; n_retrieve=30 } | ConvertTo-Json)
```

**Filter to a single document (optional):**
```powershell
Invoke-RestMethod http://127.0.0.1:8000/search -Method POST -ContentType "application/json" `
  -Body (@{ query="dopamine neurons degeneration"; k=5; n_retrieve=30; doc_id="PMC11275912_0" } | ConvertTo-Json)
```

**Response shape (trimmed):**
```json
{
  "query": "...",
  "k": 5,
  "passages": [
    {
      "retriever_score": 0.47,
      "rerank_score": 5.12,
      "text": "Parkinson’s disease ...",
      "doc_id": "PMC11527254_0",
      "chunk_index": 0,
      "chunk_total": null,
      "citation": {
        "pmcid": "PMC11527254",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11527254/",
        "doc_id": "PMC11527254_0",
        "chunk_index": 0
      }
    }
  ]
}
```

---

## 🧾 Logs & Visualization

Every `/search` request appends one JSON line to `logs/search_log.jsonl` with a timestamp and small hit sample.

Simple analysis chart:
```powershell
python scripts/analyze_search_log.py
```
This prints the top `doc_id`s and opens a bar chart of recent results.

Validation set (n=10 queries):
```powershell
python scripts/validate_rerank.py
# Expect: 10/10 queries succeeded.
```

---

## 🛠️ Troubleshooting

- **Connection refused to Qdrant (WinError 10061):** Start Docker Desktop and run the `docker run ...` command above.  
- **404 “Collection doesn’t exist”:** Run `python data_pipeline/uploader_qdrant.py` to create/populate `biomed_chunks`.  
- **500 “Reranker failed”:** Install Torch CPU and upgrade `sentence-transformers/transformers`, then restart the API.  
- **Port already in use:** change `--port 8000` (API) or Docker’s `-p 6333:6333` to a free port.  
- **IntelliJ IDEA:** create a run config with `Script: uvicorn` and `Params: api.main:app --host 127.0.0.1 --port 8000 --reload`.

---

## 🗂️ Project Layout (key paths)
```
data_pipeline/
  fetch_pubmed.py         # PubMed search/export
  pmid_to_pmcid.py        # Map PMIDs -> PMCID
  download_pmc_xml.py     # Get PMC XML
  download_pdfs.py        # (optional) PDFs
  pdf_to_text.py          # (optional) PDF -> text
  chunker.py              # XML/TXT -> JSON chunks (~600 tokens, 10% overlap)
  uploader_qdrant.py      # Embed (MiniLM-L6-v2) + upload to Qdrant (biomed_chunks)

api/
  main.py                 # FastAPI: / (health), /search (reranked passages with citations)

scripts/
  analyze_search_log.py   # Simple bar chart from logs
  validate_rerank.py      # 10-query validation

logs/
  search_log.jsonl        # One JSON per search (append-only)
```

---

## 📏 Definition of Done — Sprint 2

- **User Story 3**: Query encoder = doc model ✅ • FastAPI `/search` ✅ • Top‑k cosine ✅ • Logs + viz ✅  
- **User Story 4**: Retriever + reranker ✅ • Top‑5 with citations ✅ • Structured JSON ✅ • 10/10 validation ✅

---

Happy searching! 🧪🔎
