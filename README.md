# 🧬 Biomedical-RAG-Agent

An end-to-end **Retrieval-Augmented Generation (RAG)** pipeline for biomedical literature.  
It automatically fetches biomedical papers from PubMed/PMC, extracts full text (XML + optional PDFs), chunks the content, generates embeddings, and stores them in **Qdrant** for semantic search and later RAG applications.

---

## 🚀 Quick Start Guide

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Lutherpick/Biomedical-RAG-Agent.git
cd Biomedical-RAG-Agent
2️⃣ Create a Virtual Environment
On Windows (PowerShell)
bash
Copy code
python -m venv .venv
.venv\Scripts\Activate.ps1
On macOS / Linux
bash
Copy code
python3 -m venv .venv
source .venv/bin/activate
3️⃣ Install Dependencies
bash
Copy code
pip install -r requirements.txt
If pip is outdated:

bash
Copy code
python -m pip install --upgrade pip
4️⃣ Configure Your NCBI Account
NCBI requires your email and optionally an API key for higher request limits.

Log in or create an account at https://www.ncbi.nlm.nih.gov/account/

Go to API Key Management and create an API key.

Then set your environment variable:

Windows (PowerShell)
bash
Copy code
setx NCBI_API_KEY "your_api_key_here"
macOS / Linux
bash
Copy code
export NCBI_API_KEY="your_api_key_here"
5️⃣ Run the Data Pipeline
Each script builds upon the previous one.
You can run them step-by-step or automate them later.

(1) Fetch PubMed Articles
bash
Copy code
python data_pipeline/fetch_pubmed.py
Outputs:

pubmed_export/pubmed_first_1000.xml

pubmed_export/pubmed_first_1000.jsonl

(2) Map PubMed → PMC IDs
bash
Copy code
python data_pipeline/pmid_to_pmcid.py
Output: pubmed_export/pmid_pmcid_map.json

(3) Download PMC Full-Text XML
bash
Copy code
python data_pipeline/download_pmc_xml.py
✅ Structured XML files saved to data/raw/pmc_xml/

(4) (Optional) Try to Download PDFs
bash
Copy code
python data_pipeline/download_pdfs.py
⚠️ Not all PMC entries have PDFs — this is expected.
Missing ones are logged automatically.

(5) Convert PDFs to Text (optional)
bash
Copy code
python data_pipeline/pdf_to_text.py
(6) Chunk Texts for Embedding
bash
Copy code
python data_pipeline/chunker.py
→ Output: data/processed/chunks/*.json

6️⃣ Run Qdrant for Vector Storage
Make sure Docker Desktop or WSL2 is running, then start Qdrant:

bash
Copy code
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest
✅ Dashboard: http://localhost:6333/dashboard

7️⃣ Upload Chunks to Qdrant
bash
Copy code
python data_pipeline/uploader_qdrant.py
Embeds text chunks using SentenceTransformer (all-MiniLM-L6-v2)
and uploads them to the Qdrant collection biomed_chunks.

8️⃣ Test Semantic Search
bash
Copy code
pytest data_pipeline/test_search.py -q
Confirms the embedding + retrieval pipeline works.

🧩 Project Structure
pgsql
Copy code
biomed-rag-agent/
│
├── data_pipeline/
│   ├── fetch_pubmed.py           → Fetches PubMed metadata
│   ├── pmid_to_pmcid.py          → Maps PubMed → PMC
│   ├── download_pmc_xml.py       → Downloads full text XML
│   ├── download_pdfs.py          → Attempts PDF download
│   ├── pdf_to_text.py            → Converts PDFs to text
│   ├── chunker.py                → Splits text into chunks
│   ├── uploader_qdrant.py        → Uploads embeddings
│   └── test_search.py            → Semantic retrieval test
│
├── pubmed_export/                → Metadata + mappings
├── data/raw/                     → Raw PMC XML + PDFs
├── data/processed/               → Chunked text JSON
├── requirements.txt
└── README.md
⚙️ Useful Commands
