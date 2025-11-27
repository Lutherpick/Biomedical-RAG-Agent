# PMC XML Paragraph Chunking Pipeline

This repository builds a ~4 000-article corpus of open-access preclinical animal studies, downloads their full text from PubMed Central (PMC), and converts each article into paragraph-level JSONL “chunks” suitable for semantic search and RAG systems.

The pipeline has four main stages:

1. Build a manifest of 4 000 PMC articles (`build_manifest.py`).
2. Download JATS/NIHMS XML full texts for those PMCIDs (`download_xmls.py`).
3. (Optional) Inspect paragraph order for a single article (`jats_probe.py`).
4. Convert all XMLs into paragraph-level chunks (`parse_chunk.py`), using shared cleaners and splitters from `chunker.py`.

---

## 1. Requirements

Python 3.9+ is recommended.

Install the required third-party packages:

```bash
pip install pandas requests tqdm tenacity lxml python-dateutil python-dotenv

These are used for PubMed/PMC HTTP requests, CSV handling, retry logic, XML parsing, and incremental metadata summarization.

Environment variables (optional but recommended)

build_manifest.py can use an NCBI API key to increase E-utilities rate limits.
Create a .env file in the project root, e.g.:

NCBI_API_KEY=your_ncbi_key_here


The scripts automatically load this via python-dotenv.


Directory Layout

All scripts assume you run them from the repository root. They will create and use:

pmc_chunker/
  data/
    oa_file_list.csv.gz      # cached PMC OA master list (downloaded automatically)
    xml/                     # downloaded XML files (one per PMCID)
  out/
    manifest_4000.csv        # list of selected articles (≈4 000 rows)
    paragraph_chunks_4000.jsonl
                             # paragraph-level chunks for all XMLs


You do not need to create these directories manually.

How to Run the Pipeline (Step by Step)

All commands below are run from the repository root.

Step 1 – Build the 4 000-article manifest

This script selects ~4 000 open-access animal/preclinical research articles (2010–2025, English/German) from PMC, filters by publication type, and stratifies them into topic buckets.

python build_manifest.py


Output:

pmc_chunker/out/manifest_4000.csv


This CSV contains at least:

PMCID, PMID

doi, year

title, journal, topic

file (OA .tar.gz path), license

You normally run this only when you want to rebuild or update the corpus selection.

Step 2 – Download XML full texts from PMC

This script reads manifest_4000.csv and downloads the corresponding XML full texts.

python download_xmls.py


What it does:

For each PMCID in pmc_chunker/out/manifest_4000.csv:

Calls the PMC OA API to discover XML or .tgz packages.

Prefers direct XML links; if not available, downloads the .tgz, extracts the largest .xml/.nxml file, and saves it.

Skips PMCIDs whose XML has already been downloaded.

Output:

pmc_chunker/data/xml/PMCxxxxxxx.xml  # one file per PMCID


You can safely rerun this script; existing XML files are skipped.

Step 4 – Chunk all XMLs into paragraph-level JSONL

This is the main chunking step. It uses parse_chunk.py (main logic) and chunker.py (cleaning + splitting helpers).

Minimal command:

python parse_chunk.py \
  --xml-dir pmc_chunker/data/xml \
  --out pmc_chunker/out/paragraph_chunks_4000.jsonl


Recommended command with explicit guardrails:

python parse_chunk.py \
  --xml-dir pmc_chunker/data/xml \
  --out pmc_chunker/out/paragraph_chunks_4000.jsonl \
  --max-tokens 800 \
  --overlap-sentences 1


Options:

--xml-dir DIR
Directory containing .xml files (e.g. pmc_chunker/data/xml).

--xml PATTERN [PATTERN ...]
One or more glob patterns or file paths, e.g. --xml "pmc_chunker/data/xml/*.xml".

--out PATH
Output JSONL file. Each line is one chunk (paragraph or paragraph-split).

--max-tokens N (default: 800)
If a paragraph exceeds N tokens (approximate), it is split at sentence boundaries.

--overlap-sentences N (default: 0)
Number of sentences to overlap between consecutive chunks when splitting an overlong paragraph (e.g. 1).

--per-file
In addition to the merged output, also write one {PMCID}_out.jsonl next to each XML.

Output:

pmc_chunker/out/paragraph_chunks_4000.jsonl
pmc_chunker/data/xml/PMCxxxxxxx_out.jsonl   # only if --per-file is used


Each JSONL line contains fields such as:

Identity & metadata: pmcid, pmid, nihmsid, title, journal, year, source_file

Structure: section, subsection, section_index, subsection_index, figure_id

Text & type: text, type (e.g. abstract, introduction, figure_caption, method_paragraph, etc.)

Indices: para_local_index, split_index, chunk_index


4.3 – Embed paragraph chunks

Use the same chunks file that was produced by your paragraph chunker (paragraph_chunks_4000_merged.jsonl):

python .\src\embed_chunks.py `
  --inp .\src\pmc_chunker\out\paragraph_chunks_4000_merged.jsonl `
  --out .\src\pmc_chunker\out\embeddings_v5.npy `
  --batch 1024 `
  --fp16 `
  --normalize

4.4 – Clustering (MiniBatchKMeans, K≈700)
& .\.venv\Scripts\python.exe .\src\cluster.py `
  --chunks .\src\pmc_chunker\out\paragraph_chunks_4000_merged.jsonl `
  --out-dir .\src\pmc_chunker\out\4_4_clustering `
  --per-cluster-target 500 `
  --max-iter 300


Output of interest (4.4):
.\src\pmc_chunker\out\4_4_clustering\cluster_assignments_k700.csv

4.5 – Cluster labeling (LLM) → labeled chunks
python .\src\pmc_chunker\out\4_5_labeling\label_clusters.py `
  --chunks .\src\pmc_chunker\out\paragraph_chunks_4000_merged.jsonl `
  --assign .\src\pmc_chunker\out\4_4_clustering\cluster_assignments_k700.csv `
  --out-dir .\src\pmc_chunker\out\4_5_labeling `
  --llm openai `
  --openai-model gpt-5.1 `
  --openai-max-new 256


Key outputs (4.5):

.\src\pmc_chunker\out\4_5_labeling\paragraph_chunks_4000_labeled.jsonl

.\src\pmc_chunker\out\4_5_labeling\cluster_labels.json

.\src\pmc_chunker\out\4_5_labeling\cluster_exemplars.json

4.5b – Rank within cluster + centroid distance
python .\src\pmc_chunker\out\4_5_labeling\compute_rank_and_distance.py `
  --chunks .\src\pmc_chunker\out\4_5_labeling\paragraph_chunks_4000_labeled.jsonl `
  --assign .\src\pmc_chunker\out\4_4_clustering\cluster_assignments_k700.csv `
  --emb   .\src\pmc_chunker\out\embeddings_v5.npy `
  --out   .\src\pmc_chunker\out\4_5_labeling\paragraph_chunks_4000_labeled_ranked.jsonl


Output:

.\src\pmc_chunker\out\4_5_labeling\paragraph_chunks_4000_labeled_ranked.jsonl
(now with rank_in_cluster and centroid_distance)

4.6 – Query generation (label-based queries)
python .\src\generate_queries.py `
  --chunks .\src\pmc_chunker\out\4_5_labeling\paragraph_chunks_4000_labeled_ranked.jsonl `
  --out    .\src\pmc_chunker\out\4_6_query_generation\queries_gpt51.jsonl `
  --llm openai `
  --openai-model gpt-5.1 `
  --openai-max-new 64 `
  --min-per-cluster 1 `
  --max-per-cluster 1


Output:

.\src\pmc_chunker\out\4_6_query_generation\queries_gpt51.jsonl

4.7 – Query deduplication

Here you just run the dedup script in 4_7_deduplication which reads queries_gpt51.jsonl and writes queries_gpt51_dedup.jsonl.

If your script is called deduplicate_queries.py, the command is:

python .\src\pmc_chunker\out\4_7_deduplication\deduplicate_queries.py


Output (expected):

.\src\pmc_chunker\out\4_7_deduplication\queries_gpt51_dedup.jsonl

(If the script name in your repo differs, just replace deduplicate_queries.py accordingly.)

4.8 – Build evaluation set (per-cluster queries)

Run the eval-setup script under 4_8_eval_setup. It takes the deduped queries and creates queries_eval.jsonl (one query per cluster) plus config.

Example (adjust script name to match your repo):

python .\src\pmc_chunker\out\4_8_eval_setup\prepare_eval_dataset.py


Expected outputs:

.\src\pmc_chunker\out\4_8_eval_setup\queries_eval.jsonl

.\src\pmc_chunker\out\4_8_eval_setup\retrieval_config.json

.\src\pmc_chunker\out\4_8_eval_setup\qdrant_id_map.csv

4.8b – Normalize labels and propagate everywhere

This step:

Merges near-duplicate labels.

Normalizes labels in:

cluster_labels.json

paragraph_chunks_4000_labeled_ranked.jsonl

all query files (raw, dedup, eval).

Run:

python .\src\pmc_chunker\out\4_5_labeling\normalize_labels_and_propagate.py


Outputs (among others):

.\src\pmc_chunker\out\4_5_labeling\cluster_labels_merged.json

.\src\pmc_chunker\out\4_5_labeling\paragraph_chunks_4000_labeled_ranked_merged.jsonl

.\src\pmc_chunker\out\4_6_query_generation\queries_gpt51_norm.jsonl

.\src\pmc_chunker\out\4_7_deduplication\queries_gpt51_dedup_norm.jsonl

.\src\pmc_chunker\out\4_8_eval_setup\queries_eval_norm.jsonl

4.9 – Upload paragraphs to Qdrant

Now upload the normalized, ranked chunks + embeddings to Qdrant:

.\.venv\Scripts\python.exe .\src\qdrant_upload_paragraphs.py `
  --inp .\src\pmc_chunker\out\4_5_labeling\paragraph_chunks_4000_labeled_ranked_merged.jsonl `
  --emb .\src\pmc_chunker\out\embeddings_v5.npy `
  --collection biomed_paragraphs `
  --distance cosine `
  --recreate


(If the collection already exists and you don’t want to drop it, omit --recreate.)

4.9b – Run retrieval for the normalized eval queries

This uses queries_eval_norm.jsonl and writes retrieval_results.jsonl under 4_9_retrieval.

& .\.venv\Scripts\python.exe .\src\pmc_chunker\out\4_9_retrieval\run_retrieval.py `
  --backend openai `
  --openai-api-key $env:OPENAI_API_KEY


Expected output:

.\src\pmc_chunker\out\4_9_retrieval\retrieval_results.jsonl

4.10 – Compute metrics (normalized labels)

Macro P@5 / R@5 over all topics:

python .\src\pmc_chunker\out\4_10_metrics\evaluate_metrics_norm.py


Outputs:

.\src\pmc_chunker\out\4_10_metrics\metrics_report_norm.json

.\src\pmc_chunker\out\4_10_metrics\evaluation_summary_norm.csv

.\src\pmc_chunker\out\4_10_metrics\pipeline_log_norm.json

.\src\pmc_chunker\out\4_10_metrics\manual_spot_checks_norm.jsonl

4.10b – Metrics on frequent topics (min 5 queries per label)
python .\src\pmc_chunker\out\4_10_metrics\evaluate_metrics_norm_filtered.py


Outputs:

.\src\pmc_chunker\out\4_10_metrics\metrics_report_norm_min5.json

.\src\pmc_chunker\out\4_10_metrics\evaluation_summary_norm_min5.csv

.\src\pmc_chunker\out\4_10_metrics\pipeline_log_norm_min5.json
