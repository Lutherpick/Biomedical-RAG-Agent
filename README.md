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
