#!/usr/bin/env python3
r"""
4.10 metrics – normalized labels (all topics).

Computes:
- Precision@K and Recall@K over all queries
- Per-label metrics
- Optional groundedness score via LLM-as-judge (0–3)
- Manual spot-check sample
- Logs software versions + input checksums to pipeline_log_norm.json
- Writes evaluation_summary_norm.csv and metrics_report_norm.json

Assumed directory layout (run from anywhere; paths are resolved relative to this file):

out/
  4_5_labeling/
    paragraph_chunks_4000_labeled_ranked.jsonl   # (not mandatory; only needed if you later want text)
  4_8_eval_setup/
    queries_eval_norm.jsonl
  4_9_retrieval/
    retrieval_results.jsonl
  4_10_metrics/
    evaluate_metrics_norm.py                     # <- this file (HERE)

Input schema (minimal):

queries_eval_norm.jsonl (one per query, normalized labels):
  - query_id : int or str
  - query    : str (or 'question' / 'text')
  - label    : str (or 'cluster_label' / 'target_label')

retrieval_results.jsonl:
  - query_id          : matches queries
  - retrieved_labels  : list[str] (normalized labels, length K)
  - retrieved_chunk_ids : list[str] (optional, used only for spot-check convenience)

Optional groundedness answers file (only if you run with --answers):
answers.jsonl:
  - query_id : matches queries
  - query    : str (or 'question' / 'text')
  - answer   : str
  - contexts : list[str] (or 'context' / 'retrieved_texts')

Groundedness:
  0 = Ungrounded
  1 = Weakly grounded
  2 = Partially grounded
  3 = Fully grounded

You can run without groundedness (no OpenAI dependency) – it is optional.

Example (from repo root, PowerShell):

  python .\src\pmc_chunker\out\4_10_metrics\evaluate_metrics_norm.py

With groundedness on a small sample (e.g. 12 queries) and an answers file:

  python .\src\pmc_chunker\out\4_10_metrics\evaluate_metrics_norm.py `
    --answers .\src\pmc_chunker\out\4_10_metrics\answers_sample.jsonl `
    --groundedness-samples 12 `
    --openai-model gpt-5.1
"""

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


HERE = Path(__file__).resolve().parent
OUT_DEFAULT = HERE  # out/4_10_metrics
K_DEFAULT = 5
GROUND_SAMPLES_DEFAULT = 0  # 0 => skip LLM-as-judge


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def detect_column(df: pd.DataFrame, candidates: List[str], name: str) -> str:
    """Pick the first column from candidates that exists in df."""
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find {name} column; tried: {candidates}")


def sha256_of(path: Path) -> str:
    """Compute SHA256 checksum of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_list(obj: Any) -> List[Any]:
    """Ensure obj is a Python list; decode JSON strings if needed."""
    if isinstance(obj, list):
        return obj
    if obj is None:
        return []
    if isinstance(obj, str):
        obj = obj.strip()
        if not obj:
            return []
        try:
            parsed = json.loads(obj)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            # treat as single-item list
            return [obj]
    return [obj]


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def compute_retrieval_metrics(
        df: pd.DataFrame,
        label_col: str,
        k: int,
) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]:
    """
    Compute precision@k and recall@k for each query and aggregate per label.

    Recall@k is defined here as "hit rate":
      1 if at least one relevant label appears in the top-k retrieved labels,
      0 otherwise.
    """

    def _row_metrics(row: pd.Series) -> pd.Series:
        gold = row[label_col]
        retrieved = ensure_list(row["retrieved_labels"])[:k]
        hits = sum(1 for lbl in retrieved if lbl == gold)
        precision = hits / float(k) if k > 0 else 0.0
        recall = 1.0 if hits > 0 else 0.0
        return pd.Series(
            {
                "hits_at_k": hits,
                "precision_at_k": precision,
                "recall_at_k": recall,
            }
        )

    m = df.apply(_row_metrics, axis=1)
    df_with = pd.concat([df.reset_index(drop=True), m], axis=1)

    macro_precision = float(df_with["precision_at_k"].mean())
    macro_recall = float(df_with["recall_at_k"].mean())

    per_label = (
        df_with.groupby(label_col)
        .agg(
            n_queries=("query_id", "count"),
            precision_at_k=("precision_at_k", "mean"),
            recall_at_k=("recall_at_k", "mean"),
        )
        .reset_index()
        .rename(columns={label_col: "label"})
        .sort_values("n_queries", ascending=False)
    )

    per_label["precision_at_k"] = per_label["precision_at_k"].astype(float)
    per_label["recall_at_k"] = per_label["recall_at_k"].astype(float)

    metrics = {
        "k": int(k),
        "n_queries": int(len(df_with)),
        "macro_precision_at_k": macro_precision,
        "macro_recall_at_k": macro_recall,
        "per_label": [
            {
                "label": str(row["label"]),
                "n_queries": int(row["n_queries"]),
                "precision_at_k": float(row["precision_at_k"]),
                "recall_at_k": float(row["recall_at_k"]),
            }
            for _, row in per_label.iterrows()
        ],
    }

    return df_with, metrics, per_label


# ---------------------------------------------------------------------------
# Groundedness via LLM-as-judge (optional)
# ---------------------------------------------------------------------------

def evaluate_groundedness_llm(
        answers_path: Path,
        model: str,
        n_samples: int,
        seed: int = 2025,
) -> Optional[Dict[str, Any]]:
    """
    Optional LLM-as-judge groundedness evaluation.

    Expects answers_path JSONL with at least:
      - query_id
      - query (or question/text)
      - answer
      - contexts (list[str]) OR 'context' OR 'retrieved_texts'

    Returns dict with:
      - n_samples
      - mean_score
      - scores: list of {query_id, score, raw_judge}
    """
    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        print(
            "[WARN] openai package not installed; skipping groundedness.",
            file=sys.stderr,
        )
        return None

    df = pd.read_json(answers_path, lines=True)
    if df.empty:
        print("[INFO] answers file is empty; skipping groundedness.", file=sys.stderr)
        return None

    q_col = detect_column(df, ["query", "question", "text"], "question")

    if "answer" not in df.columns:
        raise ValueError("answers file must contain an 'answer' column.")
    ctx_col = detect_column(
        df, ["contexts", "context", "retrieved_texts"], "contexts"
    )

    if n_samples <= 0:
        print("[INFO] groundedness-samples <= 0; skipping LLM-as-judge.", file=sys.stderr)
        return None

    n = min(int(n_samples), len(df))
    df_sample = df.sample(n=n, random_state=seed)

    client = OpenAI()
    system_instructions = (
        "You are a strict evaluator of factual grounding for retrieval-augmented answers.\n"
        "Given a question, a set of retrieved context snippets, and an answer, "
        "judge how well the answer is supported by the context.\n"
        "Use the following scale:\n"
        "0 = Ungrounded (answer unrelated or contradicts context)\n"
        "1 = Weakly grounded (topic related but key claims not clearly supported)\n"
        "2 = Partially grounded (most key claims supported; some extrapolation)\n"
        "3 = Fully grounded (all claims directly supported by the context)\n"
        "Output FIRST the numeric score (0, 1, 2, or 3), then a very short justification."
    )

    scores: List[Dict[str, Any]] = []
    numeric_scores: List[int] = []

    for _, row in df_sample.iterrows():
        question = str(row[q_col])
        answer = str(row["answer"])
        contexts = ensure_list(row[ctx_col])
        context_block = "\n\n".join(f"- {c}" for c in contexts)

        prompt = (
            f"Question:\n{question}\n\n"
            f"Retrieved context snippets:\n{context_block}\n\n"
            f"Answer to evaluate:\n{answer}\n\n"
            "Now provide your groundedness score (0–3) and a very brief justification."
        )

        try:
            response = client.responses.create(
                model=model,
                instructions=system_instructions,
                input=prompt,
            )
            raw = str(response.output_text).strip()
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] LLM call failed for query_id={row.get('query_id')}: {e}", file=sys.stderr)
            continue

        m = re.search(r"\b([0-3])\b", raw)
        if not m:
            print(
                f"[WARN] Could not parse groundedness score from: {raw!r}",
                file=sys.stderr,
            )
            continue

        score = int(m.group(1))
        numeric_scores.append(score)
        scores.append(
            {
                "query_id": row.get("query_id"),
                "score": score,
                "judge_output": raw,
            }
        )

    if not numeric_scores:
        print("[WARN] No groundedness scores produced; skipping.", file=sys.stderr)
        return None

    mean_score = float(sum(numeric_scores) / len(numeric_scores))
    return {
        "n_samples": len(numeric_scores),
        "mean_score": mean_score,
        "scale": {
            "0": "Ungrounded",
            "1": "Weakly grounded",
            "2": "Partially grounded",
            "3": "Fully grounded",
        },
        "scores": scores,
    }


# ---------------------------------------------------------------------------
# Manual spot checks
# ---------------------------------------------------------------------------

def create_manual_spot_checks(
        df: pd.DataFrame,
        label_col: str,
        query_col: str,
        out_path: Path,
        n_spots: int = 15,
        seed: int = 42,
) -> Dict[str, Any]:
    """
    Select a small random subset of queries for manual spot-checks and
    write them to JSONL for easy inspection.
    """
    if df.empty or n_spots <= 0:
        return {"n": 0, "path": None, "query_ids": []}

    n = min(int(n_spots), len(df))
    sample = df.sample(n=n, random_state=seed)

    records = []
    for _, row in sample.iterrows():
        rec = {
            "query_id": row["query_id"],
            "query": row[query_col],
            "label": row[label_col],
            "retrieved_labels": ensure_list(row["retrieved_labels"]),
            "hits_at_k": row.get("hits_at_k"),
            "precision_at_k": row.get("precision_at_k"),
            "recall_at_k": row.get("recall_at_k"),
        }
        if "retrieved_chunk_ids" in row:
            rec["retrieved_chunk_ids"] = ensure_list(row["retrieved_chunk_ids"])
        records.append(rec)

    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return {
        "n": n,
        "path": str(out_path),
        "query_ids": [r["query_id"] for r in records],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="4.10 metrics (normalized labels, all topics)."
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=None,
        help="Path to queries_eval_norm.jsonl (normalized). "
             "If omitted, resolve relative to this file.",
    )
    parser.add_argument(
        "--retrieval",
        type=Path,
        default=None,
        help="Path to retrieval_results.jsonl. "
             "If omitted, resolve relative to this file.",
    )
    parser.add_argument(
        "--answers",
        type=Path,
        default=None,
        help="Optional: JSONL with answers+contexts for groundedness.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for metrics_report_norm.json, "
             "evaluation_summary_norm.csv, pipeline_log_norm.json. "
             "Defaults to this script's directory.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=K_DEFAULT,
        help=f"Rank cutoff K for precision@K / recall@K (default: {K_DEFAULT}).",
    )
    parser.add_argument(
        "--groundedness-samples",
        type=int,
        default=GROUND_SAMPLES_DEFAULT,
        help=(
            "Number of queries to sample for LLM-as-judge groundedness. "
            "0 = skip groundedness (default)."
        ),
    )
    parser.add_argument(
        "--openai-model",
        type=str,
        default="gpt-5.1",
        help="Model name for groundedness LLM-as-judge (default: gpt-5.1).",
    )
    parser.add_argument(
        "--spot-checks",
        type=int,
        default=15,
        help="Number of queries to export for manual spot-checks (default: 15).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve defaults relative to this file.
    if args.out_dir is None:
        args.out_dir = OUT_DEFAULT
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.queries is None:
        args.queries = HERE.parent / "4_8_eval_setup" / "queries_eval_norm.jsonl"
    if args.retrieval is None:
        args.retrieval = HERE.parent / "4_9_retrieval" / "retrieval_results.jsonl"

    if not args.queries.is_file():
        raise FileNotFoundError(f"queries file not found: {args.queries}")
    if not args.retrieval.is_file():
        raise FileNotFoundError(f"retrieval file not found: {args.retrieval}")

    # Load inputs
    queries_df = pd.read_json(args.queries, lines=True)
    retrieval_df = pd.read_json(args.retrieval, lines=True)

    if queries_df.empty:
        raise RuntimeError("queries_eval_norm is empty.")
    if retrieval_df.empty:
        raise RuntimeError("retrieval_results is empty.")

    label_col = detect_column(
        queries_df, ["label", "cluster_label", "target_label", "normalized_label"], "label"
    )
    query_col = detect_column(queries_df, ["query", "query_text", "question", "text"], "query")


    # Merge on query_id
    if "query_id" not in queries_df.columns or "query_id" not in retrieval_df.columns:
        raise ValueError("Both queries and retrieval files must contain 'query_id' column.")

    merged = queries_df.merge(
        retrieval_df,
        on="query_id",
        how="inner",
        suffixes=("_q", "_r"),
    )

    if merged.empty:
        raise RuntimeError("No overlapping query_id between queries and retrieval results.")

    # Compute retrieval metrics
    df_with_metrics, retrieval_metrics, per_label_df = compute_retrieval_metrics(
        merged, label_col=label_col, k=args.k
    )

    # Manual spot checks
    spot_path = args.out_dir / "manual_spot_checks_norm.jsonl"
    spot_info = create_manual_spot_checks(
        df_with_metrics,
        label_col=label_col,
        query_col=query_col,
        out_path=spot_path,
        n_spots=args.spot_checks,
    )

    # Optional groundedness
    groundedness_report: Optional[Dict[str, Any]] = None
    if args.answers is not None and args.groundedness_samples > 0:
        if not args.answers.is_file():
            raise FileNotFoundError(f"answers file not found: {args.answers}")
        groundedness_report = evaluate_groundedness_llm(
            answers_path=args.answers,
            model=args.openai_model,
            n_samples=args.groundedness_samples,
        )

    # Build metrics_report_norm.json
    metrics_report = {
        "retrieval": retrieval_metrics,
        "groundedness": groundedness_report,
    }

    metrics_path = args.out_dir / "metrics_report_norm.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_report, f, indent=2, ensure_ascii=False)

    # Build evaluation_summary_norm.csv
    summary_df = per_label_df.copy()
    summary_df["is_macro_row"] = False

    macro_row = pd.DataFrame(
        [
            {
                "label": "__ALL__",
                "n_queries": retrieval_metrics["n_queries"],
                "precision_at_k": retrieval_metrics["macro_precision_at_k"],
                "recall_at_k": retrieval_metrics["macro_recall_at_k"],
                "is_macro_row": True,
            }
        ]
    )

    summary_df = pd.concat([summary_df, macro_row], ignore_index=True)
    summary_path = args.out_dir / "evaluation_summary_norm.csv"
    summary_df.to_csv(summary_path, index=False)

    # Build pipeline_log_norm.json
    pipeline_log = {
        "run_at": dt.datetime.now().isoformat(),
        "k": int(args.k),
        "paths": {
            "queries_eval_norm": str(args.queries),
            "retrieval_results": str(args.retrieval),
            "metrics_report_norm": str(metrics_path),
            "evaluation_summary_norm": str(summary_path),
            "manual_spot_checks_norm": str(spot_path),
            "answers": str(args.answers) if args.answers is not None else None,
        },
        "input_checksums": {
            "queries_eval_norm_sha256": sha256_of(args.queries),
            "retrieval_results_sha256": sha256_of(args.retrieval),
            **(
                {"answers_sha256": sha256_of(args.answers)}
                if args.answers is not None and args.answers.is_file()
                else {}
            ),
        },
        "software": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "pandas_version": pd.__version__,
        },
        "spot_checks": spot_info,
        "groundedness": groundedness_report,
        "env": {
            "cwd": os.getcwd(),
        },
    }

    log_path = args.out_dir / "pipeline_log_norm.json"
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(pipeline_log, f, indent=2, ensure_ascii=False)

    # Short console summary
    print(f"[INFO] K={retrieval_metrics['k']} | n_queries={retrieval_metrics['n_queries']}")
    print(
        f"[INFO] P@{args.k} (macro) = {retrieval_metrics['macro_precision_at_k']:.3f} | "
        f"R@{args.k} (macro hit-rate) = {retrieval_metrics['macro_recall_at_k']:.3f}"
    )
    if groundedness_report is not None:
        print(
            f"[INFO] Groundedness mean score (0–3) over "
            f"{groundedness_report['n_samples']} samples "
            f"= {groundedness_report['mean_score']:.3f}"
        )
    print(f"[INFO] metrics_report_norm.json      -> {metrics_path}")
    print(f"[INFO] evaluation_summary_norm.csv   -> {summary_path}")
    print(f"[INFO] pipeline_log_norm.json        -> {log_path}")
    print(f"[INFO] manual_spot_checks_norm.jsonl -> {spot_path}")


if __name__ == "__main__":
    main()
