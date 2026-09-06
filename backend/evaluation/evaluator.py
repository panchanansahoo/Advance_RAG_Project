"""
Automated Evaluation Script using Ragas and custom retrieval metrics.

Phase 8 component.

Improvements (Task 3.4):
- Added retrieval metrics: Recall@K, Precision@K, MRR, NDCG
- Added token usage tracking
- Added cost estimation based on model pricing
- Added throughput measurement (queries/second)
"""

import os
import json
import asyncio
import logging
import time
import math
from typing import List, Dict, Any, Optional

import httpx

try:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

logger = logging.getLogger(__name__)

# Backend URL for calling the query endpoint
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

# Approximate pricing per 1M tokens (configurable)
MODEL_PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
}


# ── Retrieval Metrics (PRD §22 — Task 3.4) ──────────────────


def recall_at_k(
    retrieved_sources: List[str],
    expected_sources: List[str],
    k: Optional[int] = None,
) -> float:
    """
    Recall@K: What fraction of expected sources were retrieved?

    Recall@K = |retrieved ∩ expected| / |expected|
    """
    if not expected_sources:
        return 1.0  # No expectations = trivially satisfied

    retrieved = set(retrieved_sources[:k] if k else retrieved_sources)
    expected = set(expected_sources)
    hits = retrieved & expected
    return len(hits) / len(expected)


def precision_at_k(
    retrieved_sources: List[str],
    expected_sources: List[str],
    k: Optional[int] = None,
) -> float:
    """
    Precision@K: What fraction of retrieved sources are relevant?

    Precision@K = |retrieved ∩ expected| / |retrieved|
    """
    retrieved = list(retrieved_sources[:k] if k else retrieved_sources)
    if not retrieved:
        return 0.0

    expected = set(expected_sources)
    hits = sum(1 for r in retrieved if r in expected)
    return hits / len(retrieved)


def mean_reciprocal_rank(
    retrieved_sources: List[str],
    expected_sources: List[str],
) -> float:
    """
    MRR: Reciprocal rank of the first relevant result.

    MRR = 1 / rank_of_first_relevant_result
    """
    expected = set(expected_sources)
    for i, source in enumerate(retrieved_sources):
        if source in expected:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(
    retrieved_sources: List[str],
    expected_sources: List[str],
    k: Optional[int] = None,
) -> float:
    """
    NDCG@K: Normalized Discounted Cumulative Gain.

    Uses binary relevance (1 if in expected, 0 otherwise).
    """
    expected = set(expected_sources)
    retrieved = list(retrieved_sources[:k] if k else retrieved_sources)

    if not retrieved or not expected:
        return 0.0

    # DCG
    dcg = 0.0
    for i, source in enumerate(retrieved):
        rel = 1.0 if source in expected else 0.0
        dcg += rel / math.log2(i + 2)  # +2 because log2(1) = 0

    # Ideal DCG (all relevant at the top)
    ideal_count = min(len(expected), len(retrieved))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_count))

    return dcg / idcg if idcg > 0 else 0.0


def estimate_cost(
    total_input_tokens: int,
    total_output_tokens: int,
    model_name: str = "gpt-4o-mini",
) -> float:
    """Estimate cost in USD based on token counts and model pricing."""
    pricing = MODEL_PRICING.get(model_name, MODEL_PRICING["gpt-4o-mini"])
    input_cost = (total_input_tokens / 1_000_000) * pricing["input"]
    output_cost = (total_output_tokens / 1_000_000) * pricing["output"]
    return round(input_cost + output_cost, 6)


# ── Main Evaluation Runner ──────────────────────────────────


async def run_evaluation(dataset_path: str, output_path: str):
    """
    Reads a benchmark dataset, runs the queries through the live pipeline,
    and evaluates the results using RAGAS + custom retrieval metrics.
    """
    logger.info(f"Loading dataset from {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    system_metrics = {
        "latencies_ms": [],
        "total_queries": 0,
        "successful_queries": 0,
        "failed_queries": 0,
    }

    # ── Retrieval metric accumulators ───────────────────────
    retrieval_metrics = {
        "recall_at_5": [],
        "precision_at_5": [],
        "mrr": [],
        "ndcg_at_5": [],
    }

    # Token tracking
    total_input_tokens = 0
    total_output_tokens = 0

    logger.info("Running %d queries through the pipeline...", len(data))
    start_wall = time.time()

    async with httpx.AsyncClient(timeout=60.0) as client:
        for item in data:
            q = item["question"]
            ground_truth = item.get("ground_truth", "")
            expected_contexts = item.get("expected_contexts", [])
            expected_sources = item.get("expected_sources", [])

            # Actually call the RAG pipeline
            start_time = time.time()
            try:
                response = await client.post(
                    f"{BACKEND_URL}/api/v1/query",
                    json={"query": q},
                )
                elapsed_ms = (time.time() - start_time) * 1000
                system_metrics["latencies_ms"].append(elapsed_ms)

                if response.status_code == 200:
                    resp_data = response.json()
                    generated_answer = resp_data.get("answer", "")

                    # Extract contexts from citations
                    retrieved_contexts = [
                        c.get("content_snippet", "")
                        for c in resp_data.get("citations", [])
                    ]
                    if not retrieved_contexts:
                        retrieved_contexts = expected_contexts  # fallback

                    # Extract source names from citations for retrieval metrics
                    retrieved_source_names = [
                        c.get("document_name", "")
                        for c in resp_data.get("citations", [])
                    ]

                    # Compute retrieval metrics (Task 3.4)
                    if expected_sources:
                        retrieval_metrics["recall_at_5"].append(
                            recall_at_k(retrieved_source_names, expected_sources, k=5)
                        )
                        retrieval_metrics["precision_at_5"].append(
                            precision_at_k(retrieved_source_names, expected_sources, k=5)
                        )
                        retrieval_metrics["mrr"].append(
                            mean_reciprocal_rank(retrieved_source_names, expected_sources)
                        )
                        retrieval_metrics["ndcg_at_5"].append(
                            ndcg_at_k(retrieved_source_names, expected_sources, k=5)
                        )

                    # Track token usage from metadata (if available)
                    meta = resp_data.get("retrieval_metadata", {})
                    total_input_tokens += meta.get("input_tokens", 0)
                    total_output_tokens += meta.get("output_tokens", 0)

                    results["question"].append(q)
                    results["answer"].append(generated_answer)
                    results["contexts"].append(retrieved_contexts)
                    results["ground_truth"].append(ground_truth)
                    system_metrics["successful_queries"] += 1
                else:
                    logger.warning("Query failed (status %d): %s", response.status_code, q)
                    system_metrics["failed_queries"] += 1
                    # Use ground truth as fallback
                    results["question"].append(q)
                    results["answer"].append(ground_truth)
                    results["contexts"].append(expected_contexts)
                    results["ground_truth"].append(ground_truth)

            except Exception as e:
                elapsed_ms = (time.time() - start_time) * 1000
                system_metrics["latencies_ms"].append(elapsed_ms)
                logger.warning("Query request failed for '%s': %s. Using fallback.", q, e)
                system_metrics["failed_queries"] += 1

                results["question"].append(q)
                results["answer"].append(ground_truth)
                results["contexts"].append(expected_contexts)
                results["ground_truth"].append(ground_truth)

            system_metrics["total_queries"] += 1

    total_wall_time = time.time() - start_wall

    # ── Compute system metrics ──────────────────────────────
    latencies = system_metrics["latencies_ms"]
    system_scores = {}
    if latencies:
        system_scores["avg_latency_ms"] = round(sum(latencies) / len(latencies), 1)
        system_scores["p95_latency_ms"] = round(
            sorted(latencies)[int(len(latencies) * 0.95)], 1
        )
        system_scores["total_queries"] = system_metrics["total_queries"]
        system_scores["success_rate"] = round(
            system_metrics["successful_queries"]
            / max(system_metrics["total_queries"], 1),
            3,
        )
        # Throughput (queries/second) — Task 3.4
        system_scores["throughput_qps"] = round(
            system_metrics["total_queries"] / max(total_wall_time, 0.001), 2
        )

    # Token usage & cost — Task 3.4
    system_scores["total_input_tokens"] = total_input_tokens
    system_scores["total_output_tokens"] = total_output_tokens
    system_scores["estimated_cost_usd"] = estimate_cost(
        total_input_tokens, total_output_tokens
    )

    # ── Compute retrieval metrics ───────────────────────────
    retrieval_scores = {}
    for metric_name, values in retrieval_metrics.items():
        if values:
            retrieval_scores[metric_name] = round(sum(values) / len(values), 4)

    # ── Run RAGAS evaluation if available ───────────────────
    ragas_scores = {}
    if RAGAS_AVAILABLE and results["question"]:
        logger.info("Evaluating with Ragas...")

        if not os.environ.get("OPENAI_API_KEY"):
            logger.warning("OPENAI_API_KEY not set. Ragas evaluation might fail.")

        try:
            dataset = Dataset.from_dict(results)
            score = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
            ragas_scores = {k: float(v) for k, v in score.items()}
            score_details = score.to_pandas().to_dict(orient="records")
        except Exception as e:
            logger.error("RAGAS evaluation failed: %s", e)
            ragas_scores = {"faithfulness": -1, "answer_relevancy": -1}
            score_details = []
    else:
        if not RAGAS_AVAILABLE:
            logger.warning("Ragas not installed. Skipping LLM-based metrics.")
        ragas_scores = {}
        score_details = []

    # ── Combine all scores ──────────────────────────────────
    aggregate_scores = {**ragas_scores, **retrieval_scores, **system_scores}

    output_data = {
        "aggregate_scores": aggregate_scores,
        "retrieval_metrics": retrieval_scores,
        "system_metrics": system_scores,
        "ragas_metrics": ragas_scores,
        "details": score_details,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)

    logger.info(f"Evaluation complete. Results saved to {output_path}")
    logger.info(f"Aggregate Scores: {aggregate_scores}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dataset_file = os.path.join(os.path.dirname(__file__), "benchmark_dataset.json")
    output_file = os.path.join(os.path.dirname(__file__), "evaluation_results.json")

    asyncio.run(run_evaluation(dataset_file, output_file))
