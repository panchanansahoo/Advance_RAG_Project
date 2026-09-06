"""
Evaluation API — endpoints for triggering benchmarks and fetching results.

PRD §22-24.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/evaluation", tags=["Evaluation"])

EVAL_DIR = Path(__file__).resolve().parent.parent / "evaluation"
RESULTS_FILE = EVAL_DIR / "evaluation_results.json"
DATASET_FILE = EVAL_DIR / "benchmark_dataset.json"
EXPERIMENT_JSON_FILE = EVAL_DIR / "experiment_comparison.json"
EXPERIMENT_MD_FILE = EVAL_DIR / "experiment_comparison.md"


class EvaluationResultsResponse(BaseModel):
    """Response model for evaluation results."""
    aggregate_scores: Dict[str, float] = Field(default_factory=dict)
    details: List[Dict[str, Any]] = Field(default_factory=list)
    status: str = "complete"


class EvaluationRunResponse(BaseModel):
    """Response when triggering an evaluation run."""
    message: str
    status: str = "started"


@router.get("/results", response_model=EvaluationResultsResponse)
async def get_evaluation_results():
    """
    Fetch the latest evaluation results.
    Returns the most recent benchmark run results, or defaults if no run exists yet.
    """
    if RESULTS_FILE.exists():
        try:
            with open(RESULTS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return EvaluationResultsResponse(
                aggregate_scores=data.get("aggregate_scores", {}),
                details=data.get("details", []),
                status="complete",
            )
        except Exception as e:
            logger.error("Failed to read evaluation results: %s", e)
            raise HTTPException(status_code=500, detail="Failed to read evaluation results")
    else:
        # No results yet — return placeholder indicating no run
        return EvaluationResultsResponse(
            aggregate_scores={},
            details=[],
            status="no_results",
        )


@router.post("/run", response_model=EvaluationRunResponse)
async def trigger_evaluation(background_tasks: BackgroundTasks):
    """
    Trigger a benchmark evaluation run in the background.
    Results will be saved and accessible via GET /results.
    """
    if not DATASET_FILE.exists():
        raise HTTPException(status_code=404, detail="Benchmark dataset not found")

    background_tasks.add_task(_run_evaluation_task)
    return EvaluationRunResponse(
        message="Evaluation started in background. Poll GET /api/v1/evaluation/results for results.",
        status="started",
    )


@router.get("/experiments")
async def get_experiment_comparison():
    """Fetch the latest multi-architecture experiment comparison report."""
    if EXPERIMENT_JSON_FILE.exists():
        try:
            with open(EXPERIMENT_JSON_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error("Failed to read experiment comparison: %s", e)
            raise HTTPException(status_code=500, detail="Failed to read experiment report")
    return {
        "status": "no_results",
        "message": "No experiment comparison runs yet. Trigger POST /api/v1/evaluation/experiments/run.",
    }


@router.post("/experiments/run")
async def trigger_experiment_comparison(background_tasks: BackgroundTasks):
    """Trigger a multi-architecture comparison run across all PRD §23 presets in the background."""
    if not DATASET_FILE.exists():
        raise HTTPException(status_code=404, detail="Benchmark dataset not found")

    background_tasks.add_task(_run_experiments_task)
    return {
        "message": "Experiment comparison run started in background. Poll GET /api/v1/evaluation/experiments.",
        "status": "started",
    }


async def _run_evaluation_task():
    """Background task to run evaluation."""
    try:
        from backend.evaluation.evaluator import run_evaluation
        await run_evaluation(
            dataset_path=str(DATASET_FILE),
            output_path=str(RESULTS_FILE),
        )
    except Exception as e:
        logger.error("Background evaluation failed: %s", e)


async def _run_experiments_task():
    """Background task to run multi-architecture experiment comparison."""
    try:
        from backend.evaluation.experiments import ExperimentRunner
        runner = ExperimentRunner()
        report = await runner.run_comparison(dataset_path=str(DATASET_FILE))
        runner.save_report(report, output_dir=str(EVAL_DIR))
        logger.info("Experiment comparison background run completed.")
    except Exception as e:
        logger.error("Background experiment comparison failed: %s", e)
