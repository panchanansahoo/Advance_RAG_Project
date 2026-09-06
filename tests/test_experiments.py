"""Tests for the Experiment Comparison Framework (PRD §23 & §24)."""

import pytest
import os
import json
import tempfile
from unittest.mock import AsyncMock, patch

from backend.evaluation.experiments import (
    ExperimentConfig,
    ExperimentResult,
    ExperimentRunner,
    ComparisonReport,
    QueryResult,
    PRESET_VANILLA_RAG,
    PRESET_HYBRID_RAG,
    PRESET_HYBRID_RERANKER,
    PRESET_ADVANCED_RAG,
    PRESET_AGENTIC_RAG,
    STANDARD_PRESETS,
)


class TestExperimentPresets:
    """Tests for experiment presets and configurations."""

    def test_standard_presets_count(self):
        assert len(STANDARD_PRESETS) == 5

    def test_vanilla_rag_preset(self):
        assert PRESET_VANILLA_RAG.name == "Vanilla RAG"
        assert PRESET_VANILLA_RAG.retrieval_strategy == "vector"
        assert not PRESET_VANILLA_RAG.reranker_enabled
        assert not PRESET_VANILLA_RAG.query_rewriting_enabled
        assert not PRESET_VANILLA_RAG.context_compression_enabled
        assert not PRESET_VANILLA_RAG.agentic_rag_force

    def test_hybrid_rag_preset(self):
        assert PRESET_HYBRID_RAG.name == "Hybrid RAG"
        assert PRESET_HYBRID_RAG.retrieval_strategy == "hybrid"
        assert not PRESET_HYBRID_RAG.reranker_enabled

    def test_hybrid_reranker_preset(self):
        assert PRESET_HYBRID_RERANKER.name == "Hybrid + Reranker"
        assert PRESET_HYBRID_RERANKER.retrieval_strategy == "hybrid"
        assert PRESET_HYBRID_RERANKER.reranker_enabled

    def test_advanced_rag_preset(self):
        assert PRESET_ADVANCED_RAG.name == "Advanced RAG"
        assert PRESET_ADVANCED_RAG.query_rewriting_enabled
        assert PRESET_ADVANCED_RAG.context_compression_enabled
        assert PRESET_ADVANCED_RAG.verification_enabled

    def test_agentic_rag_preset(self):
        assert PRESET_AGENTIC_RAG.name == "Agentic RAG"
        assert PRESET_AGENTIC_RAG.agentic_rag_enabled
        assert PRESET_AGENTIC_RAG.agentic_rag_force


class TestExperimentRunner:
    """Tests for the ExperimentRunner execution and report generation."""

    @pytest.fixture
    def sample_dataset(self):
        return [
            {
                "question": "What is machine learning?",
                "ground_truth": "Machine learning is a subfield of artificial intelligence.",
                "expected_sources": ["ml_intro.pdf", "ai_overview.pdf"],
            },
            {
                "question": "How does gradient descent work?",
                "ground_truth": "Gradient descent iteratively minimizes loss functions.",
                "expected_sources": ["optimization.pdf"],
            },
        ]

    @pytest.mark.asyncio
    async def test_run_single_experiment_with_mock_query_fn(self, sample_dataset):
        runner = ExperimentRunner()

        async def mock_query_fn(query: str, config: ExperimentConfig):
            return {
                "answer": f"Mock answer for {query}",
                "sources": ["ml_intro.pdf", "extra.pdf"],
                "input_tokens": 150,
                "output_tokens": 50,
                "metadata": {"test": True},
            }

        result = await runner.run_single_experiment(
            PRESET_HYBRID_RAG, sample_dataset, query_fn=mock_query_fn
        )

        assert isinstance(result, ExperimentResult)
        assert result.config_name == "Hybrid RAG"
        assert result.metrics["total_queries"] == 2.0
        assert result.metrics["success_rate"] == 1.0
        assert result.metrics["recall_at_5"] > 0
        assert len(result.query_results) == 2
        assert result.query_results[0].question == "What is machine learning?"

    @pytest.mark.asyncio
    async def test_run_comparison_report(self, sample_dataset):
        runner = ExperimentRunner()

        # Mock query fn that gives higher recall for hybrid/reranker
        async def mock_query_fn(query: str, config: ExperimentConfig):
            if config.name == "Vanilla RAG":
                sources = ["extra.pdf"]  # Lower recall
            else:
                sources = ["ml_intro.pdf", "ai_overview.pdf", "optimization.pdf"]  # Higher recall
            return {
                "answer": "Test answer",
                "sources": sources,
                "input_tokens": 100,
                "output_tokens": 30,
            }

        configs = [PRESET_VANILLA_RAG, PRESET_HYBRID_RAG, PRESET_HYBRID_RERANKER]
        report = await runner.run_comparison(
            configs=configs, dataset=sample_dataset, query_fn=mock_query_fn
        )

        assert isinstance(report, ComparisonReport)
        assert report.total_queries == 2
        assert len(report.experiments) == 3
        assert len(report.comparison_table) == 3
        assert "Hybrid RAG" in report.deltas_vs_baseline
        assert "Hybrid + Reranker" in report.deltas_vs_baseline

        # Generate markdown report
        md = runner.generate_markdown_report(report)
        assert "# RAG Architecture Benchmark" in md
        assert "Vanilla RAG" in md
        assert "Hybrid RAG" in md
        assert "Hybrid + Reranker" in md
        assert "Recall@5" in md

    @pytest.mark.asyncio
    async def test_save_report(self, sample_dataset):
        runner = ExperimentRunner()

        async def mock_query_fn(query: str, config: ExperimentConfig):
            return {"answer": "A", "sources": ["doc.pdf"], "input_tokens": 10, "output_tokens": 5}

        report = await runner.run_comparison(
            configs=[PRESET_VANILLA_RAG, PRESET_HYBRID_RAG],
            dataset=sample_dataset,
            query_fn=mock_query_fn,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = runner.save_report(report, output_dir=tmp_dir)
            assert os.path.exists(paths["json"])
            assert os.path.exists(paths["markdown"])

            with open(paths["json"], "r", encoding="utf-8") as f:
                data = json.load(f)
                assert data["total_queries"] == 2
                assert len(data["experiments"]) == 2
