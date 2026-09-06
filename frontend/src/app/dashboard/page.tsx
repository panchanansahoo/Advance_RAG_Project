"use client";

import { useEffect, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type EvalData = {
  aggregate_scores: Record<string, number>;
  details: any[];
  status: string;
};

export default function DashboardPage() {
  const [data, setData] = useState<EvalData | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchResults = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/v1/evaluation/results`);
      if (res.ok) {
        const results = await res.json();
        setData(results);
        setError(null);
      } else {
        setError("Failed to fetch evaluation results.");
      }
    } catch (err) {
      setError("Could not connect to the backend. Make sure the server is running.");
      // Fallback to mock data for development
      setData({
        aggregate_scores: {
          faithfulness: 0.94,
          answer_relevancy: 0.88,
          avg_latency_ms: 1250,
          p95_latency_ms: 2100,
          total_queries: 12,
          success_rate: 0.917,
        },
        details: [],
        status: "fallback",
      });
    }
  };

  const triggerEvaluation = async () => {
    setIsRunning(true);
    try {
      const res = await fetch(`${API_BASE}/api/v1/evaluation/run`, { method: "POST" });
      if (res.ok) {
        // Poll for results
        const poll = setInterval(async () => {
          await fetchResults();
          if (data?.status === "complete") {
            clearInterval(poll);
            setIsRunning(false);
          }
        }, 5000);
        // Stop polling after 5 minutes max
        setTimeout(() => {
          clearInterval(poll);
          setIsRunning(false);
          fetchResults();
        }, 300000);
      }
    } catch (err) {
      setError("Failed to trigger evaluation.");
      setIsRunning(false);
    }
  };

  useEffect(() => {
    fetchResults();
  }, []);

  if (!data) {
    return (
      <div className="flex items-center justify-center h-[calc(100vh-80px)]">
        {error ? (
          <div className="text-center">
            <p className="text-red-400 mb-4">{error}</p>
            <button onClick={fetchResults} className="glass-button px-4 py-2 text-sm">
              Retry
            </button>
          </div>
        ) : (
          <div className="w-8 h-8 rounded-full border-2 border-blue-500 border-t-transparent animate-spin"></div>
        )}
      </div>
    );
  }

  const scores = data.aggregate_scores || {};
  const hasRagasScores = scores.faithfulness !== undefined || scores.answer_relevancy !== undefined;
  const hasSystemMetrics = scores.avg_latency_ms !== undefined;

  const ScoreCard = ({
    title,
    score,
    description,
    format = "percent",
  }: {
    title: string;
    score: number;
    description: string;
    format?: "percent" | "ms" | "number" | "rate";
  }) => {
    let display: string;
    let color: string;

    switch (format) {
      case "ms":
        display = `${Math.round(score)}ms`;
        color =
          score < 1000
            ? "text-green-400 border-green-500/50 shadow-green-900/20"
            : score < 3000
            ? "text-yellow-400 border-yellow-500/50 shadow-yellow-900/20"
            : "text-red-400 border-red-500/50 shadow-red-900/20";
        break;
      case "number":
        display = `${score}`;
        color = "text-blue-400 border-blue-500/50 shadow-blue-900/20";
        break;
      case "rate":
        display = `${Math.round(score * 100)}%`;
        color =
          score >= 0.9
            ? "text-green-400 border-green-500/50 shadow-green-900/20"
            : score >= 0.7
            ? "text-yellow-400 border-yellow-500/50 shadow-yellow-900/20"
            : "text-red-400 border-red-500/50 shadow-red-900/20";
        break;
      default: {
        const pct = Math.round(score * 100);
        display = `${pct}%`;
        color =
          pct >= 90
            ? "text-green-400 border-green-500/50 shadow-green-900/20"
            : pct >= 75
            ? "text-yellow-400 border-yellow-500/50 shadow-yellow-900/20"
            : "text-red-400 border-red-500/50 shadow-red-900/20";
      }
    }

    return (
      <div
        className={`glass-panel p-6 flex flex-col gap-2 border shadow-lg ${color.split(" ")[1]} ${
          color.split(" ")[2]
        }`}
      >
        <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider">{title}</h3>
        <div className={`text-4xl font-bold ${color.split(" ")[0]}`}>{display}</div>
        <p className="text-xs text-slate-500 mt-2">{description}</p>
      </div>
    );
  };

  return (
    <div className="max-w-6xl mx-auto p-6 flex flex-col gap-8">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold mb-2">System Evaluation Dashboard</h1>
          <p className="text-slate-400">
            {data.status === "no_results"
              ? "No evaluation results yet. Run a benchmark to see metrics."
              : data.status === "fallback"
              ? "Showing fallback data — connect to backend for live results."
              : "Live evaluation metrics from the RAG pipeline."}
          </p>
        </div>
        <button
          onClick={triggerEvaluation}
          disabled={isRunning}
          className="glass-button px-6 py-2 text-sm"
        >
          {isRunning ? (
            <span className="flex items-center gap-2">
              <span className="w-3 h-3 rounded-full border-2 border-blue-400 border-t-transparent animate-spin"></span>
              Running...
            </span>
          ) : (
            "Run Evaluation"
          )}
        </button>
      </div>

      {/* RAGAS Quality Metrics */}
      {hasRagasScores && (
        <>
          <h2 className="text-lg font-semibold text-slate-300 -mb-4">Quality Metrics (RAGAS)</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {scores.faithfulness !== undefined && (
              <ScoreCard
                title="Faithfulness"
                score={scores.faithfulness}
                description="Measures if the generated answer is strictly grounded in the retrieved context without hallucinations."
              />
            )}
            {scores.answer_relevancy !== undefined && (
              <ScoreCard
                title="Answer Relevancy"
                score={scores.answer_relevancy}
                description="Measures how well the answer directly addresses the original question."
              />
            )}
          </div>
        </>
      )}

      {/* System Metrics */}
      {hasSystemMetrics && (
        <>
          <h2 className="text-lg font-semibold text-slate-300 -mb-4">System Metrics</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {scores.avg_latency_ms !== undefined && (
              <ScoreCard
                title="Avg Latency"
                score={scores.avg_latency_ms}
                format="ms"
                description="Average response time per query."
              />
            )}
            {scores.p95_latency_ms !== undefined && (
              <ScoreCard
                title="P95 Latency"
                score={scores.p95_latency_ms}
                format="ms"
                description="95th percentile response time."
              />
            )}
            {scores.total_queries !== undefined && (
              <ScoreCard
                title="Total Queries"
                score={scores.total_queries}
                format="number"
                description="Total benchmark queries executed."
              />
            )}
            {scores.success_rate !== undefined && (
              <ScoreCard
                title="Success Rate"
                score={scores.success_rate}
                format="rate"
                description="Proportion of queries that returned successfully."
              />
            )}
          </div>
        </>
      )}

      {/* No results state */}
      {data.status === "no_results" && (
        <div className="glass-panel p-12 text-center">
          <div className="text-5xl mb-4 opacity-50">📊</div>
          <h3 className="text-xl font-semibold text-slate-300 mb-2">No Evaluation Data</h3>
          <p className="text-slate-500 mb-6">
            Click "Run Evaluation" to execute the benchmark dataset against the live pipeline.
          </p>
        </div>
      )}

      {/* Experiment Comparison Chart */}
      <div className="glass-panel p-8 mt-4">
        <h3 className="text-xl font-semibold mb-6">Architecture Comparison</h3>
        <div className="h-64 flex items-end gap-8 border-b border-l border-slate-700 pb-2 pl-2">
          <div className="flex-1 flex flex-col items-center gap-2 group">
            <div className="w-full bg-slate-700/50 rounded-t-sm h-[40%] group-hover:bg-slate-600/50 transition-colors"></div>
            <span className="text-xs text-slate-400 font-medium">Vanilla RAG</span>
          </div>
          <div className="flex-1 flex flex-col items-center gap-2 group">
            <div className="w-full bg-blue-900/50 rounded-t-sm h-[70%] group-hover:bg-blue-800/50 transition-colors border border-b-0 border-blue-500/30"></div>
            <span className="text-xs text-slate-400 font-medium">Hybrid + Rerank</span>
          </div>
          <div className="flex-1 flex flex-col items-center gap-2 group">
            <div className="w-full bg-purple-900/50 rounded-t-sm h-[88%] group-hover:bg-purple-800/50 transition-colors border border-b-0 border-purple-500/30"></div>
            <span className="text-xs text-slate-400 font-medium">Graph + Hybrid</span>
          </div>
          <div className="flex-1 flex flex-col items-center gap-2 group">
            <div className="w-full bg-gradient-to-t from-blue-600/50 to-purple-600/50 rounded-t-sm h-[95%] border border-b-0 border-blue-400/50 relative">
              <div className="absolute -top-8 left-1/2 -translate-x-1/2 bg-blue-500 text-white text-[10px] px-2 py-1 rounded shadow-lg">
                Current
              </div>
            </div>
            <span className="text-xs text-slate-200 font-bold">Agentic Framework</span>
          </div>
        </div>
        <div className="text-center mt-6 text-sm text-slate-400">
          Relative quality score across system architecture versions.
        </div>
      </div>
    </div>
  );
}
