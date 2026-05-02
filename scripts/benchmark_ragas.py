"""
Ragas/OpenRouter evaluation entrypoint.

The implementation lives under src/langchain_rag_mcp/evals so this file stays
as a thin CLI wrapper.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))

from qdrant_client import QdrantClient

from langchain_rag_mcp.config import Settings
from langchain_rag_mcp.embeddings import create_embeddings
from langchain_rag_mcp.env import load_project_env
from langchain_rag_mcp.evals.openrouter_judge import (
    JudgeSmokeResult,  # re-exported so tests can reference benchmark_ragas.JudgeSmokeResult
    OpenRouterJudgeConfig,
    create_openai_compatible_client,
    create_ragas_llm,
)
from langchain_rag_mcp.evals.ragas_runner import (
    RagasEvalSample,
    golden_case_to_sample,
    require_ragas,
)
from langchain_rag_mcp.retrieval import rerank

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from benchmark_golden import CANDIDATE_K, CHUNK_CAP, GOLDEN_CASES, TOP_K, GoldenCase


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="cheap", choices=["cheap", "strict"])
    parser.add_argument("--dry-run", action="store_true", help="Valida configuracao sem chamar OpenRouter/Ragas.")
    parser.add_argument("--smoke", action="store_true", help="Faz uma chamada simples ao judge OpenRouter.")
    parser.add_argument("--limit", type=int, default=0, help="Limita a quantidade de casos avaliados.")
    parser.add_argument("--fail-under", type=float, default=0.60, help="Score minimo para exit 0.")
    parser.add_argument("--output", metavar="PATH", help="Salva resultado em arquivo (.json ou .md).")
    return parser


def _print_config(config: OpenRouterJudgeConfig, *, mode: str) -> None:
    print(f"Ragas judge mode: {mode}")
    print("OpenRouter model fallback order:")
    for index, model in enumerate(config.models, start=1):
        print(f"  {index}. {model}")


def _collect_samples(
    cases: list[GoldenCase],
    qdrant: QdrantClient,
    embeddings,
    settings: Settings,
) -> list[RagasEvalSample]:
    samples = []
    chunk_cap = max(settings.chunk_cap, CHUNK_CAP)

    for case in cases:
        query_vector = embeddings.embed_query(case.query)
        points = qdrant.query_points(
            collection_name=settings.collection,
            query=query_vector,
            limit=max(settings.candidate_k, CANDIDATE_K),
            with_payload=True,
        ).points
        results = rerank(case.query, points, max(settings.top_k, TOP_K))
        sample = golden_case_to_sample(case, results, chunk_cap=chunk_cap)
        samples.append(sample)
        print(f"  retrieved {len(results)} chunks — {case.query[:72]}")

    return samples


def _build_metrics(llm):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from ragas.metrics import ContextPrecision, ContextRecall

    return [ContextRecall(llm=llm), ContextPrecision(llm=llm)]


def _extract_scores(result) -> dict[str, float]:
    if hasattr(result, "items"):
        items = result.items()
    else:
        items = getattr(result, "_repr_dict", {}).items()
    return {
        str(key): float(value)
        for key, value in items
        if isinstance(value, (int, float)) and not math.isnan(float(value))
    }


def _average_score(scores: dict[str, float]) -> float:
    return sum(scores.values()) / len(scores) if scores else 0.0


def _per_sample_scores(result) -> list[dict[str, float]]:
    """Extracts per-sample score dicts from the ragas EvaluationResult."""
    scores = getattr(result, "scores", None)
    if not scores:
        return []
    return [
        {k: round(float(v), 4) for k, v in row.items() if isinstance(v, (int, float)) and not math.isnan(float(v))}
        for row in scores
    ]


def _build_report(
    samples: list[RagasEvalSample],
    per_sample: list[dict[str, float]],
    aggregate: dict[str, float],
    config: OpenRouterJudgeConfig,
    settings: Settings,
    run_at: str,
) -> dict:
    cases_data = []
    for i, sample in enumerate(samples):
        row: dict = {
            "query": sample.user_input,
            "category": sample.metadata.get("category", ""),
            "expected_sources": sample.metadata.get("expected_sources", []),
            "required_terms": sample.metadata.get("required_terms", []),
            "retrieved_chunks": len(sample.retrieved_contexts),
        }
        if i < len(per_sample):
            row["scores"] = per_sample[i]
        cases_data.append(row)

    return {
        "run_at": run_at,
        "system": {
            "embedding_provider": settings.embedding_provider,
            "embedding_model": settings.google_embedding_model or "llamacpp",
            "vector_db": "qdrant",
            "collection": settings.collection,
            "top_k": settings.top_k,
            "candidate_k": settings.candidate_k,
            "min_score": settings.min_score,
        },
        "judge": {
            "model": config.primary_model,
            "fallback_models": list(config.models),
            "temperature": config.temperature,
        },
        "metrics": {
            "aggregate": {k: round(v, 4) for k, v in aggregate.items()},
            "average": round(sum(aggregate.values()) / len(aggregate), 4) if aggregate else 0.0,
        },
        "cases": cases_data,
    }


def _score_bar(score: float, width: int = 10) -> str:
    filled = round(score * width)
    return "█" * filled + "░" * (width - filled)


def _render_markdown(report: dict) -> str:
    sys_info = report["system"]
    judge = report["judge"]
    metrics = report["metrics"]
    aggregate = metrics["aggregate"]
    avg = metrics["average"]
    cases = report["cases"]

    lines: list[str] = []
    lines.append("# RAG Evaluation Report")
    lines.append(f"\n**Date:** {report['run_at']}  ")
    lines.append(f"**Cases evaluated:** {len(cases)}  ")
    lines.append(f"**Judge model:** `{judge['model']}`\n")

    lines.append("## System")
    lines.append("| Parameter | Value |")
    lines.append("|---|---|")
    lines.append(f"| Embedding provider | `{sys_info['embedding_provider']}` |")
    lines.append(f"| Embedding model | `{sys_info['embedding_model']}` |")
    lines.append(f"| Vector DB | `{sys_info['vector_db']}` |")
    lines.append(f"| Collection | `{sys_info['collection']}` |")
    lines.append(f"| Top-K retrieved | `{sys_info['top_k']}` |")
    lines.append(f"| Candidate pool | `{sys_info['candidate_k']}` |")
    lines.append(f"| Min similarity score | `{sys_info['min_score']}` |")

    lines.append("\n## Aggregate Metrics")
    lines.append("| Metric | Score | Bar |")
    lines.append("|---|---|---|")
    for metric, score in aggregate.items():
        lines.append(f"| {metric} | **{score:.3f}** | `{_score_bar(score)}` |")
    lines.append(f"| **Average** | **{avg:.3f}** | `{_score_bar(avg)}` |")

    lines.append("\n### What these metrics mean")
    lines.append(
        "- **Context Recall** — fraction of the expected answer that is covered by the retrieved chunks. "
        "A score of 1.0 means everything the user would need to know was retrieved."
    )
    lines.append(
        "- **Context Precision** — fraction of retrieved chunks that are actually relevant to the query. "
        "A score of 1.0 means no noisy or irrelevant chunks were returned."
    )

    if cases and cases[0].get("scores"):
        lines.append("\n## Per-Case Breakdown")
        metric_keys = list(cases[0]["scores"].keys())
        header = "| # | Category | Query | " + " | ".join(metric_keys) + " |"
        separator = "|---|---|---| " + " | ".join(["---"] * len(metric_keys)) + " |"
        lines.append(header)
        lines.append(separator)
        for i, case in enumerate(cases, start=1):
            scores_str = " | ".join(
                f"{case['scores'].get(k, 0):.3f}" for k in metric_keys
            )
            query_short = case["query"][:60] + ("…" if len(case["query"]) > 60 else "")
            lines.append(f"| {i} | `{case['category']}` | {query_short} | {scores_str} |")

    lines.append("\n## Golden Cases Covered")
    lines.append("| Category | Query | Expected Sources |")
    lines.append("|---|---|---|")
    for case in cases:
        sources = ", ".join(f"`{s}`" for s in case["expected_sources"]) or "—"
        query_short = case["query"][:65] + ("…" if len(case["query"]) > 65 else "")
        lines.append(f"| `{case['category']}` | {query_short} | {sources} |")

    lines.append(f"\n---\n*Generated by [langchain-rag-mcp](https://github.com/your-repo) · Judge: `{judge['model']}`*")

    return "\n".join(lines)


def _save_output(path: str, report: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.suffix == ".md":
        output.write_text(_render_markdown(report), encoding="utf-8")
    else:
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nSaved to {output.resolve()}")


def run(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_project_env()
    config = OpenRouterJudgeConfig.from_env()
    _print_config(config, mode=args.mode)

    if args.dry_run:
        return 0

    if args.smoke:
        result = create_openai_compatible_client(config).smoke_judge()
        print(f"Smoke judge response: {result.text}")
        print(f"Model used: {result.model}")
        return 0

    require_ragas()

    from ragas import EvaluationDataset, evaluate

    settings = Settings.from_env()
    qdrant = QdrantClient(**settings.qdrant_client_kwargs(), check_compatibility=False)
    embeddings = create_embeddings(
        settings.embedding_provider,
        llamacpp_url=settings.llamacpp_url,
        google_model=settings.google_embedding_model,
        google_output_dimensionality=settings.google_output_dimensionality,
    )

    positive_cases = [c for c in GOLDEN_CASES if not c.negative]
    if args.limit > 0:
        positive_cases = positive_cases[: args.limit]

    print(f"\nRunning retrieval for {len(positive_cases)} positive golden cases...")
    samples = _collect_samples(positive_cases, qdrant, embeddings, settings)

    if not samples:
        print("No samples collected. Aborting.")
        return 1

    ragas_llm = create_ragas_llm(config)
    metrics = _build_metrics(ragas_llm)
    dataset = EvaluationDataset.from_list([s.as_dict() for s in samples])

    print(f"\nEvaluating {len(samples)} samples with Ragas (judge: {config.primary_model})...")
    result = evaluate(dataset, metrics=metrics)

    aggregate = _extract_scores(result)
    per_sample = _per_sample_scores(result)

    print("\n" + "─" * 60)
    print("Ragas Results:")
    for metric, score in aggregate.items():
        print(f"  {metric}: {score:.3f}")

    if not aggregate:
        print("  No numeric Ragas scores were produced. Check judge/provider errors above.")
        return 1

    avg = _average_score(aggregate)
    print(f"\nAverage score : {avg:.3f}")
    print(f"Threshold     : {args.fail_under}")

    if args.output:
        run_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        report = _build_report(samples, per_sample, aggregate, config, settings, run_at)
        _save_output(args.output, report)

    return 0 if avg >= args.fail_under else 1


if __name__ == "__main__":
    raise SystemExit(run())
