# LangChain RAG MCP

*Read in English | [Leia em Português](README.pt-br.md)*

An MCP (Model Context Protocol) server exposing a RAG (Retrieval-Augmented Generation) pipeline over the official documentation of LangChain, LangGraph, and LangSmith. 

## Features
- **MCP Integration:** Exposes documentation retrieval capabilities seamlessly to MCP-compatible clients.
- **Advanced Retrieval:** Incorporates semantic chunking, metadata filtering, and optimized retrieval using Local or Cloud Embeddings.
- **Evaluation & Metrics:** Built-in benchmarking pipelines and evaluators powered by Ragas.
- **Analytics:** Internal metrics and usage tracking.

## Directory Structure

```text
.
├── src/langchain_rag_mcp/   # Main MCP and RAG package source code
├── scripts/                 # Benchmarks, evaluators, and reporting scripts
├── tests/                   # Automated unit and integration tests
├── docs/                    # Project documentation (ADRs, Roadmaps)
├── benchmarks/              # Historical benchmark snapshots
├── data/                    # Generated metrics and local runtime data (ignored in git)
└── models/                  # Locally downloaded embedding models (ignored in git)
```

Root-level files handle project configuration, dependency management, and primary application entrypoints.

## Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (recommended for fast dependency management)
- Docker (optional, for running Qdrant locally)

## Configuration

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Configure your environment variables. 
   For a local Qdrant instance, `QDRANT_API_KEY` can be left empty. For Qdrant Cloud:
   ```env
   QDRANT_URL=https://YOUR-CLUSTER.qdrant.io
   QDRANT_API_KEY=your_api_key
   ```

## Setup & Execution

### 1. Start Infrastructure
Launch the local infrastructure (Qdrant) via Docker Compose:
```bash
./scripts/start.sh
```
*(Windows users can execute `./scripts/start.ps1`)*

### 2. Index Documentation
Populate the vector database by running the indexer. Provide a valid source URL (e.g., LangChain's `llms.txt`):
```bash
uv run langchain-rag-indexer --source-url https://docs.langchain.com/llms.txt
```

### 3. Run the MCP Server
Launch the main MCP server:
```bash
uv run langchain-rag-mcp
```

## Testing & Benchmarks

### Tests
Run the standard unit test suite:
```bash
uv run python -m unittest discover -v
```

### Benchmarks
Execute performance, retrieval, and golden dataset benchmarks:
```bash
uv run scripts/benchmark_golden.py
uv run scripts/benchmark.py
```

### Ragas Evaluation
Validate Ragas and OpenRouter judge configurations. The judge pipeline attempts to use `deepseek/deepseek-v4-flash` and falls back to `google/gemini-2.5-flash` on failure.
```bash
env PYTHONPATH=src uv run scripts/benchmark_ragas.py --dry-run
```

**Latest Results (`deepseek/deepseek-v4-flash`):**
| Metric | Score | Bar |
|---|---|---|
| context_recall | **0.792** | `████████░░` |
| context_precision | **0.796** | `████████░░` |
| **Average** | **0.794** | `████████░░` |

*See `results/ragas-report.md` for a full breakdown per-query.*


### Usage Statistics
View internal usage metrics (data is persisted locally in `data/metrics.db`):
```bash
uv run scripts/stats.py
```
