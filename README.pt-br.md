# LangChain RAG MCP

*[Read in English](README.md) | Leia em Português*

> GitHub: [EderDomenici/mcp_ragger](https://github.com/EderDomenici/mcp_ragger)

Um servidor MCP (Model Context Protocol) que expõe um pipeline RAG (Retrieval-Augmented Generation) sobre a documentação oficial do LangChain, LangGraph e LangSmith.

## Funcionalidades
- **Integração MCP:** Expõe recursos de recuperação de documentação de forma transparente para clientes compatíveis com MCP.
- **Recuperação Avançada:** Incorpora *semantic chunking*, filtragem por metadados e recuperação otimizada utilizando *embeddings* Locais ou em Nuvem.
- **Avaliação e Métricas:** Pipelines de benchmark integrados e avaliadores impulsionados pelo Ragas.
- **Analytics:** Métricas internas e rastreamento de uso.

## Estrutura de Diretórios

```text
.
├── src/langchain_rag_mcp/   # Código-fonte principal do pacote MCP e RAG
├── scripts/                 # Scripts de benchmark, avaliação e relatórios
├── tests/                   # Testes automatizados unitários e de integração
├── docs/                    # Documentação do projeto (ADRs, Roadmaps)
├── benchmarks/              # Snapshots de resultados históricos de benchmark
├── data/                    # Métricas geradas e dados locais de runtime (ignorado no git)
└── models/                  # Modelos de embedding baixados localmente (ignorado no git)
```

Arquivos na raiz lidam com configuração do projeto, gerenciamento de dependências e os principais entrypoints da aplicação.

## Pré-requisitos

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (recomendado para gerenciamento rápido de dependências)
- Docker (opcional, para rodar o Qdrant localmente)

## Configuração

1. Copie o arquivo de ambiente de exemplo:
   ```bash
   cp .env.example .env
   ```

2. Configure suas variáveis de ambiente. 
   Para uma instância local do Qdrant, `QDRANT_API_KEY` pode ficar vazio. Para Qdrant Cloud:
   ```env
   QDRANT_URL=https://SEU-CLUSTER.qdrant.io
   QDRANT_API_KEY=sua_api_key
   ```

## Setup e Execução

### 1. Subir Infraestrutura
Inicie a infraestrutura local (Qdrant) via Docker Compose:
```bash
./scripts/start.sh
```
*(Usuários Windows podem executar `./scripts/start.ps1`)*

### 2. Indexar Documentação
Preencha o banco de dados vetorial executando o indexador. Forneça uma URL de origem válida (ex: `llms.txt` do LangChain):
```bash
uv run langchain-rag-indexer --source-url https://docs.langchain.com/llms.txt
```

### 3. Rodar o Servidor MCP
Inicie o servidor principal do MCP:
```bash
uv run langchain-rag-mcp
```

## Testes e Benchmarks

### Testes
Rode a suíte de testes unitários padrão:
```bash
uv run python -m unittest discover -v
```

### Benchmarks
Execute benchmarks de performance, recuperação e o dataset golden:
```bash
uv run scripts/benchmark_golden.py
uv run scripts/benchmark.py
```

### Avaliação Ragas
Valide as configurações do juiz Ragas e OpenRouter. O pipeline de juiz tenta usar `deepseek/deepseek-v4-flash` e recai para `google/gemini-2.5-flash` em caso de falha.
```bash
env PYTHONPATH=src uv run scripts/benchmark_ragas.py --dry-run
```

**Últimos Resultados (`deepseek/deepseek-v4-flash`):**
| Métrica | Pontuação | Barra |
|---|---|---|
| context_recall | **0.792** | `████████░░` |
| context_precision | **0.796** | `████████░░` |
| **Média** | **0.794** | `████████░░` |

*Veja `results/ragas-report.md` para o detalhamento completo por query.*


### Estatísticas de Uso
Veja as métricas de uso interno (os dados são persistidos localmente em `data/metrics.db`):
```bash
uv run scripts/stats.py
```
