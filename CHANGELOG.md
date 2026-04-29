# Changelog

## 2026-04-29 - LangChain RAG MCP baseline

### Estado atual

O projeto agora esta organizado como um MCP de RAG local para documentacao LangChain, LangGraph e LangSmith, indexado a partir de `https://docs.langchain.com/llms.txt`.

Resultado atual do benchmark golden:

```text
Golden hit rate : 15/15 (100%)
Recall@3        : 100%
MRR             : 0.847
Negative pass   : 100%
Avg latency     : 39ms
```

Resultado atual do benchmark smoke:

```text
Hit rate      : 58/60 (97%)
Top1 source   : 59/59 (100%)
Top3 source   : 59/59 (100%)
Avg latency   : 36ms
```

### Stack atual

- Python como linguagem principal.
- FastMCP como servidor MCP.
- Qdrant em Docker Compose como vector database.
- llama.cpp `llama-server` local para embeddings.
- Modelo local `nomic-embed-text-v1.5.Q8_0.gguf`.
- `unittest` para testes automatizados.
- Scripts cross-platform `start.py`, `start.sh` e `start.ps1`.

### Fluxo de RAG

```text
llms.txt
  -> baixa links .md/.mdx
  -> formata documentos com Source
  -> cria chunks com metadados semanticos
  -> gera embeddings locais via llama-server
  -> grava vetores no Qdrant
  -> MCP recebe query via search_docs
  -> gera embedding da query
  -> busca candidatos no Qdrant
  -> aplica rerank + filtro de cobertura
  -> retorna trechos com fonte
```

### Mudancas principais

- Extraida a implementacao do MCP para o pacote `src/langchain_rag_mcp`.
- `mcp_server.py` virou apenas um entrypoint fino.
- Adicionado `start.py` universal e wrappers `start.sh`/`start.ps1`.
- Adicionado suporte a indexacao de `llms.txt`, baixando documentos Markdown/MDX vinculados.
- Adicionado benchmark golden com casos positivos, negativos, Recall@3, MRR e cobertura de termos.
- Adicionado filtro de cobertura de termos para reduzir vazamento em perguntas fora do dominio.
- Reranker ajustado para promover docs especificos de LangChain middleware quando a query fala de agent/model/middleware.
- Benchmarks `benchmark.py` e `benchmark_golden.py` agora usam o reranker real de producao.
- Criada skill local `rag-benchmark` para orientar futuras avaliacoes de RAG.

### Comandos importantes

Subir infraestrutura local:

```bash
./start.sh
```

Indexar a documentacao LangChain completa:

```bash
.venv/bin/python indexer.py --source-url https://docs.langchain.com/llms.txt
```

Rodar testes:

```bash
.venv/bin/python -m unittest discover -v
```

Rodar golden benchmark:

```bash
.venv/bin/python benchmark_golden.py
```

Rodar smoke benchmark:

```bash
.venv/bin/python benchmark.py
```

### Proximos passos sugeridos

- Persistir snapshots de benchmark em `benchmarks/`.
- Expandir golden set antes de adicionar novos frameworks.
- Adicionar avaliacao por source aggregation se o Recall@3 cair em uma base maior.
- Transformar a indexacao multi-framework em configuracao explicita antes de misturar colecoes.
