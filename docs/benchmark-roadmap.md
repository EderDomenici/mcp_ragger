# RAG Benchmark Roadmap

Contexto: o benchmark expandido atual e util para desenvolvimento, mas nao deve ser tratado como prova final de qualidade do RAG. Ele mede recuperacao tecnica, nao qualidade completa da resposta final do MCP.

## Estado Atual

- O benchmark principal tem 60 casos.
- Mede hit rate, keyword hit, score threshold, top1 source e top3 source.
- O resultado calibrado mais recente esta salvo em `benchmarks/expanded-rerank-calibrated-2026-04-28.txt`.
- A versao atual do MCP usa busca vetorial com rerank lexical leve sobre candidatos semanticos.

## Riscos de Vies

- As queries foram escritas manualmente depois de conhecer o corpus.
- As keywords ainda sao validacao fraca: basta uma keyword aparecer para contar como keyword hit.
- `source_any` aceita familias de URL quando a documentacao mistura LangChain, LangGraph e LangSmith em paginas relacionadas.
- O benchmark nao mede diretamente a resposta final do MCP, apenas os chunks recuperados.
- Ainda ha poucos casos negativos e ambiguos.
- Nao existe comparacao A/B automatizada entre estrategias de indice.

## Melhorias Futuras

1. Extrair os casos para `benchmark_cases.json`.
2. Separar casos por tipo:
   - positivos comuns;
   - positivos com API/simbolos;
   - ambiguos;
   - negativos fora do escopo;
   - APIs antigas/removidas.
3. Adicionar validacao por fonte mais especifica quando a pagina correta for conhecida.
4. Medir `top1_keyword_hit`, `top3_keyword_hit`, `top1_source_hit`, `top3_source_hit` e MRR.
5. Criar benchmark A/B:
   - indice antigo;
   - indice novo sem rerank;
   - indice novo com rerank.
6. Adicionar avaliacao da resposta final do MCP, nao apenas dos chunks.
7. Salvar resultados em JSON/CSV alem do output textual.
8. Adicionar queries negativas, por exemplo:
   - `React useEffect cleanup`;
   - `Django middleware`;
   - `Kubernetes ingress controller`;
   - `LangChain v0.0 ConversationBufferMemory`;
   - `LCEL vs RunnableSequence`.

## Decisao Pratica

Antes de continuar otimizando chunking, priorizar:

1. benchmark menos enviesado;
2. rerank melhor;
3. avaliacao A/B;
4. casos negativos;
5. medicao da resposta final do MCP.
