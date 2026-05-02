#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_SERVER="/home/linuxbrew/.linuxbrew/bin/llama-server"
MODELS_DIR="/home/eder/Documentos/Projetos/LLM_CPP"
MODEL="$MODELS_DIR/nomic-embed-text-v1.5.Q8_0.gguf"
EMBED_PORT=8080

# --- Baixa o modelo se ainda não existe ---
if [ ! -f "$MODEL" ]; then
  echo "==> Baixando nomic-embed-text-v1.5.Q8_0.gguf (~274MB)..."
  wget -q --show-progress \
    "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q8_0.gguf" \
    -O "$MODEL"
fi

# --- Sobe Qdrant ---
echo "==> Subindo Qdrant..."
docker compose -f "$DIR/docker-compose.yml" up -d
until curl -sf http://localhost:6333/readyz > /dev/null; do sleep 1; done

# --- Inicia llama-server para embedding ---
echo "==> Iniciando llama-server (GPU)..."
"$LLAMA_SERVER" \
  -m "$MODEL" \
  --embedding \
  --pooling mean \
  -ngl 99 \
  --port $EMBED_PORT \
  --ctx-size 32768 \
  --log-disable \
  > /tmp/llama-embed.log 2>&1 &
LLAMA_PID=$!

echo "  PID $LLAMA_PID — aguardando servidor..."
until curl -sf http://localhost:$EMBED_PORT/health > /dev/null 2>&1; do sleep 1; done
echo "  llama-server pronto"

# --- venv e dependências ---
echo "==> Criando venv..."
python3 -m venv "$DIR/.venv"
"$DIR/.venv/bin/pip" install -q --upgrade pip
"$DIR/.venv/bin/pip" install -q -r "$DIR/requirements.txt"

# --- Indexa ---
echo "==> Indexando documentação LangChain..."
"$DIR/.venv/bin/python" -m langchain_rag_mcp.indexer

# --- Para o llama-server (só precisava para indexar) ---
kill $LLAMA_PID 2>/dev/null || true
echo "==> llama-server encerrado"

echo ""
echo "Setup concluído."
echo "Para usar o MCP, inicie o llama-server antes de abrir o Claude Code:"
echo ""
echo "  $LLAMA_SERVER -m $MODEL --embedding --pooling mean -ngl 99 --port $EMBED_PORT --log-disable &"
echo ""
echo "Depois registre o MCP:"
echo ""
echo "  claude mcp add langchain-rag $DIR/.venv/bin/python -m langchain_rag_mcp.server"
