# setup.ps1 - Windows + NVIDIA CUDA
$ErrorActionPreference = "Stop"

$PYTHON       = "C:\Users\LIE\AppData\Local\Programs\Python\Python313\python.exe"
$DIR          = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Definition)
$LLAMA_SERVER = "C:\Users\LIE\Documents\Projetos\Llama\llama-server.exe"
$MODEL_DIR    = "$DIR\models"
$MODEL        = "$MODEL_DIR\nomic-embed-text-v1.5.Q8_0.gguf"
$EMBED_PORT   = 8080

if (-not (Test-Path $MODEL_DIR)) {
    New-Item -ItemType Directory -Path $MODEL_DIR | Out-Null
}

if (-not (Test-Path $MODEL)) {
    Write-Host "==> Baixando nomic-embed-text-v1.5.Q8_0.gguf (~274MB)..."
    $url = "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q8_0.gguf"
    Invoke-WebRequest -Uri $url -OutFile $MODEL -UseBasicParsing
    Write-Host "    Download concluido."
}

Write-Host "==> Subindo Qdrant..."
docker compose -f "$DIR\docker-compose.yml" up -d
Write-Host "    Aguardando Qdrant ficar pronto..."
do {
    Start-Sleep -Seconds 1
    try {
        $r = Invoke-WebRequest -Uri "http://localhost:6333/readyz" -UseBasicParsing -ErrorAction Stop
        $ready = $r.StatusCode -eq 200
    } catch { $ready = $false }
} while (-not $ready)
Write-Host "    Qdrant pronto."

Write-Host "==> Iniciando llama-server com CUDA..."
$logOut = "$env:TEMP\llama-embed.log"
$logErr = "$env:TEMP\llama-embed-err.log"
$llamaArgs = "-m `"$MODEL`" --embedding --pooling mean -ngl 99 --port $EMBED_PORT --ctx-size 32768 --log-disable"
$llamaProc = Start-Process -FilePath $LLAMA_SERVER -ArgumentList $llamaArgs -RedirectStandardOutput $logOut -RedirectStandardError $logErr -PassThru -WindowStyle Hidden

Write-Host "    PID $($llamaProc.Id) - aguardando servidor de embeddings..."
do {
    Start-Sleep -Seconds 1
    try {
        $r = Invoke-WebRequest -Uri "http://localhost:$EMBED_PORT/health" -UseBasicParsing -ErrorAction Stop
        $ready = $r.StatusCode -eq 200
    } catch { $ready = $false }
} while (-not $ready)
Write-Host "    llama-server pronto."

Write-Host "==> Criando venv Python..."
& $PYTHON -m venv "$DIR\.venv"
& "$DIR\.venv\Scripts\python.exe" -m pip install -q --upgrade pip
& "$DIR\.venv\Scripts\python.exe" -m pip install -q -r "$DIR\requirements.txt"

Write-Host "==> Indexando documentacao LangChain..."
& "$DIR\.venv\Scripts\python.exe" -m langchain_rag_mcp.indexer

Write-Host "==> Encerrando llama-server..."
Stop-Process -Id $llamaProc.Id -Force -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "============================================"
Write-Host "  Setup concluido!"
Write-Host "============================================"
Write-Host ""
Write-Host "Para usar o MCP, inicie o llama-server antes de abrir o Claude Code:"
Write-Host ""
$startCmd = '  & "' + $LLAMA_SERVER + '" -m "' + $MODEL + '" --embedding --pooling mean -ngl 99 --port ' + $EMBED_PORT + ' --log-disable'
Write-Host $startCmd
Write-Host ""
Write-Host "Depois registre o MCP (uma unica vez):"
Write-Host ""
$mcpCmd = '  claude mcp add langchain-rag "' + $DIR + '\.venv\Scripts\python.exe" -m langchain_rag_mcp.server'
Write-Host $mcpCmd
Write-Host ""
