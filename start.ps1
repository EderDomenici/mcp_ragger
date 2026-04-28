# start.ps1 - Sobe Qdrant + llama-server para uso do MCP (sem re-indexar)
$LLAMA_SERVER = "C:\Users\LIE\Documents\Projetos\Llama\llama-server.exe"
$MODEL        = "C:\Users\LIE\Documents\Projetos\MCP Ragger\models\nomic-embed-text-v1.5.Q8_0.gguf"
$DIR          = Split-Path -Parent $MyInvocation.MyCommand.Definition
$EMBED_PORT   = 8080

Write-Host "==> Subindo Qdrant..."
docker compose -f "$DIR\docker-compose.yml" up -d
do {
    Start-Sleep -Seconds 1
    try {
        $r = Invoke-WebRequest -Uri "http://localhost:6333/readyz" -UseBasicParsing -ErrorAction Stop
        $ready = $r.StatusCode -eq 200
    } catch { $ready = $false }
} while (-not $ready)
Write-Host "    Qdrant pronto."

Write-Host "==> Iniciando llama-server com CUDA..."
$llamaArgs = "-m `"$MODEL`" --embedding --pooling mean -ngl 99 --port $EMBED_PORT --ctx-size 32768 --log-disable"
$llamaProc = Start-Process -FilePath $LLAMA_SERVER -ArgumentList $llamaArgs -PassThru -WindowStyle Hidden

do {
    Start-Sleep -Seconds 1
    try {
        $r = Invoke-WebRequest -Uri "http://localhost:$EMBED_PORT/health" -UseBasicParsing -ErrorAction Stop
        $ready = $r.StatusCode -eq 200
    } catch { $ready = $false }
} while (-not $ready)
Write-Host "    llama-server pronto (PID $($llamaProc.Id))."

Write-Host ""
Write-Host "Tudo pronto. Pode abrir o Claude Code."
Write-Host "Para parar: docker compose -f `"$DIR\docker-compose.yml`" stop  +  kill PID $($llamaProc.Id)"
