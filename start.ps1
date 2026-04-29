# start.ps1 - wrapper Windows para o start universal testavel.
$ErrorActionPreference = "Stop"
$DIR = Split-Path -Parent $MyInvocation.MyCommand.Definition

$venvPython = Join-Path $DIR ".venv\Scripts\python.exe"
if (Test-Path $venvPython) {
    & $venvPython (Join-Path $DIR "start.py")
} else {
    python (Join-Path $DIR "start.py")
}
