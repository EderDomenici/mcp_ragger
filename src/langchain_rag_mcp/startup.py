from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .env import load_project_env


MODEL_FILENAME = "nomic-embed-text-v1.5.Q8_0.gguf"
QDRANT_READY_URL = "http://localhost:6333/readyz"


@dataclass(frozen=True)
class StartConfig:
    project_dir: Path
    embedding_provider: str
    llama_server: Path
    model: Path
    python: Path
    embed_port: int
    ctx_size: int
    gpu_layers: int


def _default_llama_server(platform_name: str) -> Path:
    executable = "llama-server.exe" if platform_name.startswith("win") else "llama-server"
    found = shutil.which(executable)
    return Path(found) if found else Path(executable)


def _default_python(project_dir: Path, platform_name: str) -> Path:
    if platform_name.startswith("win"):
        return project_dir / ".venv" / "Scripts" / "python.exe"
    return project_dir / ".venv" / "bin" / "python"


def _default_model(project_dir: Path) -> Path:
    project_model = project_dir / "models" / MODEL_FILENAME
    if project_model.exists():
        return project_model

    if len(project_dir.parents) >= 2:
        shared_model = project_dir.parents[1] / "LLM_CPP" / MODEL_FILENAME
        if shared_model.exists():
            return shared_model

    return project_model


def _env_path(env: Mapping[str, str], key: str, default: Path) -> Path:
    value = env.get(key)
    return Path(value).expanduser() if value else default


def _env_int(env: Mapping[str, str], key: str, default: int) -> int:
    value = env.get(key)
    return int(value) if value else default


def resolve_config(
    project_dir: Path,
    env: Mapping[str, str] | None = None,
    platform_name: str | None = None,
) -> StartConfig:
    if env is None:
        load_project_env()
        env = os.environ
    platform_name = sys.platform if platform_name is None else platform_name
    project_dir = project_dir.resolve() if project_dir.exists() else project_dir

    return StartConfig(
        project_dir=project_dir,
        embedding_provider=env.get("EMBEDDING_PROVIDER", "google"),
        llama_server=_env_path(env, "LLAMA_SERVER", _default_llama_server(platform_name)),
        model=_env_path(env, "MODEL", _default_model(project_dir)),
        python=_env_path(env, "PYTHON", _default_python(project_dir, platform_name)),
        embed_port=_env_int(env, "EMBED_PORT", 8080),
        ctx_size=_env_int(env, "CTX_SIZE", 32768),
        gpu_layers=_env_int(env, "GPU_LAYERS", 99),
    )


def build_llama_command(config: StartConfig) -> list[str]:
    return [
        os.fspath(config.llama_server),
        "-m",
        os.fspath(config.model),
        "--embedding",
        "--pooling",
        "mean",
        "-ngl",
        str(config.gpu_layers),
        "--port",
        str(config.embed_port),
        "--ctx-size",
        str(config.ctx_size),
        "--log-disable",
    ]


def _wait_for_url(url: str, label: str, timeout_seconds: int = 120) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if 200 <= response.status < 300:
                    return
        except Exception:
            pass
        time.sleep(1)
    raise TimeoutError(f"Timeout waiting for {label}: {url}")


def _start_qdrant(config: StartConfig) -> None:
    print("==> Subindo Qdrant...")
    subprocess.run(
        ["docker", "compose", "-f", os.fspath(config.project_dir / "docker-compose.yml"), "up", "-d"],
        check=True,
    )
    _wait_for_url(QDRANT_READY_URL, "Qdrant")
    print("    Qdrant pronto.")


def _start_llama(config: StartConfig) -> subprocess.Popen[bytes]:
    if not config.model.exists():
        raise FileNotFoundError(
            f"Modelo nao encontrado: {config.model}\n"
            "Execute setup primeiro ou defina MODEL=/caminho/para/modelo.gguf."
        )

    print("==> Iniciando llama-server...")
    log_path = Path(tempfile.gettempdir()) / "llama-embed.log"
    log = log_path.open("ab")
    if sys.platform.startswith("win"):
        creationflags = subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS
        start_new_session = False
    else:
        creationflags = 0
        start_new_session = True

    proc = subprocess.Popen(
        build_llama_command(config),
        stdout=log,
        stderr=subprocess.STDOUT,
        creationflags=creationflags,
        start_new_session=start_new_session,
    )
    health_url = f"http://localhost:{config.embed_port}/health"
    _wait_for_url(health_url, "llama-server")
    print(f"    llama-server pronto (PID {proc.pid}). Log: {log_path}")
    return proc


def main(project_dir: Path) -> int:
    config = resolve_config(project_dir)
    _start_qdrant(config)
    llama_proc = None
    if config.embedding_provider.strip().lower() in {"llamacpp", "llama.cpp", "local"}:
        llama_proc = _start_llama(config)

    print("")
    print("Tudo pronto. Pode abrir o Claude Code.")
    print(f"Para parar: docker compose -f \"{config.project_dir / 'docker-compose.yml'}\" stop")
    if llama_proc:
        print(f"Depois encerre o llama-server pelo PID {llama_proc.pid}.")
    else:
        print("Embeddings configurados para Google Gemini API; defina GOOGLE_API_KEY antes de consultar/indexar.")
    return 0

def cli_main() -> int:
    return main(Path.cwd())

if __name__ == "__main__":
    sys.exit(cli_main())
