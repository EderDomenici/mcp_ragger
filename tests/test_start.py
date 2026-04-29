import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from langchain_rag_mcp import startup as start


class StartConfigTests(unittest.TestCase):
    def test_defaults_use_project_local_model_and_linux_venv(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)

            config = start.resolve_config(project_dir, env={}, platform_name="linux")

            self.assertEqual(config.project_dir, project_dir)
            self.assertEqual(config.model, project_dir / "models" / start.MODEL_FILENAME)
            self.assertEqual(config.python, project_dir / ".venv" / "bin" / "python")
            self.assertEqual(config.embed_port, 8080)

    def test_default_model_falls_back_to_shared_llm_cpp_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp) / "Projetos" / "MCPs" / "langchain-rag"
            shared_models = Path(tmp) / "Projetos" / "LLM_CPP"
            shared_models.mkdir(parents=True)
            shared_model = shared_models / start.MODEL_FILENAME
            shared_model.write_text("fake model")
            project_dir.mkdir(parents=True)

            config = start.resolve_config(project_dir, env={}, platform_name="linux")

            self.assertEqual(config.model, shared_model)

    def test_windows_defaults_use_exe_and_scripts_python(self):
        project_dir = Path("C:/repo/langchain-rag")

        config = start.resolve_config(project_dir, env={}, platform_name="win32")

        self.assertEqual(config.llama_server, Path("llama-server.exe"))
        self.assertEqual(config.python, project_dir / ".venv" / "Scripts" / "python.exe")

    def test_environment_overrides_paths_and_port(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            env = {
                "LLAMA_SERVER": "/opt/llama/bin/llama-server",
                "MODEL": "/models/custom.gguf",
                "EMBED_PORT": "9090",
            }

            config = start.resolve_config(project_dir, env=env, platform_name="linux")

            self.assertEqual(config.llama_server, Path("/opt/llama/bin/llama-server"))
            self.assertEqual(config.model, Path("/models/custom.gguf"))
            self.assertEqual(config.embed_port, 9090)

    def test_llama_command_is_argument_list_without_shell_quoting(self):
        config = start.StartConfig(
            project_dir=Path("/repo"),
            llama_server=Path("/bin/llama-server"),
            model=Path("/repo/models/model.gguf"),
            python=Path("/repo/.venv/bin/python"),
            embed_port=8080,
            ctx_size=32768,
            gpu_layers=99,
        )

        command = start.build_llama_command(config)

        self.assertEqual(command[0], os.fspath(config.llama_server))
        self.assertIn(os.fspath(config.model), command)
        self.assertIn("--embedding", command)
        self.assertIn("--pooling", command)
        self.assertIn("mean", command)
        self.assertIn("--port", command)
        self.assertIn("8080", command)

    def test_start_llama_detaches_process_on_linux(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model.gguf"
            model.write_text("fake model")
            config = start.StartConfig(
                project_dir=Path(tmp),
                llama_server=Path("/bin/llama-server"),
                model=model,
                python=Path(tmp) / ".venv" / "bin" / "python",
                embed_port=8080,
                ctx_size=32768,
                gpu_layers=99,
            )
            proc = Mock(pid=123)

            with (
                patch.object(start.sys, "platform", "linux"),
                patch.object(start, "_wait_for_url"),
                patch.object(start.subprocess, "Popen", return_value=proc) as popen,
            ):
                result = start._start_llama(config)

        self.assertEqual(result, proc)
        self.assertTrue(popen.call_args.kwargs["start_new_session"])


if __name__ == "__main__":
    unittest.main()
