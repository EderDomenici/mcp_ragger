import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def import_file(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class EntrypointTests(unittest.TestCase):
    def test_mcp_server_import_does_not_create_mcp(self):
        import langchain_rag_mcp.server

        with patch("langchain_rag_mcp.server.create_mcp") as create_mcp:
            module = import_file("mcp_server_import_test", ROOT / "mcp_server.py")

        create_mcp.assert_not_called()
        self.assertTrue(hasattr(module, "main"))

    def test_mcp_server_main_creates_and_runs_mcp(self):
        import langchain_rag_mcp.server

        module = import_file("mcp_server_main_test", ROOT / "mcp_server.py")
        with patch.object(module, "create_mcp") as create_mcp:
            module.main()

        create_mcp.assert_called_once_with()
        create_mcp.return_value.run.assert_called_once_with()

    def test_start_shell_wrappers_delegate_to_root_start_py(self):
        self.assertIn("start.py", (ROOT / "start.sh").read_text())
        self.assertIn("start.py", (ROOT / "start.ps1").read_text())


if __name__ == "__main__":
    unittest.main()
