import unittest

from langchain_rag_mcp.config import Settings


class SettingsTests(unittest.TestCase):
    def test_qdrant_api_key_is_loaded_and_used_when_present(self):
        settings = Settings.from_env(
            {
                "QDRANT_URL": "https://example.qdrant.io",
                "QDRANT_API_KEY": "secret-key",
            }
        )

        self.assertEqual(settings.qdrant_api_key, "secret-key")
        self.assertEqual(
            settings.qdrant_client_kwargs(),
            {"url": "https://example.qdrant.io", "api_key": "secret-key"},
        )

    def test_qdrant_api_key_is_omitted_for_local_qdrant(self):
        settings = Settings(qdrant_url="http://localhost:6333")

        self.assertIsNone(settings.qdrant_api_key)
        self.assertEqual(settings.qdrant_client_kwargs(), {"url": "http://localhost:6333"})


if __name__ == "__main__":
    unittest.main()
