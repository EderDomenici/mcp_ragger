import unittest
from types import SimpleNamespace
from unittest.mock import patch

import indexer
from langchain_rag_mcp.loaders import Document


class IndexerSourceTests(unittest.TestCase):
    def test_download_docs_uses_default_full_docs_when_no_source_url(self):
        response = SimpleNamespace(text="# Full docs", raise_for_status=lambda: None)

        with patch.object(indexer.httpx, "get", return_value=response) as get:
            text = indexer.download_docs()

        self.assertEqual(text, "# Full docs")
        get.assert_called_once_with(indexer.DOCS_URL, timeout=60, follow_redirects=True)

    def test_download_docs_loads_llms_txt_markdown_documents(self):
        def fake_get(url, timeout, follow_redirects):
            pages = {
                "https://example.test/llms.txt": "# Docs\n- [Intro](/intro.md)\n- [API](https://example.test/api.mdx)",
                "https://example.test/intro.md": "# Intro\nHello",
                "https://example.test/api.mdx": "# API\nReference",
            }
            return SimpleNamespace(text=pages[url], raise_for_status=lambda: None)

        with patch.object(indexer.httpx, "get", side_effect=fake_get):
            text = indexer.download_docs("https://example.test/llms.txt", source_limit=2)

        self.assertIn("# Intro\nSource: https://example.test/intro.md\nHello", text)
        self.assertIn("# API\nSource: https://example.test/api.mdx\nReference", text)

    def test_llms_txt_url_detection_allows_query_strings(self):
        self.assertTrue(indexer._is_llms_index_url("https://example.test/llms.txt?token=abc"))

    def test_loaded_document_format_keeps_source_inside_first_heading_section(self):
        text = indexer._format_loaded_documents(
            [
                Document(source="https://example.test/intro.md", content="# Intro\nHello"),
                Document(source="https://example.test/api.md", content="# API\nReference"),
            ]
        )

        self.assertIn("# Intro\nSource: https://example.test/intro.md", text)
        self.assertIn("# API\nSource: https://example.test/api.md", text)
        chunks = indexer.chunk_docs(text)

        self.assertEqual(chunks[0]["source"], "https://example.test/intro.md")
        self.assertEqual(chunks[-1]["source"], "https://example.test/api.md")


if __name__ == "__main__":
    unittest.main()
