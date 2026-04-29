import unittest

from langchain_rag_mcp.loaders import extract_markdown_links, fetch_documents


class LoaderTests(unittest.TestCase):
    def test_extract_markdown_links_filters_resolves_and_deduplicates(self):
        text = """
        [Guide](/docs/guide.md)
        [Guide duplicate](/docs/guide.md)
        [Reference](reference/api.mdx#section)
        [External](https://example.com/remote.md?download=1)
        [Not markdown](https://example.com/page.html)
        [Mail](mailto:docs@example.com/readme.md)
        [Anchor after md](https://example.com/readme.md#intro)
        """

        links = extract_markdown_links(text, "https://docs.example.com/llms.txt")

        self.assertEqual(
            links,
            [
                "https://docs.example.com/docs/guide.md",
                "https://docs.example.com/reference/api.mdx#section",
                "https://example.com/readme.md#intro",
            ],
        )

    def test_fetch_documents_fetches_index_and_non_empty_docs_up_to_limit(self):
        calls = []
        pages = {
            "https://docs.example.com/llms.txt": """
            [First](first.md)
            [Empty](empty.md)
            [Second](/second.mdx)
            [Third](https://other.example.com/third.md)
            """,
            "https://docs.example.com/first.md": "first content",
            "https://docs.example.com/empty.md": "   ",
            "https://docs.example.com/second.mdx": "second content",
            "https://other.example.com/third.md": "third content",
        }

        def fetch_text(url):
            calls.append(url)
            return pages[url]

        documents = fetch_documents(
            "https://docs.example.com/llms.txt",
            fetch_text,
            limit=3,
        )

        self.assertEqual(
            [(doc.source, doc.content) for doc in documents],
            [
                ("https://docs.example.com/first.md", "first content"),
                ("https://docs.example.com/second.mdx", "second content"),
            ],
        )
        self.assertEqual(
            calls,
            [
                "https://docs.example.com/llms.txt",
                "https://docs.example.com/first.md",
                "https://docs.example.com/empty.md",
                "https://docs.example.com/second.mdx",
            ],
        )

    def test_fetch_documents_reports_progress_for_each_link(self):
        pages = {
            "https://docs.example.com/llms.txt": "[First](first.md)\n[Second](second.md)",
            "https://docs.example.com/first.md": "first content",
            "https://docs.example.com/second.md": "second content",
        }
        progress = []

        fetch_documents(
            "https://docs.example.com/llms.txt",
            lambda url: pages[url],
            on_progress=lambda current, total, url: progress.append((current, total, url)),
        )

        self.assertEqual(
            progress,
            [
                (1, 2, "https://docs.example.com/first.md"),
                (2, 2, "https://docs.example.com/second.md"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
