from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable
from urllib.parse import urljoin, urlsplit


@dataclass(frozen=True)
class Document:
    source: str
    content: str


_MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\(([^)\s]+)(?:\s+[^)]*)?\)")


def extract_markdown_links(text: str, base_url: str) -> list[str]:
    links: list[str] = []
    seen: set[str] = set()

    for match in _MARKDOWN_LINK_RE.finditer(text):
        href = match.group(1).strip("<>")
        resolved = urljoin(base_url, href)
        parsed = urlsplit(resolved)

        if parsed.scheme not in {"http", "https"}:
            continue
        url_without_fragment = resolved.split("#", 1)[0]
        if not url_without_fragment.lower().endswith((".md", ".mdx")):
            continue
        if resolved in seen:
            continue

        seen.add(resolved)
        links.append(resolved)

    return links


def fetch_documents(
    index_url: str,
    fetch_text: Callable[[str], str],
    limit: int | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> list[Document]:
    index_text = fetch_text(index_url)
    links = extract_markdown_links(index_text, index_url)
    if limit is not None:
        links = links[:limit]

    documents: list[Document] = []
    total = len(links)
    for index, link in enumerate(links, start=1):
        if on_progress:
            on_progress(index, total, link)
        content = fetch_text(link)
        if not content.strip():
            continue
        documents.append(Document(source=link, content=content))

    return documents
