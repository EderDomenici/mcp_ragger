import time
import re
import argparse
import sys
import httpx
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))

from langchain_rag_mcp.loaders import fetch_documents

QDRANT_URL = "http://localhost:6333"
LLAMACPP_URL = "http://localhost:8080/v1/embeddings"
COLLECTION = "langchain_docs"
DOCS_URL = "https://docs.langchain.com/llms-full.txt"

MAX_CHUNK_CHARS = 2600  # chunk por pagina/fonte, mas ainda especifico o bastante
OVERLAP_CHARS = 120     # continuidade pequena entre chunks da mesma pagina
EMBED_INPUT_CAP = 1000  # embedding enxuto; payload continua completo
EMBED_BATCH = 32        # chunks por chamada HTTP; reduz adaptativamente se necessario
UPSERT_BATCH = 100      # pontos por upsert no Qdrant

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
SOURCE_RE = re.compile(r"^Source:\s*(.+?)\s*$", re.IGNORECASE)
SYMBOL_RE = re.compile(
    r"`([^`\n]{2,80})`|"
    r"\b([A-Z][A-Za-z0-9]+(?:[A-Z][A-Za-z0-9]+)+)\b|"
    r"\b([a-zA-Z_][\w.]{2,80}\()"
)


@dataclass
class Section:
    index: int
    title: str
    source: str
    level: int
    breadcrumb: list[str]
    lines: list[str]


@dataclass
class Chunk:
    id: int
    chunk_id: str
    title: str
    source: str
    content: str
    breadcrumb: str
    heading_level: int
    section_index: int
    chunk_index: int
    chunk_total: int = 1
    content_type: str = "concept"
    has_code: bool = False
    symbols: list[str] = field(default_factory=list)
    char_count: int = 0
    token_est: int = 0
    prev_chunk_id: str = ""
    next_chunk_id: str = ""


def _fetch_text(url: str) -> str:
    response = httpx.get(url, timeout=60, follow_redirects=True)
    response.raise_for_status()
    return response.text


def _is_llms_index_url(source_url: str) -> bool:
    return urlsplit(source_url).path.endswith("llms.txt")


def _format_loaded_document(document) -> str:
    content = document.content.strip()
    lines = content.splitlines()
    for index, line in enumerate(lines):
        if HEADING_RE.match(line):
            return "\n".join(
                lines[: index + 1]
                + [f"Source: {document.source}"]
                + lines[index + 1 :]
            ).strip()

    title = document.source.rstrip("/").rsplit("/", 1)[-1] or document.source
    return f"# {title}\nSource: {document.source}\n\n{content}"


def _format_loaded_documents(documents) -> str:
    return "\n\n---\n\n".join(_format_loaded_document(document) for document in documents)


def download_docs(source_url: str = DOCS_URL, source_limit: int | None = None) -> str:
    print(f"Downloading {source_url} ...", flush=True)
    if _is_llms_index_url(source_url):
        documents = fetch_documents(
            source_url,
            _fetch_text,
            limit=source_limit,
            on_progress=lambda current, total, url: print(
                f"  fetching linked doc {current}/{total}: {url}",
                flush=True,
            ),
        )
        return _format_loaded_documents(documents)

    return _fetch_text(source_url)


def _extract_source(lines: list[str]) -> str:
    for line in lines[:12]:
        match = SOURCE_RE.match(line.strip())
        if match:
            return match.group(1).strip()
    return ""


def _iter_sections(text: str) -> list[Section]:
    sections: list[Section] = []
    heading_stack: list[tuple[int, str]] = []
    current_lines: list[str] = []
    current_title = "LangChain documentation"
    current_level = 1
    last_source = ""
    in_fence = False

    def flush() -> None:
        nonlocal current_lines, current_title, current_level, last_source
        body = "\n".join(current_lines).strip()
        if not body:
            current_lines = []
            return

        breadcrumb = [title for _, title in heading_stack if title]
        source = _extract_source(current_lines) or last_source
        if source:
            last_source = source
        sections.append(
            Section(
                index=len(sections),
                title=current_title,
                source=source,
                level=current_level,
                breadcrumb=breadcrumb or [current_title],
                lines=body.splitlines(),
            )
        )
        current_lines = []

    for raw_line in text.replace("\r\n", "\n").splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence

        heading = HEADING_RE.match(line) if not in_fence else None
        if heading:
            flush()
            level = len(heading.group(1))
            title = heading.group(2).strip()
            heading_stack = [(l, t) for l, t in heading_stack if l < level]
            heading_stack.append((level, title))
            current_title = title
            current_level = level

        current_lines.append(line)

    flush()
    return sections


def _split_oversized_block(block: str) -> list[str]:
    if len(block) <= MAX_CHUNK_CHARS:
        return [block]

    pieces: list[str] = []
    buf: list[str] = []
    size = 0
    for line in block.splitlines():
        if len(line) > MAX_CHUNK_CHARS:
            if buf:
                pieces.append("\n".join(buf).strip())
                buf = []
                size = 0
            for start in range(0, len(line), MAX_CHUNK_CHARS):
                pieces.append(line[start : start + MAX_CHUNK_CHARS].strip())
            continue

        line_size = len(line) + 1
        if buf and size + line_size > MAX_CHUNK_CHARS:
            pieces.append("\n".join(buf).strip())
            buf = []
            size = 0
        buf.append(line)
        size += line_size

    if buf:
        pieces.append("\n".join(buf).strip())
    return [p for p in pieces if p]


def _semantic_blocks(lines: list[str]) -> list[str]:
    blocks: list[str] = []
    buf: list[str] = []
    in_fence = False

    def flush() -> None:
        nonlocal buf
        block = "\n".join(buf).strip()
        if block:
            blocks.extend(_split_oversized_block(block))
        buf = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            buf.append(line)
            if not in_fence:
                flush()
            continue

        if not in_fence and not stripped:
            flush()
            continue

        if not in_fence and HEADING_RE.match(line):
            flush()

        buf.append(line)

    flush()
    return blocks


def _overlap_blocks(blocks: list[str]) -> list[str]:
    overlap: list[str] = []
    size = 0
    for block in reversed(blocks):
        block_size = len(block) + 2
        if block_size > OVERLAP_CHARS:
            continue
        if overlap and size + block_size > OVERLAP_CHARS:
            break
        overlap.insert(0, block)
        size += block_size
    return overlap


def _pack_blocks(blocks: list[str]) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_size = 0

    for block in blocks:
        block_size = len(block) + 2
        if current and current_size + block_size > MAX_CHUNK_CHARS:
            chunks.append("\n\n".join(current).strip())
            current = _overlap_blocks(current)
            current_size = sum(len(item) + 2 for item in current)

        current.append(block)
        current_size += block_size

    if current:
        chunks.append("\n\n".join(current).strip())

    return chunks


def _group_sections_by_source(sections: list[Section]) -> list[list[Section]]:
    groups: list[list[Section]] = []
    current: list[Section] = []
    current_source = ""

    for section in sections:
        source = section.source or current_source or "document"
        if current and source != current_source:
            groups.append(current)
            current = []
        current.append(section)
        current_source = source

    if current:
        groups.append(current)
    return groups


def _first_heading(content: str, fallback: str) -> str:
    for line in content.splitlines():
        match = HEADING_RE.match(line)
        if match:
            return match.group(2).strip()
    return fallback


def _extract_symbols(content: str) -> list[str]:
    symbols: list[str] = []
    for match in SYMBOL_RE.finditer(content):
        symbol = next(group for group in match.groups() if group)
        symbol = symbol.rstrip("(").strip()
        if symbol and symbol not in symbols:
            symbols.append(symbol)
        if len(symbols) >= 20:
            break
    return symbols


def _classify_content(title: str, content: str, has_code: bool) -> str:
    haystack = f"{title}\n{content}".lower()
    if has_code:
        return "example"
    if any(term in haystack for term in (" api ", " reference", " class ", " method ", " parameter")):
        return "api_reference"
    if any(term in haystack for term in ("how to", "guide", "tutorial", "setup", "deploy", "configure")):
        return "how_to"
    return "concept"


def _content_with_context(title: str, breadcrumb: str, source: str, symbols: list[str], content: str) -> str:
    header = [f"# {title}"]
    if breadcrumb and breadcrumb != title:
        header.append(f"Breadcrumb: {breadcrumb}")
    if source:
        header.append(f"Source: {source}")
    if symbols:
        header.append("Symbols: " + ", ".join(symbols[:12]))
    header.append("")
    header.append(content)
    return "\n".join(header)


def chunk_docs(text: str) -> list[dict]:
    sections = _iter_sections(text)
    groups = _group_sections_by_source(sections)
    chunks: list[Chunk] = []

    for group_index, group in enumerate(groups):
        source = group[0].source
        fallback_title = group[0].title
        breadcrumb = " > ".join(group[0].breadcrumb)
        blocks: list[str] = []
        for section in group:
            blocks.extend(_semantic_blocks(section.lines))

        packed = _pack_blocks(blocks)
        for part_index, content in enumerate(packed):
            has_code = "```" in content
            title = _first_heading(content, fallback_title)
            symbols = _extract_symbols(content)
            chunk = Chunk(
                id=len(chunks),
                chunk_id=f"{group_index:05d}-{part_index:02d}",
                title=title,
                source=source,
                content=_content_with_context(title, breadcrumb, source, symbols, content),
                breadcrumb=breadcrumb,
                heading_level=group[0].level,
                section_index=group_index,
                chunk_index=part_index,
                content_type=_classify_content(title, content, has_code),
                has_code=has_code,
                symbols=symbols,
                char_count=len(content),
                token_est=max(1, len(content) // 4),
            )
            chunks.append(chunk)

        total = len(packed)
        for chunk in chunks[-total:]:
            chunk.chunk_total = total

    for i, chunk in enumerate(chunks):
        if i > 0:
            chunk.prev_chunk_id = chunks[i - 1].chunk_id
        if i + 1 < len(chunks):
            chunk.next_chunk_id = chunks[i + 1].chunk_id

    return [chunk.__dict__ for chunk in chunks]


def embed_text(chunk: dict) -> str:
    metadata = [
        f"Title: {chunk['title']}",
        f"Breadcrumb: {chunk['breadcrumb']}",
        f"Type: {chunk['content_type']}",
    ]
    if chunk.get("source"):
        metadata.append(f"Source: {chunk['source']}")
    if chunk.get("symbols"):
        metadata.append("Symbols: " + ", ".join(chunk["symbols"][:8]))
    metadata.append("")
    metadata.append(chunk["content"])
    return "\n".join(metadata)


def embed_batch(texts: list[str]) -> list[list[float]]:
    capped = [t[:EMBED_INPUT_CAP] for t in texts]
    try:
        response = httpx.post(
            LLAMACPP_URL,
            json={"input": capped},
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()["data"]
        data.sort(key=lambda x: x["index"])
        return [item["embedding"] for item in data]
    except httpx.HTTPStatusError:
        if len(capped) == 1:
            return [_embed_one_with_fallback(capped[0])]

        midpoint = len(capped) // 2
        return embed_batch(capped[:midpoint]) + embed_batch(capped[midpoint:])


def _embed_one_with_fallback(text: str) -> list[float]:
    for cap in (800, 500, 300):
        response = httpx.post(
            LLAMACPP_URL,
            json={"input": [text[:cap]]},
            timeout=120,
        )
        if response.status_code < 500:
            response.raise_for_status()
            return response.json()["data"][0]["embedding"]
    response.raise_for_status()
    raise RuntimeError("Embedding fallback failed")


def _print_chunk_stats(chunks: list[dict]) -> None:
    avg_chars = sum(c["char_count"] for c in chunks) / len(chunks)
    code_chunks = sum(1 for c in chunks if c["has_code"])
    max_chars = max(c["char_count"] for c in chunks)
    print(f"  {len(chunks)} chunks found")
    print(f"  avg chars/chunk: {avg_chars:.0f} | max chars: {max_chars} | code chunks: {code_chunks}")


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats-only", action="store_true", help="Baixa e chunkifica sem recriar o indice.")
    parser.add_argument("--limit", type=int, default=0, help="Indexa apenas os N primeiros chunks para teste.")
    parser.add_argument("--source-url", default=DOCS_URL, help="URL fonte: llms-full.txt, llms.txt, .md ou .mdx.")
    parser.add_argument("--source-limit", type=int, default=0, help="Limita documentos baixados de um llms.txt.")
    args = parser.parse_args()

    qdrant = QdrantClient(url=QDRANT_URL)

    text = download_docs(args.source_url, source_limit=args.source_limit or None)

    print("Chunking docs...")
    chunks = chunk_docs(text)
    _print_chunk_stats(chunks)
    if args.stats_only:
        return
    if args.limit:
        chunks = chunks[: args.limit]
        print(f"  limiting index to {len(chunks)} chunks")

    print("Getting embedding dimension...")
    dim = len(embed_batch(["test"])[0])
    print(f"  Embedding dim: {dim}")

    print(f"Creating collection '{COLLECTION}'...")
    if qdrant.collection_exists(COLLECTION):
        qdrant.delete_collection(COLLECTION)
    qdrant.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    print(f"Embedding {len(chunks)} chunks (batch={EMBED_BATCH})...")
    points: list[PointStruct] = []
    t0 = time.perf_counter()

    for start in range(0, len(chunks), EMBED_BATCH):
        batch = chunks[start : start + EMBED_BATCH]
        vectors = embed_batch([embed_text(c) for c in batch])

        for j, (chunk, vector) in enumerate(zip(batch, vectors)):
            points.append(
                PointStruct(
                    id=chunk["id"],
                    vector=vector,
                    payload=chunk,
                )
            )

        done = min(start + EMBED_BATCH, len(chunks))
        elapsed = time.perf_counter() - t0
        rate = done / elapsed
        eta = (len(chunks) - done) / rate if rate > 0 else 0
        print(f"  Embedded {done}/{len(chunks)}  {rate:.0f} chunks/s  ETA {eta:.0f}s", end="\r", flush=True)

    print(f"\nUploading to Qdrant...")
    for start in range(0, len(points), UPSERT_BATCH):
        qdrant.upsert(collection_name=COLLECTION, points=points[start : start + UPSERT_BATCH])

    total = time.perf_counter() - t0
    print(f"Done — {len(points)} chunks indexed in {total:.0f}s")


if __name__ == "__main__":
    run()
