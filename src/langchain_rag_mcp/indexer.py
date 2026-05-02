import time
import re
import argparse
import sys
import httpx
import json
import hashlib
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))

from langchain_rag_mcp.config import Settings
from langchain_rag_mcp.embeddings import create_embeddings
from langchain_rag_mcp.loaders import fetch_documents

DOCS_URL = Settings.docs_url

MAX_CHUNK_CHARS = 2600  # chunk por pagina/fonte, mas ainda especifico o bastante
OVERLAP_CHARS = 120     # continuidade pequena entre chunks da mesma pagina
EMBED_INPUT_CAP = 1000  # embedding enxuto; payload continua completo
EMBED_BATCH = 100       # chunks por chamada HTTP (Gemini suporta até 100 por request)
EMBED_WORKERS = 3       # sequencial por padrão; aumentar só com rate limit generoso confirmado
UPSERT_BATCH = 100      # pontos por upsert no Qdrant
CHECKPOINT_PATH = ROOT / "data" / "indexer_state.json"
EMBED_RETRY_ATTEMPTS = 5
EMBED_RETRY_BACKOFF_SECONDS = 30.0

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


def download_docs(source_url: str | None = None, source_limit: int | None = None) -> str:
    source_url = source_url or Settings.docs_url
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


def embed_batch(texts: list[str], embeddings) -> list[list[float]]:
    capped = [t[:EMBED_INPUT_CAP] for t in texts]
    if not _is_llamacpp_embeddings(embeddings):
        vectors = embeddings.embed_documents(capped)
        if len(vectors) == len(capped):
            return vectors
        if len(capped) == 1:
            raise RuntimeError(f"Embedding returned {len(vectors)} vectors for 1 chunk")
        midpoint = len(capped) // 2
        return embed_batch(capped[:midpoint], embeddings) + embed_batch(capped[midpoint:], embeddings)

    try:
        vectors = embeddings.embed_documents(capped)
        if len(vectors) == len(capped):
            return vectors
        if len(capped) == 1:
            raise RuntimeError(f"Embedding returned {len(vectors)} vectors for 1 chunk")
        midpoint = len(capped) // 2
        return embed_batch(capped[:midpoint], embeddings) + embed_batch(capped[midpoint:], embeddings)
    except httpx.HTTPStatusError:
        if len(capped) == 1:
            return [_embed_one_with_fallback(capped[0], embeddings)]

        midpoint = len(capped) // 2
        return embed_batch(capped[:midpoint], embeddings) + embed_batch(capped[midpoint:], embeddings)


def _is_llamacpp_embeddings(embeddings) -> bool:
    return embeddings.__class__.__name__ == "LlamaCppEmbeddings"


def _embed_one_with_fallback(text: str, embeddings) -> list[float]:
    last_error = None
    for cap in (800, 500, 300):
        try:
            return embeddings.embed_query(text[:cap])
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code < 500:
                raise
            last_error = exc
    if last_error:
        raise last_error
    raise RuntimeError("Embedding fallback failed")


def _retry_after_seconds(exc: BaseException) -> float | None:
    response = getattr(exc, "response", None)
    if response is None:
        return None
    retry_after = response.headers.get("Retry-After")
    if not retry_after:
        return None
    try:
        return float(retry_after)
    except ValueError:
        return None


def _is_retryable_quota_error(exc: BaseException) -> bool:
    response = getattr(exc, "response", None)
    if response is not None and getattr(response, "status_code", None) == 429:
        return True
    text = str(exc)
    return "429" in text or "RESOURCE_EXHAUSTED" in text


def embed_batch_with_retry(
    texts: list[str],
    embeddings,
    *,
    max_attempts: int = EMBED_RETRY_ATTEMPTS,
    initial_backoff_seconds: float = EMBED_RETRY_BACKOFF_SECONDS,
    sleep=time.sleep,
) -> list[list[float]]:
    for attempt in range(1, max_attempts + 1):
        try:
            return embed_batch(texts, embeddings)
        except Exception as exc:
            if attempt >= max_attempts or not _is_retryable_quota_error(exc):
                raise
            base = _retry_after_seconds(exc) or initial_backoff_seconds * (2 ** (attempt - 1))
            wait_seconds = base * (0.75 + random.random() * 0.5)
            print(
                f"\nEmbedding quota/rate limit hit; retrying in {wait_seconds:.0f}s "
                f"(attempt {attempt + 1}/{max_attempts})...",
                flush=True,
            )
            sleep(wait_seconds)

    raise RuntimeError("Embedding retry loop exited unexpectedly")


def load_checkpoint(path: Path) -> dict:
    if not path.exists():
        return {"completed_chunk_ids": []}
    return json.loads(path.read_text(encoding="utf-8"))


def save_checkpoint(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = {
        "completed_chunk_ids": sorted(set(state.get("completed_chunk_ids", []))),
    }
    if state.get("metadata"):
        normalized["metadata"] = state["metadata"]
    path.write_text(json.dumps(normalized, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_points(chunks: list[dict], vectors: list[list[float]]) -> list[PointStruct]:
    if len(vectors) != len(chunks):
        raise RuntimeError(f"Embedding returned {len(vectors)} vectors for {len(chunks)} chunks")
    return [
        PointStruct(
            id=chunk["id"],
            vector=vector,
            payload=chunk,
        )
        for chunk, vector in zip(chunks, vectors)
    ]


def _iter_batches(items: list, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def chunks_fingerprint(chunks: list[dict]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(str(chunk.get("id", "")).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(chunk.get("chunk_id", "")).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(chunk.get("source", "")).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(chunk.get("content", "")).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def checkpoint_metadata(
    *,
    collection_name: str,
    source_url: str,
    embedding_provider: str,
    embedding_model: str,
    embedding_dim: int,
    chunks: list[dict],
) -> dict:
    return {
        "collection_name": collection_name,
        "source_url": source_url,
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model,
        "embedding_dim": embedding_dim,
        "chunk_count": len(chunks),
        "chunks_fingerprint": chunks_fingerprint(chunks),
    }


def _validate_checkpoint_metadata(state: dict, expected_metadata: dict | None) -> None:
    if not expected_metadata:
        return
    actual_metadata = state.get("metadata")
    if actual_metadata and actual_metadata != expected_metadata:
        raise RuntimeError(
            "Existing checkpoint does not match this indexing run. "
            "Start without --resume to rebuild the collection."
        )


def _collection_vector_size(collection_info) -> int | None:
    current = collection_info
    for attr in ("config", "params", "vectors"):
        current = current.get(attr) if isinstance(current, dict) else getattr(current, attr, None)
        if current is None:
            return None
    if isinstance(current, dict):
        value = current.get("size")
    else:
        value = getattr(current, "size", None)
    return value if isinstance(value, int) else None


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _progress_line(
    *,
    indexed: int,
    total: int,
    batch_count: int,
    batch_seconds: float,
    elapsed_seconds: float,
    start_indexed: int,
) -> str:
    run_indexed = max(0, indexed - start_indexed)
    avg_seconds_per_chunk = elapsed_seconds / run_indexed if run_indexed else 0.0
    batch_seconds_per_chunk = batch_seconds / batch_count if batch_count else 0.0
    eta_seconds = (total - indexed) * avg_seconds_per_chunk if avg_seconds_per_chunk else 0.0
    return (
        f"  Embedded+upserted {indexed}/{total} | "
        f"batch {_format_duration(batch_seconds)} ({batch_seconds_per_chunk:.2f}s/chunk) | "
        f"avg {avg_seconds_per_chunk:.2f}s/chunk | "
        f"ETA {_format_duration(eta_seconds)}"
    )


def index_chunks_incrementally(
    chunks: list[dict],
    embeddings,
    qdrant,
    *,
    collection_name: str,
    checkpoint_path: Path,
    embed_batch_size: int = EMBED_BATCH,
    upsert_batch_size: int = UPSERT_BATCH,
    embed_workers: int = EMBED_WORKERS,
    embed_batch_fn=embed_batch_with_retry,
    checkpoint_metadata: dict | None = None,
) -> int:
    state = load_checkpoint(checkpoint_path)
    _validate_checkpoint_metadata(state, checkpoint_metadata)
    completed_ids = set(state.get("completed_chunk_ids", []))
    pending = [chunk for chunk in chunks if chunk["id"] not in completed_ids]
    if completed_ids:
        print(f"  Resuming: {len(completed_ids)} chunks already upserted, {len(pending)} pending")

    indexed = len(completed_ids)
    start_indexed = indexed
    t0 = time.perf_counter()
    total = len(chunks)

    checkpoint_lock = threading.Lock()

    def _embed_and_upsert(batch: list[dict]) -> int:
        nonlocal indexed
        batch_t0 = time.perf_counter()
        vectors = embed_batch_fn([embed_text(c) for c in batch], embeddings)
        points = _build_points(batch, vectors)

        with checkpoint_lock:
            for upsert_points in _iter_batches(points, upsert_batch_size):
                qdrant.upsert(collection_name=collection_name, points=upsert_points)
                completed_ids.update(point.id for point in upsert_points)
            save_checkpoint(
                checkpoint_path,
                {
                    "completed_chunk_ids": list(completed_ids),
                    "metadata": checkpoint_metadata,
                },
            )
            indexed = len(completed_ids)

        return int(time.perf_counter() - batch_t0)

    batches = list(_iter_batches(pending, embed_batch_size))
    with ThreadPoolExecutor(max_workers=embed_workers) as executor:
        futures = {executor.submit(_embed_and_upsert, batch): batch for batch in batches}
        for future in as_completed(futures):
            batch_seconds = future.result()
            elapsed = time.perf_counter() - t0
            batch = futures[future]
            with checkpoint_lock:
                current_indexed = indexed
            print(
                _progress_line(
                    indexed=current_indexed,
                    total=total,
                    batch_count=len(batch),
                    batch_seconds=batch_seconds,
                    elapsed_seconds=elapsed,
                    start_indexed=start_indexed,
                ),
                end="\r",
                flush=True,
            )

    return indexed


def ensure_collection(qdrant, collection_name: str, dim: int, *, resume: bool) -> None:
    exists = qdrant.collection_exists(collection_name)
    if resume and not exists:
        raise RuntimeError(
            f"Cannot resume indexing because collection '{collection_name}' does not exist. "
            "Run without --resume to rebuild it."
        )
    if resume and hasattr(qdrant, "get_collection"):
        vector_size = _collection_vector_size(qdrant.get_collection(collection_name))
        if vector_size is not None and vector_size != dim:
            raise RuntimeError(
                f"Cannot resume indexing because collection '{collection_name}' has vector size "
                f"{vector_size}, but current embeddings produce {dim} dimensions."
            )
    if exists and not resume:
        qdrant.delete_collection(collection_name)
        exists = False
    if not exists:
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )


def _print_chunk_stats(chunks: list[dict]) -> None:
    avg_chars = sum(c["char_count"] for c in chunks) / len(chunks)
    code_chunks = sum(1 for c in chunks if c["has_code"])
    max_chars = max(c["char_count"] for c in chunks)
    print(f"  {len(chunks)} chunks found")
    print(f"  avg chars/chunk: {avg_chars:.0f} | max chars: {max_chars} | code chunks: {code_chunks}")


def main():
    settings = Settings.from_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats-only", action="store_true", help="Baixa e chunkifica sem recriar o indice.")
    parser.add_argument("--limit", type=int, default=0, help="Indexa apenas os N primeiros chunks para teste.")
    parser.add_argument("--source-url", default=settings.docs_url, help="URL fonte: llms-full.txt, llms.txt, .md ou .mdx.")
    parser.add_argument("--source-limit", type=int, default=0, help="Limita documentos baixados de um llms.txt.")
    parser.add_argument("--embedding-provider", default=settings.embedding_provider)
    parser.add_argument("--google-embedding-model", default=settings.google_embedding_model)
    parser.add_argument("--google-embedding-dimensions", type=int, default=settings.google_output_dimensionality)
    parser.add_argument("--llamacpp-url", default=settings.llamacpp_url)
    parser.add_argument("--embed-batch", type=int, default=EMBED_BATCH, help="Chunks por chamada de embedding.")
    parser.add_argument("--embed-workers", type=int, default=EMBED_WORKERS, help="Requests de embedding simultâneos.")
    parser.add_argument("--resume", action="store_true", help="Continua do ultimo batch enviado ao Qdrant.")
    parser.add_argument("--checkpoint-path", type=Path, default=CHECKPOINT_PATH)
    args = parser.parse_args()

    qdrant = QdrantClient(**settings.qdrant_client_kwargs())
    embeddings = create_embeddings(
        args.embedding_provider,
        llamacpp_url=args.llamacpp_url,
        google_model=args.google_embedding_model,
        google_output_dimensionality=args.google_embedding_dimensions,
    )

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
    dim = len(embed_batch_with_retry(["test"], embeddings)[0])
    print(f"  Embedding dim: {dim}")

    print(f"Creating collection '{settings.collection}'...")
    ensure_collection(qdrant, settings.collection, dim, resume=args.resume)
    if not args.resume and args.checkpoint_path.exists():
        args.checkpoint_path.unlink()

    print(f"Embedding and uploading {len(chunks)} chunks (batch={args.embed_batch})...")
    t0 = time.perf_counter()
    indexed = index_chunks_incrementally(
        chunks,
        embeddings,
        qdrant,
        collection_name=settings.collection,
        checkpoint_path=args.checkpoint_path,
        embed_batch_size=args.embed_batch,
        upsert_batch_size=UPSERT_BATCH,
        embed_workers=args.embed_workers,
        checkpoint_metadata=checkpoint_metadata(
            collection_name=settings.collection,
            source_url=args.source_url,
            embedding_provider=args.embedding_provider,
            embedding_model=args.google_embedding_model,
            embedding_dim=dim,
            chunks=chunks,
        ),
    )

    total = time.perf_counter() - t0
    print(f"\nDone — {indexed} chunks indexed in {total:.0f}s")


if __name__ == "__main__":
    main()
