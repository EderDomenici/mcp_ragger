import tempfile
import unittest
from pathlib import Path

import httpx

import indexer


def _chunk(chunk_id: int, content: str) -> dict:
    return {
        "id": chunk_id,
        "chunk_id": f"chunk-{chunk_id}",
        "title": f"Chunk {chunk_id}",
        "breadcrumb": "Docs",
        "content_type": "concept",
        "content": content,
    }


class FakeEmbeddings:
    def __init__(self):
        self.calls: list[list[str]] = []

    def embed_documents(self, texts):
        self.calls.append(list(texts))
        return [[float(len(self.calls)), float(index)] for index, _ in enumerate(texts)]


class FakeQdrant:
    def __init__(self):
        self.upserts: list[list[int]] = []

    def upsert(self, collection_name, points):
        self.upserts.append([point.id for point in points])


class FlakyQuotaEmbeddings:
    def __init__(self):
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += 1
        if self.calls == 1:
            request = httpx.Request("POST", "https://example.test/embed")
            response = httpx.Response(429, request=request)
            raise httpx.HTTPStatusError("quota exhausted", request=request, response=response)
        return [[1.0] for _ in texts]


class CollapsingBatchEmbeddings:
    def __init__(self):
        self.calls: list[list[str]] = []

    def embed_documents(self, texts):
        self.calls.append(list(texts))
        if len(texts) > 1:
            return [[99.0]]
        return [[float(len(self.calls))]]


class IndexerIncrementalTests(unittest.TestCase):
    def test_checkpoint_state_can_be_saved_loaded_and_resume_skips_completed_chunks(self):
        chunks = [_chunk(0, "already indexed"), _chunk(1, "also indexed"), _chunk(2, "new chunk")]
        embeddings = FakeEmbeddings()
        qdrant = FakeQdrant()

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "indexer-checkpoint.json"
            indexer.save_checkpoint(checkpoint_path, {"completed_chunk_ids": [0, 1]})

            self.assertEqual(indexer.load_checkpoint(checkpoint_path), {"completed_chunk_ids": [0, 1]})

            indexer.index_chunks_incrementally(
                chunks,
                embeddings,
                qdrant,
                collection_name="docs",
                checkpoint_path=checkpoint_path,
                embed_batch_size=2,
                upsert_batch_size=2,
            )

            self.assertEqual(qdrant.upserts, [[2]])
            self.assertEqual(len(embeddings.calls), 1)
            self.assertIn("new chunk", embeddings.calls[0][0])
            self.assertNotIn("already indexed", embeddings.calls[0][0])
            self.assertEqual(indexer.load_checkpoint(checkpoint_path), {"completed_chunk_ids": [0, 1, 2]})

    def test_incremental_indexing_upserts_each_embedded_batch_immediately(self):
        chunks = [_chunk(0, "first"), _chunk(1, "second"), _chunk(2, "third")]
        embeddings = FakeEmbeddings()
        events: list[tuple[str, int | list[int]]] = []

        def recording_embed_batch(texts, embeddings):
            events.append(("embed", len(texts)))
            return embeddings.embed_documents(texts)

        class RecordingQdrant(FakeQdrant):
            def upsert(self, collection_name, points):
                ids = [point.id for point in points]
                events.append(("upsert", ids))
                super().upsert(collection_name, points)

        with tempfile.TemporaryDirectory() as tmpdir:
            indexer.index_chunks_incrementally(
                chunks,
                embeddings,
                RecordingQdrant(),
                collection_name="docs",
                checkpoint_path=Path(tmpdir) / "checkpoint.json",
                embed_batch_size=2,
                upsert_batch_size=100,
                embed_batch_fn=recording_embed_batch,
            )

        self.assertEqual(
            events,
            [
                ("embed", 2),
                ("upsert", [0, 1]),
                ("embed", 1),
                ("upsert", [2]),
            ],
        )

    def test_embed_batch_with_retry_retries_429_quota_error_before_succeeding(self):
        embeddings = FlakyQuotaEmbeddings()
        sleeps: list[float] = []

        vectors = indexer.embed_batch_with_retry(
            ["hello"],
            embeddings,
            max_attempts=2,
            initial_backoff_seconds=0.25,
            sleep=sleeps.append,
        )

        self.assertEqual(vectors, [[1.0]])
        self.assertEqual(embeddings.calls, 2)
        self.assertEqual(len(sleeps), 1)
        self.assertGreaterEqual(sleeps[0], 0.25 * 0.75)
        self.assertLessEqual(sleeps[0], 0.25 * 1.25)

    def test_embed_batch_splits_when_provider_returns_too_few_vectors(self):
        embeddings = CollapsingBatchEmbeddings()

        vectors = indexer.embed_batch(["first", "second", "third", "fourth"], embeddings)

        self.assertEqual(len(vectors), 4)
        self.assertEqual(embeddings.calls[0], ["first", "second", "third", "fourth"])
        self.assertIn(["first"], embeddings.calls)
        self.assertIn(["fourth"], embeddings.calls)

    def test_resume_rejects_checkpoint_for_different_corpus(self):
        chunks = [_chunk(0, "current")]

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.json"
            indexer.save_checkpoint(
                checkpoint_path,
                {
                    "completed_chunk_ids": [0],
                    "metadata": {"collection_name": "docs", "chunks_fingerprint": "old"},
                },
            )

            with self.assertRaisesRegex(RuntimeError, "checkpoint does not match"):
                indexer.index_chunks_incrementally(
                    chunks,
                    FakeEmbeddings(),
                    FakeQdrant(),
                    collection_name="docs",
                    checkpoint_path=checkpoint_path,
                    checkpoint_metadata={
                        "collection_name": "docs",
                        "chunks_fingerprint": indexer.chunks_fingerprint(chunks),
                    },
                )

    def test_build_points_rejects_vector_count_mismatch(self):
        with self.assertRaisesRegex(RuntimeError, "Embedding returned 1 vectors for 2 chunks"):
            indexer._build_points([_chunk(0, "first"), _chunk(1, "second")], [[1.0]])

    def test_resume_requires_existing_collection(self):
        class MissingCollectionQdrant:
            def collection_exists(self, collection_name):
                return False

        with self.assertRaisesRegex(RuntimeError, "Cannot resume"):
            indexer.ensure_collection(MissingCollectionQdrant(), "docs", 768, resume=True)

    def test_progress_line_reports_batch_and_average_timing(self):
        line = indexer._progress_line(
            indexed=968,
            total=5042,
            batch_count=32,
            batch_seconds=64.0,
            elapsed_seconds=256.0,
            start_indexed=712,
        )

        self.assertIn("968/5042", line)
        self.assertIn("batch 1m 4s", line)
        self.assertIn("2.00s/chunk", line)
        self.assertIn("avg 1.00s/chunk", line)
        self.assertIn("ETA 1h 7m", line)


if __name__ == "__main__":
    unittest.main()
