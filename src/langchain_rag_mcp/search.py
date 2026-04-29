import time
from collections.abc import Callable

from .config import Settings
from .retrieval import format_results, query_term_coverage, rerank


NO_RESULTS_MESSAGE = "No relevant documentation found."


class SearchService:
    def __init__(
        self,
        settings: Settings,
        embedder: Callable[[str], list[float]],
        query_points: Callable[[list[float], int], object],
        log_query: Callable[[str, float, list[float], int], None],
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        self.query_points = query_points
        self.log_query = log_query

    def search(self, query: str) -> str:
        t0 = time.perf_counter()
        query_vector = self.embedder(query)
        response = self.query_points(query_vector, self.settings.candidate_k)
        latency_ms = (time.perf_counter() - t0) * 1000

        results = [
            result
            for result in rerank(query, response.points, self.settings.top_k)
            if result.score >= self.settings.min_score
        ]

        if not results or query_term_coverage(query, results) < self.settings.min_query_term_coverage:
            self.log_query(query, latency_ms, [], 0)
            return NO_RESULTS_MESSAGE

        output = format_results(results, self.settings.chunk_cap)
        scores = [result.score for result in results]
        self.log_query(query, latency_ms, scores, len(output) // 4)
        return output
