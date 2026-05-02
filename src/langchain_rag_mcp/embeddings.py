from __future__ import annotations

from typing import Protocol


class Embeddings(Protocol):
    def embed_query(self, text: str) -> list[float]:
        ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...


class LlamaCppEmbeddings:
    def __init__(self, url: str) -> None:
        self.url = url

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        import httpx

        response = httpx.post(
            self.url,
            json={"input": texts},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()["data"]
        data.sort(key=lambda x: x["index"])
        return [item["embedding"] for item in data]


class GoogleGenAIEmbeddings:
    def __init__(
        self,
        model: str = "models/gemini-embedding-2",
        output_dimensionality: int | None = None,
    ) -> None:
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
        except ImportError as exc:
            raise RuntimeError(
                "Google embeddings require the langchain-google-genai package. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc

        kwargs = {"model": model}
        if output_dimensionality:
            kwargs["output_dimensionality"] = output_dimensionality
        self._embeddings = GoogleGenerativeAIEmbeddings(**kwargs)

    def embed_query(self, text: str) -> list[float]:
        return self._embeddings.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embeddings.embed_documents(texts)


def create_embeddings(
    provider: str,
    *,
    llamacpp_url: str,
    google_model: str,
    google_output_dimensionality: int | None = None,
) -> Embeddings:
    normalized = provider.strip().lower()
    if normalized in {"google", "gemini", "google-genai"}:
        return GoogleGenAIEmbeddings(
            model=google_model,
            output_dimensionality=google_output_dimensionality,
        )
    if normalized in {"llamacpp", "llama.cpp", "local"}:
        return LlamaCppEmbeddings(llamacpp_url)
    raise ValueError(f"Unsupported embedding provider: {provider}")
