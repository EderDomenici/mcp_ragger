class LlamaCppEmbeddings:
    def __init__(self, url: str) -> None:
        self.url = url

    def embed_query(self, text: str) -> list[float]:
        import httpx

        response = httpx.post(
            self.url,
            json={"input": [text]},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]
