import time
import httpx
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

QDRANT_URL = "http://localhost:6333"
LLAMACPP_URL = "http://localhost:8080/v1/embeddings"
COLLECTION = "langchain_docs"
DOCS_URL = "https://docs.langchain.com/llms-full.txt"

EMBED_INPUT_CAP = 500   # chars por chunk enviado ao llama-server (~125 tokens)
EMBED_BATCH = 50        # chunks por chamada HTTP
UPSERT_BATCH = 100      # pontos por upsert no Qdrant


def download_docs() -> str:
    print(f"Downloading {DOCS_URL} ...")
    response = httpx.get(DOCS_URL, timeout=60, follow_redirects=True)
    response.raise_for_status()
    return response.text


def chunk_docs(text: str) -> list[dict]:
    raw_chunks = text.split("\n# ")
    chunks = []

    for i, chunk in enumerate(raw_chunks):
        if not chunk.strip():
            continue

        if i == 0:
            chunk = chunk.lstrip("# ")

        lines = chunk.strip().splitlines()
        title = lines[0].strip()
        body = "\n".join(lines[1:]).strip()

        source = ""
        for line in lines[1:5]:
            if line.startswith("Source:"):
                source = line.replace("Source:", "").strip()
                break

        chunks.append({
            "title": title,
            "source": source,
            "content": f"# {title}\n{body}",
        })

    return chunks


def embed_batch(texts: list[str]) -> list[list[float]]:
    response = httpx.post(
        LLAMACPP_URL,
        json={"input": [t[:EMBED_INPUT_CAP] for t in texts]},
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()["data"]
    data.sort(key=lambda x: x["index"])
    return [item["embedding"] for item in data]


def run():
    qdrant = QdrantClient(url=QDRANT_URL)

    text = download_docs()

    print("Chunking docs...")
    chunks = chunk_docs(text)
    print(f"  {len(chunks)} chunks found")

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
        vectors = embed_batch([c["content"] for c in batch])

        for j, (chunk, vector) in enumerate(zip(batch, vectors)):
            points.append(
                PointStruct(
                    id=start + j,
                    vector=vector,
                    payload={
                        "title": chunk["title"],
                        "source": chunk["source"],
                        "content": chunk["content"],
                    },
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
