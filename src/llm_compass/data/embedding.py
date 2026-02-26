"""Handle FAISS embeddings."""

from typing import Any

import httpx
import numpy as np
import faiss


from llm_compass.config import Settings


EMBED_MODEL = "qwen/qwen3-embedding-8b"
EMBED_DIM = 4096  # Qwen3-Embedding-8B default benchmark dimension


class Embedding:
    settings: Settings
    index: faiss.IndexIDMap2 | None

    def __init__(self, settings: Settings):
        self.settings = settings
        if settings.get_faiss_path().exists():
            self.index = self._load_index()
        else:
            self.index = None

    def _openrouter_embed(self, texts: list[str]) -> np.ndarray:
        """Embeds multiple strings at once using the defined EMBED_MODEL.
        Returns array of shape (len(texts), EMBED_DIM)
        """
        embeddings_url = f"{self.settings.openrouter_base_url.rstrip('/')}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.settings.openrouter_api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": EMBED_MODEL,
            "input": texts,
        }  # input can be an array
        with httpx.Client(timeout=10) as client:
            r = client.post(embeddings_url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()["data"]

        vecs = np.array([item["embedding"] for item in data], dtype="float32")
        if vecs.shape[1] != EMBED_DIM:
            raise ValueError(
                f"EMBED_DIM={EMBED_DIM} doesn't match actual embedding size of "
                "'{EMBED_MODEL}' (returned {vecs.shape[1]} dim vectors)"
            )
        return vecs

    def _build_faiss_index(self, vecs: np.ndarray, doc_ids: list[int]) -> faiss.IndexIDMap2:
        """Builds a FAISS index from the given vectors and document IDs.
        Uses IndexFlatIP for exact inner product search, with L2 normalization for cosine similarity.

        Args:
            vecs: numpy array of shape (num_docs, EMBED_DIM) containing the embedding vectors
            doc_ids: list of integer document IDs corresponding to each vector
        Returns:
            A FAISS index object with the vectors indexed and associated with their IDs.
        """
        faiss.normalize_L2(vecs)  # normalize for cosine via inner product

        dim = vecs.shape[1]
        base = faiss.IndexFlatIP(dim)  # exact inner product search
        index = faiss.IndexIDMap2(base)  # enables add_with_ids, ID mapping

        ids = np.array(doc_ids, dtype=np.int64)
        index.add_with_ids(vecs, ids)  # type: ignore
        return index

    def generate_index(self, records: list[dict[str, Any]], text_key: str, id_key: str):
        """Entry method for generating index from csv data / data records.

        Args:
            records: list of dicts, each representing a row of data with text and id fields
            text_key: the key in the dict to use for embedding text
            id_key: the key in the dict to use for document ID in FAISS index

        Returns:
            None (writes index to disk); can raise exceptions on failure
        """
        print(f"Generating FAISS index for {len(records)} records...")
        texts = [record[text_key] for record in records]
        doc_ids = [record[id_key] for record in records]

        vecs = self._openrouter_embed(texts)
        self.index = self._build_faiss_index(vecs, doc_ids)
        self._write_index(self.index)

    def _write_index(self, index: faiss.IndexIDMap2):
        """Writes the given FAISS index to disk at the configured path.
        Silently overwrites any existing index file.
        """
        faiss.write_index(index, str(self.settings.get_faiss_path()))

    def _load_index(self) -> faiss.IndexIDMap2:
        return faiss.read_index(str(self.settings.get_faiss_path()))

    def search_index(self, meta: dict[int, str], query: str, k: int = 10):
        """Entry method for searching the FAISS index with a query string.
        Args:
            meta: dict mapping document IDs to their original text (for retrieval after search)
            query: the input string to embed and search against the index
            k: number of top results to return
        """
        if self.index is None:
            raise ValueError("FAISS index not found. Please generate the index before searching.")
        q = self._openrouter_embed([query])
        faiss.normalize_L2(q)  # same normalization as index vectors

        scores, ids = self.index.search(q, k)  # type: ignore
        return scores, ids
        scores = scores[0].tolist()
        ids = ids[0].tolist()

        results = []
        for score, doc_id in zip(scores, ids):
            if doc_id == -1:
                continue
            results.append(
                {
                    "doc_id": int(doc_id),
                    "score": float(score),
                    "text": meta.get(int(doc_id), ""),
                }
            )

        # Already sorted by FAISS (best first); keep explicit sort for safety
        results.sort(key=lambda x: x["score"], reverse=True)
        return results
