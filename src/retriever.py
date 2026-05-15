"""
retriever.py — FAISS-based retriever over Indian Financial News (26k articles).

Supports ablation over chunk_size and top_k.
Chunks are built with sliding window + overlap.
"""

import json
import os
import pickle
import numpy as np
from typing import List, Dict, Tuple

SEED = 42

# ── Default hyperparameters (ablation values tested: 150/250/400 words) ──────
DEFAULT_CHUNK_SIZE = 250   # words
DEFAULT_OVERLAP    = 40    # words
DEFAULT_TOP_K      = 3

INDEX_CACHE = "data/faiss_index.pkl"
CHUNKS_CACHE = "data/chunks.json"


def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    words = text.split()
    chunks = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_size])
        if len(chunk.split()) >= 20:          # skip very short tail chunks
            chunks.append(chunk)
    return chunks


def build_index(
    articles: List[Dict],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    force_rebuild: bool = False,
) -> Tuple[object, List[Dict]]:
    """
    Build (or load cached) FAISS IndexFlatIP over article chunks.

    Returns
    -------
    index   : faiss.IndexFlatIP
    chunks  : list of dicts {chunk_id, article_id, title, text, source, date}
    """
    import faiss
    from sentence_transformers import SentenceTransformer

    cache_key = f"{INDEX_CACHE}.cs{chunk_size}.ov{overlap}"
    chunks_key = f"{CHUNKS_CACHE}.cs{chunk_size}.ov{overlap}"

    if not force_rebuild and os.path.exists(cache_key) and os.path.exists(chunks_key):
        print(f"[retriever] Loading cached index ({cache_key}) ...")
        index = faiss.read_index(cache_key)
        with open(chunks_key, "r") as f:
            chunks = json.load(f)
        print(f"[retriever] Loaded {len(chunks)} chunks.")
        return index, chunks

    print(f"[retriever] Building index: {len(articles)} articles, chunk={chunk_size}w, overlap={overlap}w ...")
    model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim

    chunks = []
    for art in articles:
        for c in _chunk_text(art["text"], chunk_size, overlap):
            chunks.append({
                "chunk_id":   len(chunks),
                "article_id": art["id"],
                "title":      art.get("title", ""),
                "text":       c,
                "source":     art.get("source", ""),
                "date":       art.get("date", ""),
            })

    print(f"[retriever] Total chunks: {len(chunks)}")
    texts = [c["text"] for c in chunks]

    print("[retriever] Encoding chunks (this may take a few minutes for 26k articles) ...")
    batch = 512
    all_embs = []
    for i in range(0, len(texts), batch):
        embs = model.encode(texts[i : i + batch], show_progress_bar=False, normalize_embeddings=True)
        all_embs.append(embs)
        if i % 5000 == 0:
            print(f"  ... encoded {i}/{len(texts)}")
    embeddings = np.vstack(all_embs).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # Inner product = cosine for normalized vectors
    index.add(embeddings)

    os.makedirs("data", exist_ok=True)
    faiss.write_index(index, cache_key)
    with open(chunks_key, "w") as f:
        json.dump(chunks, f, ensure_ascii=False)
    print(f"[retriever] Index saved to {cache_key}")
    return index, chunks


def retrieve(
    query: str,
    index,
    chunks: List[Dict],
    top_k: int = DEFAULT_TOP_K,
) -> Tuple[List[Dict], List[float]]:
    """
    Retrieve top-k chunks most relevant to query.

    Returns
    -------
    results : list of chunk dicts
    scores  : corresponding cosine similarity scores
    """
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    q_emb = model.encode([query], normalize_embeddings=True).astype("float32")
    scores_raw, indices = index.search(q_emb, top_k)
    results = []
    scores = []
    for idx, score in zip(indices[0], scores_raw[0]):
        if 0 <= idx < len(chunks):
            results.append(chunks[idx])
            scores.append(float(score))
    return results, scores
