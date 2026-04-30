"""
retriever.py  ─  FAISS Vector Store + Sentence-Transformer Embeddings
=====================================================================

BEGINNER GUIDE — READ THIS FIRST
─────────────────────────────────

WHAT IS A TOKEN?
  A token is a small piece of text that the AI reads.
  "Hello world" = 2 tokens (roughly 1 token per word)
  "RBI's monetary policy" = ~5 tokens (punctuation counts too)
  
  WHY IT MATTERS: LLMs can only read a limited number of tokens at once.
  This limit is called the CONTEXT WINDOW.
    - Llama3-8B:  8,192 tokens (~6,000 words)
    - GPT-4:    128,000 tokens (~96,000 words)
  
  Our 26,000 articles = MILLIONS of tokens. We can't feed all of them.
  So we CHUNK (split) articles into small pieces and only feed the
  3 most relevant ones. That's why RETRIEVAL exists.

WHAT IS AN EMBEDDING?
  An embedding converts text into a list of numbers (a "vector").
  Example:
    "RBI raised interest rates" → [0.23, -0.11, 0.87, 0.04, ...]
    "RBI hiked repo rate"       → [0.21, -0.09, 0.85, 0.06, ...]  ← very similar!
    "Monsoon rains in Kerala"   → [-0.55, 0.72, -0.31, 0.88, ...]  ← very different
  
  Similar meaning → similar numbers → FAISS can find them quickly.

WHAT IS FAISS?
  Facebook AI Similarity Search.
  It stores millions of vectors and finds the closest ones in milliseconds.
  Like a search engine, but for meaning (not exact keywords).

FULL PIPELINE:
  Articles → [chunk into 300-word pieces] → [embed each chunk] → [store in FAISS]
  Query    → [embed query]                → [search FAISS]     → [return top-3 chunks]
"""

import os
import json
import pickle
import numpy as np
from typing import List, Tuple, Dict


# ─────────────────────────────────────────────────────────────
# CONFIGURATION  (tweak these if needed)
# ─────────────────────────────────────────────────────────────
CHUNK_SIZE    = 250    # words per chunk  (~330 tokens — safe for all models)
CHUNK_OVERLAP = 40     # overlap so sentences don't get cut mid-thought
EMBED_MODEL   = "all-MiniLM-L6-v2"   # fast + good. 384-dim vectors.
TOP_K         = 3      # how many chunks to return per query
INDEX_DIR     = "data"


# ─────────────────────────────────────────────────────────────
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split a long article into overlapping word-based chunks.

    WHY OVERLAP?
      Article: "...Infosys beat targets. CEO said growth will continue next year..."
      Without overlap, "CEO said growth" might be cut across chunk boundary.
      With 40-word overlap, both chunks share those 40 words → no info lost.

    Example:
      text = "word1 word2 ... word500"
      chunk1 = words 0..249      (250 words)
      chunk2 = words 210..459    (starts 40 words before chunk1 ended)
      chunk3 = words 420..500    (final chunk, may be shorter)
    """
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        end   = min(start + size, len(words))
        chunk = " ".join(words[start:end])
        if len(chunk.strip()) > 30:          # skip tiny fragments
            chunks.append(chunk)
        if end == len(words):
            break
        start += size - overlap              # move forward with overlap
    return chunks


# ─────────────────────────────────────────────────────────────
class FinancialRetriever:
    """Builds a searchable FAISS index and retrieves relevant chunks."""

    def __init__(self, index_dir: str = INDEX_DIR):
        self.index_dir    = index_dir
        self.faiss_path   = os.path.join(index_dir, "faiss.index")
        self.meta_path    = os.path.join(index_dir, "chunks_meta.pkl")
        self.index        = None
        self.chunks: List[str]        = []
        self.metadata: List[Dict]     = []
        self._model = None            # lazy-load to save startup time

    @property
    def model(self):
        """Lazy-load the embedding model on first use."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            print(f"Loading embedding model '{EMBED_MODEL}' ...")
            self._model = SentenceTransformer(EMBED_MODEL)
            print("Model ready.")
        return self._model

    # ── Build ──────────────────────────────────────────────────
    def build_index(self, articles: List[Dict]):
        """
        Given a list of articles, create chunks, embed them, and store in FAISS.

        Each article should be: {"title": "...", "text": "...", "source": "..."}

        TIME ESTIMATE:
          1,000 articles → ~5,000 chunks → ~2 min on CPU
          5,000 articles → ~25,000 chunks → ~10 min on CPU
        """
        import faiss

        print(f"\nBuilding index from {len(articles)} articles...")
        all_chunks, all_meta = [], []

        for art in articles:
            text  = str(art.get("text", "")).strip()
            title = str(art.get("title", "Unknown"))
            if len(text) < 100:
                continue
            for ch in chunk_text(text):
                all_chunks.append(ch)
                all_meta.append({"title": title, "source": art.get("source", "")})

        print(f"Total chunks: {len(all_chunks)}")
        print("Generating embeddings (this is the slow step)...")

        # EMBEDDING STEP
        # model.encode() converts every chunk → vector of 384 numbers
        # batch_size=128 processes 128 chunks at once (faster than 1-by-1)
        vectors = self.model.encode(
            all_chunks, show_progress_bar=True, batch_size=128, convert_to_numpy=True
        ).astype("float32")

        # Normalize vectors so cosine similarity = dot product (FAISS trick)
        faiss.normalize_L2(vectors)

        # BUILD FAISS INDEX
        # IndexFlatIP = "Flat index, Inner Product similarity"
        # "Flat" = exact search (no approximation) — good for <100k vectors
        dim   = vectors.shape[1]   # 384 for MiniLM
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)

        self.index    = index
        self.chunks   = all_chunks
        self.metadata = all_meta

        # Save to disk
        os.makedirs(self.index_dir, exist_ok=True)
        faiss.write_index(index, self.faiss_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump({"chunks": all_chunks, "metadata": all_meta}, f)

        print(f"Index saved ({index.ntotal} vectors) → {self.faiss_path}")

    # ── Load ───────────────────────────────────────────────────
    def load_index(self):
        """Load a previously built index from disk (fast — no re-embedding)."""
        import faiss
        if not os.path.exists(self.faiss_path):
            raise FileNotFoundError(
                f"No index at '{self.faiss_path}'. Run build_index() first."
            )
        self.index = faiss.read_index(self.faiss_path)
        with open(self.meta_path, "rb") as f:
            data           = pickle.load(f)
            self.chunks    = data["chunks"]
            self.metadata  = data["metadata"]
        print(f"Loaded index: {self.index.ntotal} vectors, {len(self.chunks)} chunks")

    # ── Retrieve ───────────────────────────────────────────────
    def retrieve(self, query: str, top_k: int = TOP_K) -> List[Tuple[str, Dict, float]]:
        """
        Find the top_k most relevant chunks for a query.

        Steps:
          1. Embed the query into a 384-dim vector
          2. FAISS computes dot product with every stored vector
          3. Returns top_k chunks with highest similarity scores

        Returns: [(chunk_text, metadata_dict, similarity_score), ...]
        Scores range from 0 (no match) to 1 (identical meaning)
        """
        import faiss
        if self.index is None:
            raise RuntimeError("Call build_index() or load_index() first.")

        q_vec = self.model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q_vec)

        scores, indices = self.index.search(q_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:   # FAISS returns -1 when fewer results than top_k
                continue
            results.append((self.chunks[idx], self.metadata[idx], float(score)))
        return results

    def format_context(self, results: List[Tuple]) -> str:
        """Format retrieved chunks into a readable context string for the LLM."""
        parts = []
        for i, (chunk, meta, score) in enumerate(results, 1):
            parts.append(f"[Source {i} | {meta.get('title','?')} | score={score:.2f}]\n{chunk}")
        return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────────
# QUICK TEST  (run: python src/retriever.py)
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    # Try loading real dataset first; fall back to sample
    data_path = "data/financial_news.json"
    if os.path.exists(data_path):
        with open(data_path, encoding='utf-8') as f:
            articles = json.load(f)[:200]   # use first 200 for quick test
        print(f"Using real dataset: {len(articles)} articles")
    else:
        print("Using sample articles (run load_dataset.py to get real data)")
        articles = [
            {
                "title": "RBI Hikes Repo Rate by 25 bps",
                "text": ("The Reserve Bank of India raised its benchmark repo rate by 25 basis points "
                         "to 6.75 percent on Wednesday in a unanimous MPC decision. Governor Shaktikanta Das "
                         "cited persistent inflation above the 4 percent target. Sensex fell 400 points. "
                         "Banking stocks were the worst hit. Home loan EMIs will increase by Rs 500-800 monthly "
                         "for a Rs 50 lakh loan. The move is the sixth consecutive hike since May 2022. ") * 8,
                "source": "Economic Times"
            },
            {
                "title": "Infosys Q3 FY24 Results Beat Estimates",
                "text": ("Infosys reported a 12 percent rise in net profit for Q3 FY2024 at Rs 6,106 crore, "
                         "beating analyst estimates of Rs 5,900 crore. Revenue grew 8.1 percent year-on-year "
                         "to Rs 38,821 crore. The company maintained its revenue growth guidance of 1-2.5 percent "
                         "for FY24. CEO Salil Parekh attributed strong deal wins worth $3.8 billion. "
                         "The board announced an interim dividend of Rs 18 per share. ") * 8,
                "source": "Mint"
            },
            {
                "title": "SEBI Tightens F&O Rules",
                "text": ("SEBI has proposed stricter regulations on futures and options trading to reduce "
                         "retail investor losses. The new rules increase the minimum contract size from "
                         "Rs 5 lakh to Rs 15 lakh. Weekly expiry will be restricted to one benchmark index "
                         "per exchange. SEBI data shows 9 out of 10 retail F&O traders lose money. "
                         "The rules will come into effect from January 2025. ") * 8,
                "source": "Business Standard"
            },
        ]

    retriever = FinancialRetriever()
    retriever.build_index(articles)

    # Test queries
    test_queries = [
        "What did RBI do with interest rates?",
        "How did Infosys perform in Q3?",
        "What are SEBI's new trading rules?"
    ]

    for query in test_queries:
        results = retriever.retrieve(query)
        print(f"\nQuery: '{query}'")
        print(f"Top result: [{results[0][1]['title']}] score={results[0][2]:.3f}")
        print(f"Preview: {results[0][0][:120]}...")
