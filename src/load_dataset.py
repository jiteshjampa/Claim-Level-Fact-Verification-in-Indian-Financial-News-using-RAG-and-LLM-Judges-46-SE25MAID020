"""
load_dataset.py — Downloads full Indian Financial News dataset.
Run once before anything else:
    python src/load_dataset.py
"""

import json
import os
import random

SEED = 42
random.seed(SEED)

DATA_PATH = "data/financial_news.json"


def load_full_dataset():
    """Download all articles from HuggingFace and save locally."""

    try:
        from datasets import load_dataset

        print("[load_dataset] Downloading kdave/Indian_Financial_News from HuggingFace ...")

        ds = load_dataset("kdave/Indian_Financial_News", split="train")

        print(f"[load_dataset] Raw dataset size: {len(ds)} articles")

        # Debug info
        print("\n[load_dataset] Dataset Columns:")
        print(ds.column_names)

        print("\n[load_dataset] First Row Sample:")
        print(ds[0])

        articles = []

        for i, row in enumerate(ds):

            # CORRECT FIELD NAMES
            text = str(row.get("Content", "")).strip()
            summary = str(row.get("Summary", "")).strip()
            sentiment = str(row.get("Sentiment", "")).strip()
            url = str(row.get("URL", "")).strip()

            # Generate simple title from first sentence
            title = summary[:80] if summary else f"Article {i}"

            # Keep meaningful articles
            if text and len(text) > 50:

                articles.append({
                    "id": i,
                    "title": title,
                    "text": text,
                    "summary": summary,
                    "sentiment": sentiment,
                    "url": url,
                })

        print(f"\n[load_dataset] Kept {len(articles)} articles after filtering short texts.")

        # Create data directory
        os.makedirs("data", exist_ok=True)

        # Save locally
        with open(DATA_PATH, "w", encoding="utf-8") as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)

        print(f"[load_dataset] Saved to {DATA_PATH}")

        return articles

    except Exception as e:
        print(f"\n[load_dataset] ERROR: {e}")
        print("\nInstall required package:")
        print("pip install datasets")
        raise


def load_local():
    """Load dataset from local JSON file."""

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run:\npython src/load_dataset.py"
        )

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        articles = json.load(f)

    print(f"[load_dataset] Loaded {len(articles)} articles from {DATA_PATH}")

    return articles


if __name__ == "__main__":

    articles = load_full_dataset()

    if articles:
        print("\nSample article:")
        print(f"Title: {articles[0]['title']}")
        print(f"Length: {len(articles[0]['text'])} chars")
        print(f"Sentiment: {articles[0]['sentiment']}")
    else:
        print("\nNo articles found after filtering.")