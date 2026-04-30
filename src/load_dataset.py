"""
load_dataset.py  ─  Download Indian Financial News from HuggingFace
====================================================================

Run this FIRST before anything else:
    python src/load_dataset.py

This downloads the dataset and saves it to data/financial_news.json
so retriever.py can load it without re-downloading every time.

WHAT IS HUGGINGFACE?
  HuggingFace (hf.co) = GitHub for AI models and datasets.
  The `datasets` library lets you download any public dataset in 1 line.
  kdave/Indian_Financial_News = 26k Indian financial news articles.
  It's FREE. No login needed.

WHAT IS A DATASET?
  A structured collection of examples. Like a spreadsheet.
  Each row = one news article with columns: Title, Article, Source, Date.
"""

import json
import os
import sys


def download_and_save(
    num_articles: int = 2000,      # start with 2000 for Week 1 (fast to embed)
    save_path: str = "data/financial_news.json",
    hf_dataset: str = "kdave/Indian_Financial_News"
):
    """
    Downloads Indian Financial News from HuggingFace and saves locally.

    num_articles: How many to use. More = better retrieval, slower indexing.
      500   = ~1 min to embed  (quick test)
      2000  = ~4 min to embed  (recommended for Week 1)
      10000 = ~20 min to embed (full experiment for Week 2)
    """
    # Install datasets if missing
    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing 'datasets' library from HuggingFace...")
        os.system(f"{sys.executable} -m pip install datasets -q")
        from datasets import load_dataset

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

    print(f"Downloading '{hf_dataset}' from HuggingFace...")
    print("(First download: ~100-200 MB, takes 1-3 minutes)")
    print("(Subsequent runs use cache — instant)")

    # Load dataset — HuggingFace auto-downloads and caches
    try:
        ds = load_dataset(hf_dataset, split="train")
    except Exception as e:
        print(f"\nError downloading dataset: {e}")
        print("If this fails, manually download from:")
        print(f"  https://huggingface.co/datasets/{hf_dataset}")
        print("Or use the sample data generated below:")
        _save_sample_data(save_path)
        return

    print(f"\nDataset loaded! Total articles: {len(ds)}")
    print(f"Column names: {ds.column_names}")
    print("\nSample entry:")
    for k, v in ds[0].items():
        print(f"  {k}: {str(v)[:80]}")

    # Standardize and save subset
    articles = []
    skipped  = 0
    for i, item in enumerate(ds):
        if len(articles) >= num_articles:
            break

        # Handle different possible column names in the dataset
        text   = (item.get("Article") or item.get("text") or
                  item.get("content") or item.get("Content") or item.get("body") or "")
        title  = (item.get("Title")   or item.get("title") or
                  item.get("headline") or f"Article_{i}")
        source = (item.get("Source")  or item.get("source") or "Unknown")
        date   = (item.get("Date")    or item.get("date")   or "")

        text   = str(text).strip()
        title  = str(title).strip()

        if len(text) < 150:    # skip very short or empty articles
            skipped += 1
            continue

        articles.append({
            "id":     i,
            "title":  title,
            "text":   text,
            "source": source,
            "date":   date,
        })

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)

    total_words = sum(len(a["text"].split()) for a in articles)
    print(f"\n✓ Saved {len(articles)} articles to '{save_path}'")
    print(f"  Skipped {skipped} short/empty articles")
    print(f"  Total words: {total_words:,}")
    print(f"  Avg article length: {total_words // len(articles)} words")
    print(f"  Estimated chunks: ~{(total_words // 250):,}")
    print(f"  Estimated embedding time: ~{(total_words // 250) // 2500 + 1} min on CPU")

    # Print source distribution
    from collections import Counter
    sources = Counter(a["source"] for a in articles)
    print(f"\nTop sources:")
    for src, count in sources.most_common(5):
        print(f"  {src}: {count} articles")

    return articles


def _save_sample_data(save_path: str):
    """
    Fallback: create sample data if HuggingFace download fails.
    This is enough to test the pipeline end-to-end.
    """
    print("\nCreating sample data as fallback...")
    sample = [
        {
            "id": 0,
            "title": "RBI Hikes Repo Rate by 25 bps to 6.75%",
            "text": ("The Reserve Bank of India's Monetary Policy Committee unanimously decided to "
                     "raise the repo rate by 25 basis points to 6.75 percent on Wednesday, February 8 2024. "
                     "Governor Shaktikanta Das said the decision aims to bring inflation durably to the 4 percent target. "
                     "Consumer price inflation stood at 5.69 percent in December 2023. "
                     "The repo rate is the rate at which RBI lends to commercial banks. "
                     "Higher repo rate makes loans costlier for consumers and businesses. "
                     "Home loan EMIs for a Rs 50 lakh loan are expected to rise by Rs 800 per month. "
                     "The Sensex fell 450 points and Nifty dropped 140 points after the announcement. "
                     "Banking stocks like HDFC Bank, ICICI Bank, and Axis Bank declined 2-3 percent. "
                     "This is the sixth consecutive rate hike since May 2022 when rates stood at 4 percent. ") * 5,
            "source": "Economic Times",
            "date": "2024-02-08"
        },
        {
            "id": 1,
            "title": "Infosys Reports 12% Profit Rise in Q3 FY2024",
            "text": ("IT major Infosys reported a 12 percent year-on-year rise in consolidated net profit "
                     "for the third quarter of FY2024 at Rs 6,106 crore, beating analyst estimates of Rs 5,900 crore. "
                     "Revenue grew 8.1 percent to Rs 38,821 crore in the October-December 2023 quarter. "
                     "The company maintained its revenue growth guidance of 1-2.5 percent in constant currency for FY24. "
                     "CEO and MD Salil Parekh said deal wins of $3.8 billion were strong despite macro uncertainty. "
                     "The board declared an interim dividend of Rs 18 per share. "
                     "Operating margin came in at 20.5 percent, up 60 basis points quarter-on-quarter. "
                     "Headcount declined by 1,600 employees to 317,240 as the company reduced hiring. "
                     "The stock rose 3.5 percent to Rs 1,560 on the Bombay Stock Exchange after results. ") * 5,
            "source": "Mint",
            "date": "2024-01-11"
        },
        {
            "id": 2,
            "title": "SEBI Proposes Stricter F&O Trading Rules to Protect Retail Investors",
            "text": ("Securities and Exchange Board of India proposed new regulations for the futures and options "
                     "segment to reduce losses by retail investors. The regulator's consultation paper proposes "
                     "increasing minimum contract size from Rs 5 lakh to Rs 15 lakh. "
                     "Weekly index expiry contracts will be restricted to one per exchange per week. "
                     "SEBI data reveals that 9 out of 10 individual F&O traders lost money in FY2022-23. "
                     "Total retail losses in F&O stood at Rs 51,689 crore in FY2023. "
                     "Upfront collection of options premium from buyers will be mandatory. "
                     "Exchange-level position limits for index derivatives will be removed. "
                     "The consultation paper invited public comments until October 4, 2024. "
                     "New rules are expected to reduce speculative retail trading volume by 30-40 percent. ") * 5,
            "source": "Business Standard",
            "date": "2024-07-30"
        },
        {
            "id": 3,
            "title": "Adani Group's Total Debt Rises to Rs 2.27 Lakh Crore",
            "text": ("Adani Group's aggregate net debt rose 17 percent to Rs 2.27 lakh crore in FY2024 "
                     "as the conglomerate accelerated infrastructure investments. "
                     "The group's EBITDA grew 45 percent to Rs 82,917 crore providing comfortable debt coverage. "
                     "Net debt to EBITDA ratio improved to 2.74 times from 3.27 times a year ago. "
                     "Adani Green Energy had the highest debt at Rs 63,600 crore. "
                     "The group invested Rs 80,000 crore in capex across airports, ports, and green energy in FY24. "
                     "Adani Ports grew revenue 23 percent handling 420 million metric tonnes of cargo. "
                     "The group aims to reduce debt-to-equity ratio below 1 by FY2027. ") * 5,
            "source": "Financial Express",
            "date": "2024-06-15"
        },
        {
            "id": 4,
            "title": "India GDP Growth Forecast at 7.2% for FY2024: IMF",
            "text": ("The International Monetary Fund revised India's GDP growth forecast upward to 7.2 percent "
                     "for fiscal year 2024, making it the fastest growing major economy. "
                     "The revision is 30 basis points higher than IMF's January forecast of 6.8 percent. "
                     "Strong domestic consumption and government capital expenditure drove the upgrade. "
                     "The Centre's capital expenditure target of Rs 10 lakh crore for FY24 was key. "
                     "Services exports grew 12 percent while goods exports declined 3 percent. "
                     "The IMF cautioned about food inflation driven by uneven monsoon distribution. "
                     "India overtook China's 4.6 percent growth to lead among G20 economies. "
                     "The World Bank separately forecast 7.5 percent growth for India in FY25. ") * 5,
            "source": "Hindu Business Line",
            "date": "2024-04-16"
        }
    ]
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(sample, f, indent=2)
    print(f"✓ Sample data saved to '{save_path}' ({len(sample)} articles)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2000, help="Number of articles to download")
    parser.add_argument("--sample", action="store_true", help="Use sample data instead of downloading")
    args = parser.parse_args()

    if args.sample:
        _save_sample_data("data/financial_news.json")
    else:
        download_and_save(num_articles=args.n)
