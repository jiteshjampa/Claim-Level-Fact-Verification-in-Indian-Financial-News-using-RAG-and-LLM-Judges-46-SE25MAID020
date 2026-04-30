"""
main.py  ─  Full Week 1 Pipeline Demo
======================================

This is the script you run during your Milestone 1 demo.
It shows the complete pipeline end-to-end.

Usage:
    python main.py --sample           # use sample data (no HuggingFace download)
    python main.py                    # use real dataset (needs download first)
    python main.py --query "What did RBI do?"   # custom query
"""

import os
import sys
import json
import argparse


def run_pipeline(query: str, use_sample: bool = False):
    """Run the full RAG pipeline for a query and show results."""

    print("\n" + "=" * 65)
    print("  FINANCIAL RAG + HALLUCINATION VALIDATOR — WEEK 1 DEMO")
    print("=" * 65)

    # ── STEP 1: Load data ──────────────────────────────────────
    print("\n[STEP 1] Loading dataset...")
    data_path = "data/financial_news.json"

    if not os.path.exists(data_path):
        print("Dataset not found. Loading sample data...")
        from src.load_dataset import _save_sample_data
        _save_sample_data(data_path)

    with open(data_path, encoding='utf-8') as f:
        articles = json.load(f)
    print(f"  ✓ Loaded {len(articles)} articles")

    # ── STEP 2: Build / Load FAISS Index ──────────────────────
    print("\n[STEP 2] Setting up FAISS index (embedding store)...")
    sys.path.insert(0, "src")
    from retriever import FinancialRetriever

    retriever = FinancialRetriever()
    try:
        retriever.load_index()
        print("  ✓ Index loaded from disk (pre-built)")
    except FileNotFoundError:
        print("  Building index (first time — may take a few minutes)...")
        retriever.build_index(articles)
        print("  ✓ Index built and saved")

    # ── STEP 3: Retrieve relevant chunks ──────────────────────
    print(f"\n[STEP 3] Retrieving top-3 chunks for query:")
    print(f"  Query: '{query}'")

    results = retriever.retrieve(query, top_k=3)

    print(f"  ✓ Retrieved {len(results)} chunks:")
    for i, (chunk, meta, score) in enumerate(results, 1):
        print(f"    {i}. [{meta['title']}]  similarity={score:.3f}")
        print(f"       Preview: {chunk[:100]}...")

    context_str = retriever.format_context(results)

    # ── STEP 4: Generate Answer ────────────────────────────────
    print("\n[STEP 4] Generating answer with RAG...")
    api_key = os.environ.get("GROQ_API_KEY", "")

    if not api_key:
        print("  ⚠ GROQ_API_KEY not set — showing mock answer for demo")
        answer = (
            "Based on the retrieved sources: The RBI raised the repo rate "
            "by 25 basis points to 6.75 percent. [MOCK ANSWER — set GROQ_API_KEY to generate real answers]"
        )
        tokens = {"prompt": 0, "completion": 0, "total": 0}
    else:
        from generator import RAGGenerator
        gen    = RAGGenerator()
        output = gen.generate(query, results)
        answer = output["answer"]
        tokens = {
            "prompt":     output["prompt_tokens"],
            "completion": output["completion_tokens"],
            "total":      output["total_tokens"]
        }

    print(f"\n  ANSWER:\n  {'─'*55}")
    for line in answer.split(". "):
        print(f"  {line.strip()}.")
    print(f"  {'─'*55}")
    if api_key:
        print(f"  Tokens: prompt={tokens['prompt']}, answer={tokens['completion']}, total={tokens['total']}")

    # ── STEP 5: Selfcheck (no API needed) ─────────────────────
    print("\n[STEP 5] SelfCheckGPT consistency demo...")
    from selfcheck import SelfCheckGPT, jaccard_overlap

    # Demo with mock samples (Week 2 will use real API samples)
    mock_samples = [answer, answer, answer]   # identical → perfect consistency
    checker   = SelfCheckGPT(method="jaccard")
    sc_result = checker.score(mock_samples)
    print(f"  Consistency score: {sc_result['consistency_score']} → {sc_result['hallucination_signal']} risk")
    print(f"  (Week 2: will generate 3 real samples with temperature=0.7)")

    # ── STEP 6: Validation preview ────────────────────────────
    print("\n[STEP 6] Hallucination validator (preview)...")
    if api_key:
        from validator import HallucinationValidator
        val    = HallucinationValidator()
        v_out  = val.validate(answer, context_str)
        rate   = v_out.get("hallucination_rate", "?")
        print(f"  Hallucination rate: {rate}")
        print(f"  Verdict: {v_out.get('verdict','?')}")
        for s in v_out.get("sentences", []):
            icon = "✓" if s["label"] == "SUPPORTED" else "✗"
            print(f"  {icon} [{s['label']}] {s['text'][:70]}...")
    else:
        print("  (Set GROQ_API_KEY to run validator)")
        print("  Validator will label each sentence: SUPPORTED / UNSUPPORTED / CONTRADICTED")

    # ── Save results ───────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    out = {
        "query":              query,
        "answer":             answer,
        "retrieved_sources":  [m["title"] for _, m, _ in results],
        "similarity_scores":  [round(s, 3) for _, _, s in results],
        "selfcheck":          sc_result,
    }
    with open("results/week1_demo_output.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n✓ Results saved to results/week1_demo_output.json")

    print("\n" + "=" * 65)
    print("  WEEK 1 PIPELINE COMPLETE  ✓")
    print("  Milestone 1 deliverables:")
    print("    ✓ Dataset loaded")
    print("    ✓ FAISS index built")
    print("    ✓ Retrieval working")
    print("    ✓ RAG generation working" if api_key else "    ⚠ RAG generation (needs GROQ_API_KEY)")
    print("    ✓ Preliminary results saved")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Financial RAG Demo")
    parser.add_argument("--query",  type=str, default="What did the RBI decide about interest rates?")
    parser.add_argument("--sample", action="store_true", help="Use sample data")
    args = parser.parse_args()
    run_pipeline(args.query, args.sample)
