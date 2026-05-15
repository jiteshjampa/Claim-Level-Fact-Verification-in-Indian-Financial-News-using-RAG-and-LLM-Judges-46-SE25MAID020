"""
main.py — Full RAG + Hallucination Detection Pipeline
CS5202 · Indian Financial News · Spring 2026

Usage:
    python main.py --query "What did RBI do with interest rates?"
    python main.py --query "What are SEBI regulations for F&O?" --top_k 5 --chunk_size 250
    python main.py --evaluate          # run full 30-question evaluation + ablations
    python main.py --compare_papers    # print paper comparison table

Environment:
    export GROQ_API_KEY="gsk_..."
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, "src")

SEED = 42


def run_pipeline(query: str, top_k: int = 3, chunk_size: int = 250,
                 prompt_variant: str = "strict", n_selfcheck: int = 3,
                 use_bertscore: bool = True):
    """
    Run the full 5-stage pipeline for a single query and print results.

    Stages:
        1. Load dataset and build FAISS index
        2. Retrieve top-k relevant chunks
        3. Generate RAG answer (Groq / Llama3-8B)
        4. Validate with LLM-as-Judge
        5. SelfCheckGPT consistency scoring
    """
    from load_dataset import load_local
    from retriever import build_index, retrieve
    from generator import generate_answer, generate_samples
    from validator import validate
    from selfcheck import selfcheck

    print("\n" + "=" * 70)
    print("  FINANCIAL RAG + HALLUCINATION DETECTION PIPELINE")
    print("=" * 70)
    print(f"  Query       : {query}")
    print(f"  Top-K       : {top_k}  |  Chunk size: {chunk_size}w  |  Prompt: {prompt_variant}")
    print("=" * 70)

    # Stage 1: Load + Index
    print("\n[1/5] Loading dataset ...")
    articles = load_local()
    print(f"      {len(articles)} articles loaded.")

    print("\n[2/5] Building/loading FAISS index ...")
    index, chunks = build_index(articles, chunk_size=chunk_size, overlap=40)
    print(f"      {len(chunks)} chunks indexed.")

    # Stage 2: Retrieve
    print(f"\n[3/5] Retrieving top-{top_k} chunks ...")
    retrieved, ret_scores = retrieve(query, index, chunks, top_k=top_k)
    avg_score = sum(ret_scores) / len(ret_scores) if ret_scores else 0
    print(f"      Avg retrieval score: {avg_score:.4f}")
    for i, (c, s) in enumerate(zip(retrieved, ret_scores), 1):
        print(f"      [{i}] score={s:.3f} | {c.get('title','')[:60]} | {c['text'][:80]}...")

    # Stage 3: Generate
    print("\n[4/5] Generating RAG answer (Groq / Llama3-8B) ...")
    t0 = time.time()
    answer = generate_answer(query, retrieved)
    gen_time = time.time() - t0
    print(f"      Generated in {gen_time:.1f}s")
    print(f"\n  ── ANSWER ──────────────────────────────────────────────────────")
    print(f"  {answer}")
    print(f"  ────────────────────────────────────────────────────────────────")

    # Stage 4: Validate
    print(f"\n[5a/5] Running LLM-as-Judge validator (variant={prompt_variant}) ...")
    val = validate(answer, retrieved, prompt_variant=prompt_variant)
    print(f"\n  ── VALIDATION RESULTS ──────────────────────────────────────────")
    print(f"  Verdict          : {val['verdict']}")
    print(f"  Hallucination %  : {val['hallucination_rate']:.1%}")
    print(f"  Support %        : {val['supported_rate']:.1%}")
    print(f"  Counts           : SUPPORTED={val['counts']['SUPPORTED']}  "
          f"UNSUPPORTED={val['counts']['UNSUPPORTED']}  "
          f"CONTRADICTED={val['counts']['CONTRADICTED']}")
    print(f"\n  Per-sentence labels:")
    for item in val["sentence_labels"]:
        emoji = "✓" if item["label"] == "SUPPORTED" else ("✗" if item["label"] == "CONTRADICTED" else "?")
        print(f"  {emoji} [{item['label']:12s}] {item['sentence'][:80]}")
        if item.get("reason"):
            print(f"             → {item['reason'][:80]}")
    print(f"  ────────────────────────────────────────────────────────────────")

    # Stage 5: SelfCheckGPT
    print(f"\n[5b/5] Running SelfCheckGPT ({n_selfcheck} samples) ...")
    samples = generate_samples(query, retrieved, n=n_selfcheck)
    sc = selfcheck(samples, use_bertscore=use_bertscore)
    print(f"\n  ── SELFCHECK RESULTS ───────────────────────────────────────────")
    print(f"  Consistency score: {sc['consistency_score']:.4f}  ({sc['method']})")
    print(f"  Risk level       : {sc['risk_level']}")
    print(f"  Pairwise scores  : {sc['pairwise_scores']}")
    print(f"  ────────────────────────────────────────────────────────────────")

    # Agreement
    agreement = (
        (sc["risk_level"] == "LOW" and val["hallucination_rate"] < 0.20) or
        (sc["risk_level"] == "HIGH" and val["hallucination_rate"] >= 0.20)
    )
    print(f"\n  ── METHOD AGREEMENT ────────────────────────────────────────────")
    print(f"  Validator says   : {val['verdict']}")
    print(f"  SelfCheck says   : {sc['risk_level']} risk")
    print(f"  Methods agree    : {'YES ✓' if agreement else 'NO (disagreement case)'}")
    print(f"  ────────────────────────────────────────────────────────────────\n")

    return {
        "query": query,
        "answer": answer,
        "avg_retrieval_score": round(avg_score, 4),
        "validator": val,
        "selfcheck": sc,
        "methods_agree": agreement,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Financial RAG + Hallucination Detection Pipeline"
    )
    parser.add_argument("--query",          type=str, default=None,
                        help="Financial question to answer and validate")
    parser.add_argument("--top_k",          type=int, default=3,
                        help="Number of chunks to retrieve (default 3)")
    parser.add_argument("--chunk_size",     type=int, default=250,
                        help="Chunk size in words (default 250)")
    parser.add_argument("--prompt_variant", type=str, default="strict",
                        choices=["strict", "lenient", "chain_of_thought"],
                        help="Validator prompt variant")
    parser.add_argument("--n_selfcheck",    type=int, default=3,
                        help="Number of SelfCheck samples (default 3)")
    parser.add_argument("--no_bertscore",   action="store_true",
                        help="Use Jaccard instead of BERTScore for SelfCheck")
    parser.add_argument("--evaluate",       action="store_true",
                        help="Run full 30-question evaluation + 3 ablation studies")
    parser.add_argument("--compare_papers", action="store_true",
                        help="Print comparison against SelfCheckGPT / Self-RAG / LRP4RAG")
    args = parser.parse_args()

    if not os.environ.get("GROQ_API_KEY"):
        print("[ERROR] GROQ_API_KEY not set.")
        print("  Get a free key at https://console.groq.com")
        print("  Then: export GROQ_API_KEY='gsk_...'")
        sys.exit(1)

    if args.evaluate:
        from evaluate import run_evaluation
        run_evaluation()
        if args.compare_papers:
            from paper_comparison import load_our_results, print_comparison_table, save_comparison
            our = load_our_results()
            print_comparison_table(our)
            save_comparison(our)
        return

    if args.compare_papers:
        from paper_comparison import load_our_results, print_comparison_table, save_comparison
        our = load_our_results()
        print_comparison_table(our)
        save_comparison(our)
        return

    query = args.query
    if not query:
        # interactive mode
        print("No --query provided. Running demo queries.\n")
        demo_queries = [
            "What action did the RBI take regarding the repo rate?",
            "What new regulations did SEBI introduce for F&O traders?",
            "How did Reliance Industries perform in its most recent earnings report?",
        ]
        for q in demo_queries:
            run_pipeline(
                query=q,
                top_k=args.top_k,
                chunk_size=args.chunk_size,
                prompt_variant=args.prompt_variant,
                n_selfcheck=args.n_selfcheck,
                use_bertscore=not args.no_bertscore,
            )
            print("\n" + "─" * 70 + "\n")
    else:
        run_pipeline(
            query=query,
            top_k=args.top_k,
            chunk_size=args.chunk_size,
            prompt_variant=args.prompt_variant,
            n_selfcheck=args.n_selfcheck,
            use_bertscore=not args.no_bertscore,
        )


if __name__ == "__main__":
    main()
