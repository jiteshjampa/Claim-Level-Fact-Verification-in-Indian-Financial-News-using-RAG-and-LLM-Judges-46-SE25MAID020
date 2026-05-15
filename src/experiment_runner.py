"""
experiment_runner.py  ─  Full Evaluation Suite for Final Milestone
===================================================================

Runs 30 financial questions through the complete pipeline and
collects all metrics needed for the report:
  - Retrieval similarity scores
  - Validator: hallucination rate, support rate, verdict per question
  - SelfCheckGPT: consistency score, hallucination signal
  - Agreement between Validator and SelfCheck (for ablation table)

Usage:
    python src/experiment_runner.py
    python src/experiment_runner.py --n 10          # quick test (10 questions)
    python src/experiment_runner.py --mock          # no API key needed

Outputs:
    results/full_experiment.json   ─ all raw results
    results/metrics_summary.json   ─ aggregate metrics (for report tables)
    results/ablation_table.csv     ─ per-question ablation rows
"""

import os
import sys
import json
import csv
import argparse
import time
from typing import List, Dict

sys.path.insert(0, os.path.dirname(__file__))


# ─────────────────────────────────────────────────────────────
# 30 DIVERSE FINANCIAL QUESTIONS
# Covers: RBI policy, corporate earnings, SEBI rules, macro,
#         markets, banking, FDI, commodities, IPO, debt.
# ─────────────────────────────────────────────────────────────
EVAL_QUESTIONS = [
    # Monetary Policy
    "What did the RBI decide about interest rates?",
    "What is the current repo rate set by RBI?",
    "How did RBI address inflation in its monetary policy?",
    # Corporate Earnings
    "How did Infosys perform in its latest quarterly results?",
    "What were TCS's revenue and profit figures for the recent quarter?",
    "How did Reliance Industries perform in its latest earnings?",
    "What guidance did Wipro give for the next quarter?",
    # SEBI and Regulation
    "What new rules did SEBI propose for F&O trading?",
    "What changes did SEBI make to mutual fund regulations?",
    "How did SEBI address insider trading recently?",
    # Macro Economy
    "What is India's GDP growth rate according to the IMF?",
    "How has India's current account deficit changed recently?",
    "What is the status of India's foreign exchange reserves?",
    # Banking Sector
    "How did HDFC Bank perform in its recent quarter?",
    "What happened with Yes Bank's financial recovery?",
    "What is the NPA situation in Indian public sector banks?",
    # Markets
    "What caused the recent Sensex and Nifty movement?",
    "How did FII and DII flows affect Indian markets recently?",
    "What is the outlook for the Indian rupee against the dollar?",
    # Commodities
    "How have crude oil prices impacted India's trade balance?",
    "What is the current gold import situation in India?",
    # FDI and Investment
    "What sectors received the most FDI in India recently?",
    "How did India's startup funding environment change last quarter?",
    # Debt and Bonds
    "What is the latest yield on 10-year Indian government bonds?",
    "How has India's fiscal deficit been tracking against targets?",
    # IPO and Capital Markets
    "What major IPOs were listed on Indian exchanges recently?",
    "How did recent SME IPOs perform on listing day?",
    # Insurance and Fintech
    "What regulatory changes affected Indian insurance companies?",
    "How is UPI transaction volume trending in India?",
    # Trade
    "How did India's exports perform in the recent month?",
]


def run_mock_pipeline(question: str, retriever) -> Dict:
    """
    Run pipeline without API key — uses mock answers.
    Retrieval is real; generation and validation are mocked.
    """
    results = retriever.retrieve(question, top_k=3)
    scores = [round(s, 3) for _, _, s in results]
    sources = [m["title"] for _, m, _ in results]

    mock_answer = (
        f"Based on the retrieved sources: "
        f"[MOCK ANSWER for '{question[:40]}...' — set GROQ_API_KEY for real answers]"
    )

    return {
        "question": question,
        "answer": mock_answer,
        "retrieved_sources": sources,
        "similarity_scores": scores,
        "avg_retrieval_score": round(sum(scores) / len(scores), 3) if scores else 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "context_str": retriever.format_context(results),
        "mock": True,
    }


def run_full_pipeline(question: str, retriever, generator) -> Dict:
    """Run complete pipeline with live API."""
    results = retriever.retrieve(question, top_k=3)
    scores = [round(s, 3) for _, _, s in results]
    sources = [m["title"] for _, m, _ in results]

    gen_output = generator.generate(question, results, temperature=0.1)

    return {
        "question": question,
        "answer": gen_output["answer"],
        "retrieved_sources": sources,
        "similarity_scores": scores,
        "avg_retrieval_score": round(sum(scores) / len(scores), 3) if scores else 0,
        "prompt_tokens": gen_output["prompt_tokens"],
        "completion_tokens": gen_output["completion_tokens"],
        "total_tokens": gen_output["total_tokens"],
        "context_str": gen_output["context_str"],
        "mock": False,
    }


def run_selfcheck(question: str, context_str: str, retriever, generator, n_samples: int = 3) -> Dict:
    """
    Generate N samples with temperature=0.7 for SelfCheckGPT.
    Falls back to mock if no API key.
    """
    from selfcheck import SelfCheckGPT

    checker = SelfCheckGPT(method="jaccard")

    if not generator or not generator.api_key:
        # Mock: identical samples → perfect consistency (demonstrates the concept)
        mock_answer = f"[MOCK SELFCHECK SAMPLE for: {question[:40]}]"
        samples = [mock_answer] * n_samples
    else:
        retrieved = retriever.retrieve(question, top_k=3)
        sample_outputs = generator.generate_multiple(question, retrieved, n_samples=n_samples, temperature=0.7)
        samples = [o["answer"] for o in sample_outputs]

    sc_result = checker.score(samples)
    sc_result["samples"] = samples
    return sc_result


def run_validation(answer: str, context_str: str, validator) -> Dict:
    """Run LLM-as-Judge validator. Returns mock result if no API."""
    if not validator or not validator.api_key:
        # Mock validation result for demo
        sentences = answer.split(". ")
        mock_sentences = []
        for s in sentences:
            if s.strip():
                mock_sentences.append({
                    "text": s.strip(),
                    "label": "SUPPORTED",
                    "reason": "MOCK — set GROQ_API_KEY to run real validation",
                    "confidence": 0.0
                })
        return {
            "sentences": mock_sentences,
            "hallucination_rate": 0.0,
            "supported_count": len(mock_sentences),
            "unsupported_count": 0,
            "contradicted_count": 0,
            "verdict": "MOCK_RESULT",
            "mock": True,
        }
    return validator.validate(answer, context_str)


def main():
    parser = argparse.ArgumentParser(description="Run full evaluation experiment")
    parser.add_argument("--n",    type=int, default=30,  help="Number of questions to run")
    parser.add_argument("--mock", action="store_true",   help="Skip API calls (demo mode)")
    parser.add_argument("--selfcheck_samples", type=int, default=3)
    args = parser.parse_args()

    questions = EVAL_QUESTIONS[:args.n]
    api_key   = os.environ.get("GROQ_API_KEY", "")
    use_mock  = args.mock or not api_key

    if use_mock:
        print("⚠ Running in MOCK mode (no GROQ_API_KEY). Retrieval is real; answers are placeholder.")
        print("  Set GROQ_API_KEY for real answers and validation.\n")

    # ── Load components ──────────────────────────────────────
    from retriever import FinancialRetriever
    retriever = FinancialRetriever()
    try:
        retriever.load_index()
    except FileNotFoundError:
        print("No FAISS index found. Building from data/financial_news.json...")
        with open("data/financial_news.json", encoding="utf-8") as f:
            articles = json.load(f)
        retriever.build_index(articles)

    generator = None
    validator = None
    if not use_mock:
        from generator import RAGGenerator
        from validator import HallucinationValidator
        generator = RAGGenerator()
        validator = HallucinationValidator()

    # ── Run experiments ───────────────────────────────────────
    all_results = []
    print(f"\nRunning {len(questions)} questions...\n{'='*65}")

    for i, q in enumerate(questions):
        print(f"\n[{i+1}/{len(questions)}] {q}")
        t0 = time.time()

        # Step 1: Generate answer
        if use_mock:
            row = run_mock_pipeline(q, retriever)
        else:
            row = run_full_pipeline(q, retriever, generator)
            time.sleep(0.5)  # respect Groq rate limits

        print(f"  Retrieval scores: {row['similarity_scores']}")
        print(f"  Avg score: {row['avg_retrieval_score']}")

        # Step 2: Validate answer
        val_result = run_validation(row["answer"], row["context_str"], validator)
        row["validation"] = val_result
        h_rate = val_result.get("hallucination_rate", -1)
        print(f"  Hallucination rate: {h_rate:.1%}" if h_rate >= 0 else "  Hallucination: MOCK")

        # Step 3: SelfCheck
        sc_result = run_selfcheck(q, row["context_str"], retriever, generator, args.selfcheck_samples)
        row["selfcheck"] = sc_result
        print(f"  SelfCheck consistency: {sc_result['consistency_score']} ({sc_result['hallucination_signal']} risk)")

        elapsed = time.time() - t0
        print(f"  Time: {elapsed:.1f}s")

        all_results.append(row)

    # ── Aggregate metrics ─────────────────────────────────────
    print(f"\n{'='*65}\nComputing aggregate metrics...")
    metrics = compute_aggregate_metrics(all_results)
    ablation = build_ablation_table(all_results)

    # ── Save outputs ──────────────────────────────────────────
    os.makedirs("results", exist_ok=True)

    # Full results (strip context_str for file size)
    save_results = []
    for r in all_results:
        row = {k: v for k, v in r.items() if k != "context_str"}
        if "validation" in row:
            val = row["validation"]
            row["validation_summary"] = {
                "hallucination_rate": val.get("hallucination_rate"),
                "verdict": val.get("verdict"),
                "supported_count": val.get("supported_count"),
                "unsupported_count": val.get("unsupported_count"),
                "contradicted_count": val.get("contradicted_count"),
                "mock": val.get("mock", False),
            }
            row.pop("validation", None)
        save_results.append(row)

    with open("results/full_experiment.json", "w") as f:
        json.dump(save_results, f, indent=2)

    with open("results/metrics_summary.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open("results/ablation_table.csv", "w", newline="") as f:
        if ablation:
            writer = csv.DictWriter(f, fieldnames=ablation[0].keys())
            writer.writeheader()
            writer.writerows(ablation)

    # Print summary
    print(f"\n{'='*65}")
    print("EXPERIMENT COMPLETE — METRICS SUMMARY")
    print(f"{'='*65}")
    print(f"  Questions run:          {metrics['n_questions']}")
    print(f"  Avg retrieval score:    {metrics['avg_retrieval_score']:.3f}")
    print(f"  Avg hallucination rate: {metrics.get('avg_hallucination_rate', 'N/A (mock)')}")
    print(f"  Avg selfcheck score:    {metrics['avg_selfcheck_score']:.3f}")
    print(f"  Validator-SelfCheck agreement: {metrics.get('agreement_rate', 'N/A')}")
    print(f"\n  Saved:")
    print(f"    results/full_experiment.json")
    print(f"    results/metrics_summary.json")
    print(f"    results/ablation_table.csv")


def compute_aggregate_metrics(results: List[Dict]) -> Dict:
    """Compute all aggregate metrics across all questions."""
    retrieval_scores = [r["avg_retrieval_score"] for r in results]

    halluc_rates = []
    support_rates = []
    verdict_counts = {}
    for r in results:
        val = r.get("validation", {})
        h = val.get("hallucination_rate", -1)
        if h >= 0 and not val.get("mock", False):
            halluc_rates.append(h)
            support_rates.append(1 - h)
        v = val.get("verdict", "UNKNOWN")
        verdict_counts[v] = verdict_counts.get(v, 0) + 1

    sc_scores = [r["selfcheck"]["consistency_score"] for r in results if "selfcheck" in r]
    sc_signals = {}
    for r in results:
        sig = r.get("selfcheck", {}).get("hallucination_signal", "UNKNOWN")
        sc_signals[sig] = sc_signals.get(sig, 0) + 1

    # Agreement rate: SelfCheck signal vs Validator rate
    agreement_count = 0
    comparable = 0
    for r in results:
        val = r.get("validation", {})
        sc  = r.get("selfcheck", {})
        h   = val.get("hallucination_rate", -1)
        sc_score = sc.get("consistency_score", -1)
        if h >= 0 and sc_score >= 0 and not val.get("mock", False):
            comparable += 1
            val_clean = h <= 0.3
            sc_clean  = sc_score >= 0.55
            if val_clean == sc_clean:
                agreement_count += 1

    def safe_mean(lst):
        return round(sum(lst) / len(lst), 4) if lst else None

    return {
        "n_questions":           len(results),
        "avg_retrieval_score":   safe_mean(retrieval_scores),
        "min_retrieval_score":   round(min(retrieval_scores), 3) if retrieval_scores else None,
        "max_retrieval_score":   round(max(retrieval_scores), 3) if retrieval_scores else None,
        "avg_hallucination_rate":safe_mean(halluc_rates),
        "avg_support_rate":      safe_mean(support_rates),
        "verdict_distribution":  verdict_counts,
        "avg_selfcheck_score":   safe_mean(sc_scores) if sc_scores else 0,
        "selfcheck_signal_distribution": sc_signals,
        "agreement_rate":        round(agreement_count / comparable, 3) if comparable > 0 else "N/A (mock)",
        "n_comparable":          comparable,
        "mock_mode":             any(r.get("mock", False) for r in results),
    }


def build_ablation_table(results: List[Dict]) -> List[Dict]:
    """Build per-question ablation comparison rows."""
    rows = []
    for r in results:
        val = r.get("validation", {})
        sc  = r.get("selfcheck", {})
        h   = val.get("hallucination_rate", -1)
        sc_score = sc.get("consistency_score", -1)

        if h >= 0 and sc_score >= 0:
            sc_clean  = sc_score >= 0.55
            val_clean = h <= 0.30
            agrees    = "YES" if sc_clean == val_clean else "NO — analyze"
        else:
            agrees = "N/A (mock)"

        rows.append({
            "question":              r["question"][:55],
            "avg_retrieval_score":   r["avg_retrieval_score"],
            "top_source":            r["retrieved_sources"][0][:40] if r["retrieved_sources"] else "?",
            "hallucination_rate":    h if h >= 0 else "mock",
            "validator_verdict":     val.get("verdict", "mock"),
            "selfcheck_score":       sc_score,
            "selfcheck_signal":      sc.get("hallucination_signal", "?"),
            "methods_agree":         agrees,
            "total_tokens":          r.get("total_tokens", 0),
        })
    return rows


if __name__ == "__main__":
    main()
