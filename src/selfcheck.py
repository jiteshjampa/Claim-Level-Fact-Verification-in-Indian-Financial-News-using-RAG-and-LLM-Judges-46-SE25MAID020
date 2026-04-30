"""
selfcheck.py  ─  SelfCheckGPT Consistency Scoring
==================================================

PAPER: "SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection"
       Manakul et al., 2023 (arXiv:2303.08896)

THE CORE IDEA (explain it like this in your demo):
  Imagine asking a student the same exam question on 3 different days.
  If they KNOW the answer → same answer each time (consistent).
  If they're GUESSING   → different answers each time (inconsistent).

  Same trick with LLMs:
    1. Ask same question 3 times with temperature=0.7 (adds randomness)
    2. Compare the 3 answers — how similar are they?
    3. Low similarity → model was uncertain → likely hallucinating

YOUR ABLATION (why this matters for 30-point Results section):
  You have TWO hallucination detection methods:
    Method A: Validator LLM  (validator.py) → explicit SUPPORTED/UNSUPPORTED labels
    Method B: SelfCheckGPT   (this file)    → implicit consistency score

  For your ablation table, run both on 20-30 questions and show:
    | Question | Validator Score | SelfCheck Score | Do they agree? |
    |----------|-----------------|-----------------|----------------|
    | Q1       | 0% halluc.      | 0.82 consistent | YES ✓          |
    | Q2       | 67% halluc.     | 0.31 consistent | YES ✓          |
    | Q3       | 33% halluc.     | 0.55 consistent | PARTIAL        |

  If they agree ~70% of time → both methods are valid.
  If they disagree → interesting case study for your Analysis section!

BERTSCORE vs SIMPLE OVERLAP:
  Simple Overlap (Jaccard): Count common words. No install needed.
  BERTScore:                BERT reads meaning, not just words.
                            "raised" and "hiked" = same meaning → higher score.
  
  Both are valid for Week 1. Use BERTScore in Week 2 for better results.
"""

import json
import numpy as np
from typing import List, Dict


def jaccard_overlap(text1: str, text2: str) -> float:
    """
    Word-level Jaccard similarity. No extra libraries needed.

    Jaccard(A, B) = |A ∩ B| / |A ∪ B|
      = (words in both) / (all unique words)

    Example:
      text1 = "RBI raised rates to 6.75 percent"  → {RBI, raised, rates, to, 6.75, percent}
      text2 = "RBI hiked the repo rate"            → {RBI, hiked, the, repo, rate}
      intersection = {RBI}   → size 1
      union        = 10 words → size 10
      Jaccard = 1/10 = 0.10  (low — because "raised"≠"hiked" for Jaccard)
    """
    w1 = set(text1.lower().split())
    w2 = set(text2.lower().split())
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)


def bertscore_similarity(texts: List[str]) -> float:
    """
    BERTScore-based pairwise similarity.
    Understands MEANING — "raised" and "hiked" score high even though different words.

    Requires: pip install bert-score
    Falls back to Jaccard if not installed.
    """
    try:
        from bert_score import score as bs
        scores = []
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                _, _, F1 = bs([texts[i]], [texts[j]], lang="en", verbose=False)
                scores.append(float(F1.mean()))
        return float(np.mean(scores)) if scores else 1.0
    except ImportError:
        print("bert-score not installed → using Jaccard overlap instead.")
        print("Install later: pip install bert-score")
        scores = []
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                scores.append(jaccard_overlap(texts[i], texts[j]))
        return float(np.mean(scores)) if scores else 1.0


class SelfCheckGPT:
    """Consistency-based hallucination detection (no reference needed)."""

    def __init__(self, method: str = "jaccard"):
        """
        method: "jaccard"    → fast, no extra install, good for Week 1
                "bertscore"  → slower, more accurate, better for Week 2
        """
        assert method in ("jaccard", "bertscore"), "method must be 'jaccard' or 'bertscore'"
        self.method = method

    def score(self, samples: List[str]) -> Dict:
        """
        Score consistency across N answer samples.

        Parameters:
            samples: list of N answer strings (all for the same question)

        Returns:
            {
              "consistency_score": 0.72,       ← 0=inconsistent, 1=identical
              "hallucination_signal": "LOW",    ← LOW/MEDIUM/HIGH
              "pairwise_scores": [...],
              "interpretation": "..."
            }

        HOW TO INTERPRET:
          score > 0.70 → HIGH consistency  → model is confident → LOW hallucination risk
          score 0.40-0.70 → MEDIUM         → uncertain → validate with validator.py
          score < 0.40 → LOW consistency   → model is unsure → HIGH hallucination risk
        """
        if len(samples) < 2:
            return {"error": "Need at least 2 samples"}

        pairs = []
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                if self.method == "bertscore":
                    s = bertscore_similarity([samples[i], samples[j]])
                else:
                    s = jaccard_overlap(samples[i], samples[j])
                pairs.append(round(s, 4))

        avg = float(np.mean(pairs))

        if avg >= 0.70:
            signal        = "LOW"
            interpretation = "Model is consistent → likely knows the answer → low hallucination risk"
        elif avg >= 0.40:
            signal        = "MEDIUM"
            interpretation = "Moderate variation → verify with LLM validator"
        else:
            signal        = "HIGH"
            interpretation = "High inconsistency → model is uncertain → likely hallucinating"

        return {
            "consistency_score":    round(avg, 4),
            "hallucination_signal": signal,
            "method":               self.method,
            "n_samples":            len(samples),
            "pairwise_scores":      pairs,
            "interpretation":       interpretation,
        }

    def ablation_table_row(
        self,
        question:          str,
        selfcheck_result:  Dict,
        validator_result:  Dict,
    ) -> Dict:
        """
        Create one row of your ablation comparison table.

        This is what goes in your Results section (30 points):

        | Question | SelfCheck Score | Hallucination Rate | Agreement |
        |----------|-----------------|--------------------|-----------|
        | Q1       | 0.82 (LOW)      | 0.10               | YES       |
        | Q2       | 0.29 (HIGH)     | 0.67               | YES       |
        """
        sc_score   = selfcheck_result.get("consistency_score", -1)
        sc_signal  = selfcheck_result.get("hallucination_signal", "?")
        val_rate   = validator_result.get("hallucination_rate", -1)

        # Agreement: SelfCheck LOW ↔ Validator low hallucination rate
        # SelfCheck HIGH ↔ Validator high hallucination rate
        if sc_score >= 0 and val_rate >= 0:
            sc_predicts_clean   = sc_score >= 0.55
            val_predicts_clean  = val_rate <= 0.30
            agrees = sc_predicts_clean == val_predicts_clean
        else:
            agrees = None

        return {
            "question":              question[:60],
            "selfcheck_score":       sc_score,
            "selfcheck_signal":      sc_signal,
            "validator_halluc_rate": val_rate,
            "methods_agree":         agrees,
            "note": "" if agrees else "DISAGREE → analyze this case in report"
        }


# ─────────────────────────────────────────────────────────────
# QUICK TEST  (run: python src/selfcheck.py)
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    checker = SelfCheckGPT(method="jaccard")

    print("=" * 60)
    print("TEST 1: Consistent samples (model knows the answer)")
    consistent = [
        "The RBI raised the repo rate by 25 basis points to 6.75 percent.",
        "India's central bank hiked the repo rate by 25 bps to 6.75%.",
        "The Reserve Bank of India increased its policy rate by 25 bps to reach 6.75%."
    ]
    r1 = checker.score(consistent)
    print(json.dumps(r1, indent=2))

    print("\n" + "=" * 60)
    print("TEST 2: Inconsistent samples (hallucination signal!)")
    inconsistent = [
        "The RBI raised the repo rate by 25 basis points to 6.75 percent.",
        "The RBI cut interest rates by 50 basis points to support growth.",
        "The RBI kept rates unchanged citing global uncertainty."
    ]
    r2 = checker.score(inconsistent)
    print(json.dumps(r2, indent=2))

    print("\n" + "=" * 60)
    print("ABLATION ROW EXAMPLE:")
    mock_validator = {"hallucination_rate": 0.67}
    row = checker.ablation_table_row(
        "What did RBI do with rates?", r2, mock_validator
    )
    print(json.dumps(row, indent=2))
