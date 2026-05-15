"""
selfcheck.py — SelfCheckGPT consistency scoring.

Implements two similarity measures:
  1. Jaccard overlap (token-level)  — Week 1 baseline
  2. BERTScore (semantic)           — Final evaluation

Reference: Manakul et al. (2023). SelfCheckGPT: Zero-Resource Black-Box
Hallucination Detection. arXiv:2303.08896.

Risk bands:
  LOW    ≥ 0.70   — model is consistent, likely reliable
  MEDIUM  0.40–0.70
  HIGH   < 0.40   — inconsistent, high hallucination risk
"""

import re
from itertools import combinations
from typing import List, Tuple, Dict

LOW_THRESHOLD    = 0.70
MEDIUM_THRESHOLD = 0.40


# ── Similarity measures ───────────────────────────────────────────────────────

def _tokenize(text: str) -> set:
    return set(re.findall(r'\b\w+\b', text.lower()))


def _jaccard(a: str, b: str) -> float:
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta and not tb:
        return 1.0
    return len(ta & tb) / len(ta | tb)


def _bertscore_pairwise(samples: List[str]) -> float:
    """Compute mean BERTScore F1 across all pairs."""
    try:
        from bert_score import score as bs_score
        pairs = list(combinations(range(len(samples)), 2))
        if not pairs:
            return 1.0
        refs  = [samples[j] for i, j in pairs]
        cands = [samples[i] for i, j in pairs]
        P, R, F = bs_score(cands, refs, lang="en", verbose=False)
        return float(F.mean().item())
    except ImportError:
        print("[selfcheck] bert_score not installed, falling back to Jaccard.")
        return _jaccard_consistency(samples)


def _jaccard_consistency(samples: List[str]) -> float:
    """Mean pairwise Jaccard overlap across all sample pairs."""
    pairs = list(combinations(samples, 2))
    if not pairs:
        return 1.0
    scores = [_jaccard(a, b) for a, b in pairs]
    return sum(scores) / len(scores)


# ── Public API ────────────────────────────────────────────────────────────────

def selfcheck(
    samples: List[str],
    use_bertscore: bool = True,
) -> Dict:
    """
    Compute SelfCheckGPT consistency score for N answer samples.

    Parameters
    ----------
    samples        : list of answer strings (N ≥ 2, typically 3)
    use_bertscore  : if True, use BERTScore; else use Jaccard

    Returns
    -------
    {
        "consistency_score": float,   # 0–1, higher = more consistent
        "risk_level":        str,     # LOW / MEDIUM / HIGH
        "method":            str,     # jaccard / bertscore
        "n_samples":         int,
        "pairwise_scores":   list,    # individual pair scores
    }
    """
    if len(samples) < 2:
        return {
            "consistency_score": 1.0,
            "risk_level": "LOW",
            "method": "n/a",
            "n_samples": len(samples),
            "pairwise_scores": [],
        }

    method = "bertscore" if use_bertscore else "jaccard"

    if use_bertscore:
        try:
            from bert_score import score as _
            consistency = _bertscore_pairwise(samples)
        except Exception:
            method = "jaccard"
            consistency = _jaccard_consistency(samples)
    else:
        consistency = _jaccard_consistency(samples)

    pairs = list(combinations(samples, 2))
    if method == "jaccard":
        pairwise = [_jaccard(a, b) for a, b in pairs]
    else:
        try:
            from bert_score import score as bs_score
            refs  = [b for a, b in pairs]
            cands = [a for a, b in pairs]
            P, R, F = bs_score(cands, refs, lang="en", verbose=False)
            pairwise = [float(f) for f in F]
        except Exception:
            pairwise = [_jaccard(a, b) for a, b in pairs]

    if consistency >= LOW_THRESHOLD:
        risk = "LOW"
    elif consistency >= MEDIUM_THRESHOLD:
        risk = "MEDIUM"
    else:
        risk = "HIGH"

    return {
        "consistency_score": round(consistency, 4),
        "risk_level":        risk,
        "method":            method,
        "n_samples":         len(samples),
        "pairwise_scores":   [round(s, 4) for s in pairwise],
    }
