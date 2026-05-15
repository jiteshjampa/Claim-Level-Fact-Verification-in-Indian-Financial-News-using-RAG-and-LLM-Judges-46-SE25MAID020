"""
validator.py — LLM-as-Judge hallucination validator.

Labels each sentence in the generated answer as:
  SUPPORTED      — fact is clearly present and matches the source
  UNSUPPORTED    — fact is absent from source (model added it from parametric knowledge)
  CONTRADICTED   — answer conflicts with what source says

Three prompt variants for ablation:
  strict          — default; hard number/date matching
  lenient         — accepts paraphrasing and approximate phrasing
  chain_of_thought — requires enumeration of key facts before labeling
"""

import re
import time
from typing import List, Dict, Tuple
from generator import _groq_call

# ── Prompt templates ──────────────────────────────────────────────────────────

_STRICT_SYSTEM = """You are a strict fact-checking assistant for financial content.
Your job: given SOURCE DOCUMENTS and a GENERATED ANSWER, label EVERY sentence in the answer.

Rules:
- SUPPORTED: the exact fact (number, entity, action) is present in sources.
  "Approximately correct" is still UNSUPPORTED. 6.5% ≠ 6.75%.
- UNSUPPORTED: fact not found in any source, even if plausible.
- CONTRADICTED: fact is explicitly contradicted by a source.
- Do NOT use any external knowledge. Only what is in the sources.
- Output format — one line per sentence:
  SENTENCE: <sentence text>
  LABEL: <SUPPORTED|UNSUPPORTED|CONTRADICTED>
  REASON: <one-sentence justification citing source text>
  ---"""

_LENIENT_SYSTEM = """You are a fact-checking assistant for financial content.
Your job: given SOURCE DOCUMENTS and a GENERATED ANSWER, label EVERY sentence in the answer.

Rules:
- SUPPORTED: the fact is present in sources, including acceptable paraphrases or near-equivalent phrasing.
  "Hiked rates" is the same as "raised rates". Minor lexical variation is fine.
- UNSUPPORTED: fact is absent from all sources and cannot be inferred.
- CONTRADICTED: fact is explicitly contradicted by a source.
- Output format — one line per sentence:
  SENTENCE: <sentence text>
  LABEL: <SUPPORTED|UNSUPPORTED|CONTRADICTED>
  REASON: <one-sentence justification>
  ---"""

_COT_SYSTEM = """You are a careful financial fact-checker using chain-of-thought reasoning.

Step 1: List the KEY FACTS mentioned in the source documents (numbers, entities, dates, events).
Step 2: For each sentence in the generated answer, check if its facts appear in Step 1.
Step 3: Label each sentence:
  SUPPORTED      — fact found in Step 1, exact or near-exact
  UNSUPPORTED    — fact not found in Step 1
  CONTRADICTED   — fact conflicts with Step 1

Output format:
KEY_FACTS: <bullet list of key facts from sources>
---
SENTENCE: <sentence text>
LABEL: <SUPPORTED|UNSUPPORTED|CONTRADICTED>
REASON: <which key fact supports/refutes this>
---"""

PROMPT_TEMPLATES = {
    "strict":          _STRICT_SYSTEM,
    "lenient":         _LENIENT_SYSTEM,
    "chain_of_thought": _COT_SYSTEM,
}

VALID_LABELS = {"SUPPORTED", "UNSUPPORTED", "CONTRADICTED"}


def _split_sentences(text: str) -> List[str]:
    """Simple sentence splitter."""
    # split on . ? ! followed by space/newline
    raw = re.split(r'(?<=[.?!])\s+', text.strip())
    return [s.strip() for s in raw if len(s.strip()) > 10]


def _parse_labels(response: str) -> List[Dict]:
    """Parse structured label output from the validator LLM."""
    results = []
    blocks = re.split(r'\n---+\n?', response)
    for block in blocks:
        sentence_match = re.search(r'SENTENCE:\s*(.+)', block)
        label_match    = re.search(r'LABEL:\s*(SUPPORTED|UNSUPPORTED|CONTRADICTED)', block)
        reason_match   = re.search(r'REASON:\s*(.+)', block)
        if sentence_match and label_match:
            results.append({
                "sentence": sentence_match.group(1).strip(),
                "label":    label_match.group(1).strip(),
                "reason":   reason_match.group(1).strip() if reason_match else "",
            })
    return results


def validate(
    answer: str,
    chunks: List[Dict],
    prompt_variant: str = "strict",
) -> Dict:
    """
    Run LLM-as-Judge validation on a generated answer.

    Returns
    -------
    {
        "sentence_labels": [...],      # list of {sentence, label, reason}
        "hallucination_rate": float,   # fraction of non-SUPPORTED sentences
        "supported_rate":    float,
        "verdict":           str,      # MOSTLY_RELIABLE / PARTIALLY_RELIABLE / UNRELIABLE
        "raw_response":      str,
        "prompt_variant":    str,
        "counts":            {SUPPORTED, UNSUPPORTED, CONTRADICTED}
    }
    """
    system_prompt = PROMPT_TEMPLATES.get(prompt_variant, _STRICT_SYSTEM)

    context = "\n\n".join(
        f"[SOURCE {i+1}] {c.get('title','')}\n{c['text']}"
        for i, c in enumerate(chunks)
    )

    user_msg = (
        f"SOURCE DOCUMENTS:\n{context}\n\n"
        f"GENERATED ANSWER:\n{answer}\n\n"
        "Now label every sentence in the GENERATED ANSWER."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_msg},
    ]

    raw_response = _groq_call(messages, temperature=0.0, max_tokens=1500)
    sentence_labels = _parse_labels(raw_response)

    # Fallback: if parsing failed, try to count sentences manually
    if not sentence_labels:
        sentences = _split_sentences(answer)
        sentence_labels = [
            {"sentence": s, "label": "UNSUPPORTED", "reason": "parse failed"}
            for s in sentences
        ]

    counts = {"SUPPORTED": 0, "UNSUPPORTED": 0, "CONTRADICTED": 0}
    for item in sentence_labels:
        lbl = item["label"] if item["label"] in VALID_LABELS else "UNSUPPORTED"
        counts[lbl] += 1

    total = len(sentence_labels) or 1
    hallucination_rate = (counts["UNSUPPORTED"] + counts["CONTRADICTED"]) / total
    supported_rate = counts["SUPPORTED"] / total

    if hallucination_rate < 0.15:
        verdict = "MOSTLY_RELIABLE"
    elif hallucination_rate < 0.40:
        verdict = "PARTIALLY_RELIABLE"
    else:
        verdict = "UNRELIABLE"

    return {
        "sentence_labels":    sentence_labels,
        "hallucination_rate": round(hallucination_rate, 4),
        "supported_rate":     round(supported_rate, 4),
        "verdict":            verdict,
        "raw_response":       raw_response,
        "prompt_variant":     prompt_variant,
        "counts":             counts,
    }
