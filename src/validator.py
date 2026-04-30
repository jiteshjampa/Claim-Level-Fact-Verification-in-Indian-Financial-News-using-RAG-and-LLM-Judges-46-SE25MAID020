"""
validator.py  ─  LLM-as-Judge Hallucination Validator
======================================================

THE CORE CONTRIBUTION OF YOUR PROJECT.

WHAT IT DOES:
  After generator.py produces an answer, this runs a SECOND LLM call.
  The second LLM acts as a "judge" — it reads the original context
  and checks every sentence of the answer:
    SUPPORTED    → sentence is backed by context  ✓
    UNSUPPORTED  → sentence not found in context  ✗ (hallucination!)
    CONTRADICTED → sentence conflicts with context ✗✗ (worse!)

WHY IS THIS YOUR CORE CONTRIBUTION?
  Most RAG systems just generate an answer and stop.
  You ADD a validation layer that catches hallucinations BEFORE
  the answer reaches the user. That's novel and graded at 30 points.

WEEK 1: Include this file. You can skip running it (no API key needed yet).
WEEK 2: Run it. Compute metrics. Compare with SelfCheckGPT.

ADVANCED ADDITIONS (to make project stronger):
  - Confidence scoring (not just label, but 0-100 confidence)
  - Multi-hop fact checking (trace facts back to specific sentences)
  - Fine-tuned validator (train a small model to do what the LLM does)
  - See "ADVANCED EXTENSIONS" at bottom of file.
"""

import os
import re
import json
from typing import List, Dict, Tuple, Optional


# ─────────────────────────────────────────────────────────────
# THE VALIDATOR PROMPT — The heart of your project
# This is what you'll modify and ablate in Week 2.
# ─────────────────────────────────────────────────────────────
VALIDATOR_PROMPT = """You are a strict financial fact-checker. Your job is to verify every claim in a generated answer against provided source documents.

RETRIEVED SOURCE DOCUMENTS (Ground Truth):
{context}

GENERATED ANSWER TO VERIFY:
{answer}

INSTRUCTIONS:
For EACH sentence in the GENERATED ANSWER, assign exactly one label:
  - SUPPORTED:    The exact claim (including numbers, names, dates) is explicitly stated in the sources
  - UNSUPPORTED:  The claim is absent from the sources (possible hallucination)  
  - CONTRADICTED: The claim directly conflicts with information in the sources

IMPORTANT:
- Be STRICT. "Approximately correct" is still UNSUPPORTED if the number doesn't match.
- Numbers must match exactly. "6.75%" is not "7%".
- Only use what is written in the sources above — not your general knowledge.

Return ONLY this JSON (no explanation text before or after):
{{
  "sentences": [
    {{
      "text": "exact sentence from answer",
      "label": "SUPPORTED",
      "reason": "Source 1 explicitly states this figure",
      "confidence": 0.95
    }},
    {{
      "text": "another sentence",
      "label": "UNSUPPORTED",
      "reason": "No mention of this claim in any source",
      "confidence": 0.88
    }}
  ],
  "hallucination_rate": 0.25,
  "supported_count": 3,
  "unsupported_count": 1,
  "contradicted_count": 0,
  "verdict": "MOSTLY_RELIABLE"
}}"""


class HallucinationValidator:
    """LLM-as-Judge validator for RAG-generated answers."""

    def __init__(self, model: str = "llama3-8b-8192"):
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass

        self.api_key = os.environ.get("GROQ_API_KEY", "")
        self.model   = model
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from groq import Groq
            self._client = Groq(api_key=self.api_key)
        return self._client

    def validate(self, answer: str, context: str) -> Dict:
        """
        Validate an answer against its source context.

        Parameters:
            answer:  generated text from generator.py
            context: the context string that was fed to the generator

        Returns dict with sentence labels + overall hallucination rate.
        """
        prompt = VALIDATOR_PROMPT.format(context=context, answer=answer)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a precise financial fact-checker. Return only valid JSON."},
                {"role": "user",   "content": prompt}
            ],
            temperature=0.0,     # MUST be 0 — deterministic fact-checking
            max_tokens=1500,
        )

        raw = response.choices[0].message.content.strip()

        try:
            # Strip any accidental markdown code fences
            clean  = re.sub(r"```(?:json)?|```", "", raw).strip()
            result = json.loads(clean)
        except json.JSONDecodeError as e:
            result = {
                "sentences":          [],
                "hallucination_rate": -1,
                "verdict":            "PARSE_ERROR",
                "error":              str(e),
                "raw_response":       raw[:300]
            }

        return result

    def batch_validate(self, qa_results: List[Dict]) -> List[Dict]:
        """
        Validate a batch of QA results.
        qa_results: list of outputs from generator.generate()
        """
        validated = []
        for i, result in enumerate(qa_results):
            print(f"  Validating {i+1}/{len(qa_results)}: {result['question'][:50]}...")
            v = self.validate(result["answer"], result["context_str"])
            result["validation"] = v
            validated.append(result)
        return validated

    def compute_aggregate_metrics(self, validated_results: List[Dict]) -> Dict:
        """
        Compute overall metrics across all validated results.

        METRICS YOUR REPORT NEEDS (for the 30-point Results section):
        ─────────────────────────────────────────────────────────────
        1. Hallucination Rate  = % of sentences labeled UNSUPPORTED or CONTRADICTED
        2. Support Rate        = % of sentences labeled SUPPORTED
        3. Per-query breakdown = which questions caused most hallucinations?

        In Week 2, you'll add:
        4. Precision/Recall    vs. human-labeled ground truth on 50 examples
        5. Agreement rate      with SelfCheckGPT consistency scores
        """
        all_sentences = []
        for r in validated_results:
            v = r.get("validation", {})
            all_sentences.extend(v.get("sentences", []))

        if not all_sentences:
            return {"error": "No sentences found in validation results"}

        counts = {"SUPPORTED": 0, "UNSUPPORTED": 0, "CONTRADICTED": 0}
        for s in all_sentences:
            lbl = s.get("label", "UNSUPPORTED").upper()
            counts[lbl] = counts.get(lbl, 0) + 1

        total = len(all_sentences)
        hallucination_rate = (counts["UNSUPPORTED"] + counts["CONTRADICTED"]) / total

        return {
            "total_sentences":    total,
            "counts":             counts,
            "hallucination_rate": round(hallucination_rate, 4),
            "support_rate":       round(counts["SUPPORTED"] / total, 4),
            "n_queries":          len(validated_results),
            "summary": (
                f"{counts['UNSUPPORTED']} unsupported + {counts['CONTRADICTED']} contradicted "
                f"out of {total} total sentences = {hallucination_rate:.1%} hallucination rate"
            )
        }


# ─────────────────────────────────────────────────────────────
# ADVANCED EXTENSIONS (add these to make project stronger)
# ─────────────────────────────────────────────────────────────
class AdvancedValidator(HallucinationValidator):
    """
    Extended validator with additional features for stronger project.
    Include 1-2 of these to differentiate from a basic implementation.
    """

    MULTI_PROMPT_TEMPLATE = """You are validating the same answer using a different approach.
    
Instead of sentence-by-sentence, identify KEY CLAIMS in the answer
(numbers, names, organizations, dates, percentages, events).

For each key claim, verify against context:

CONTEXT:
{context}

ANSWER:
{answer}

Return JSON:
{{
  "key_claims": [
    {{"claim": "RBI raised rates to 6.75%", "verified": true, "source_sentence": "..."}},
    {{"claim": "Markets rose 500 points", "verified": false, "source_sentence": null}}
  ],
  "unverified_claim_rate": 0.33
}}"""

    def validate_key_claims(self, answer: str, context: str) -> Dict:
        """
        Alternative validation: extract and verify KEY CLAIMS (numbers, names, facts).
        This is your ABLATION VARIANT — compare it with sentence-level validation.

        ABLATION means: "What if we did it differently? Does it change results?"
        Comparing two validation strategies = 1 ablation for your results section.
        """
        prompt = self.MULTI_PROMPT_TEMPLATE.format(context=context, answer=answer)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Return only valid JSON."},
                {"role": "user",   "content": prompt}
            ],
            temperature=0.0,
            max_tokens=800,
        )
        raw = response.choices[0].message.content.strip()
        try:
            clean = re.sub(r"```(?:json)?|```", "", raw).strip()
            return json.loads(clean)
        except Exception:
            return {"error": "parse failed", "raw": raw[:200]}

    def ensemble_validate(self, answer: str, context: str) -> Dict:
        """
        Run BOTH validation strategies and combine results.
        Ensemble = using multiple approaches and combining.
        This is a strong addition that shows sophisticated methodology.
        """
        sent_result  = self.validate(answer, context)
        claim_result = self.validate_key_claims(answer, context)

        # Combine scores
        sent_halluc  = sent_result.get("hallucination_rate", -1)
        claim_halluc = claim_result.get("unverified_claim_rate", -1)

        if sent_halluc >= 0 and claim_halluc >= 0:
            combined = (sent_halluc + claim_halluc) / 2
        else:
            combined = max(sent_halluc, claim_halluc)

        return {
            "sentence_level":    sent_result,
            "claim_level":       claim_result,
            "combined_hallucination_rate": round(combined, 4),
            "verdict": "HIGH_RISK" if combined > 0.3 else "LOW_RISK"
        }


# ─────────────────────────────────────────────────────────────
# QUICK TEST  (run: python src/validator.py)
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test with a known hallucination
    context = """[Source 1 — RBI Rate Decision (relevance=0.91)]
The Reserve Bank of India unanimously raised the repo rate by 25 basis points to 6.75 percent 
on February 8, 2024. Governor Shaktikanta Das said the move targets inflation at 5.69 percent.
The Sensex fell 450 points after the announcement."""

    good_answer = ("The RBI raised the repo rate by 25 basis points to 6.75 percent "
                   "on February 8, 2024. Governor Das cited inflation as the reason.")

    bad_answer  = ("The RBI raised the repo rate by 25 basis points to 6.75 percent. "
                   "Markets celebrated with Sensex rising 800 points on the decision. "    # CONTRADICTED
                   "Analysts expect three more rate hikes in 2024.")                        # UNSUPPORTED

    validator = HallucinationValidator()

    if not validator.api_key:
        print("No GROQ_API_KEY set. Showing prompt that would be sent:\n")
        print(VALIDATOR_PROMPT.format(context=context, answer=bad_answer)[:600])
        print("\n[Set GROQ_API_KEY to run actual validation]")
    else:
        print("Testing good answer:")
        print(json.dumps(validator.validate(good_answer, context), indent=2))

        print("\nTesting hallucinated answer:")
        print(json.dumps(validator.validate(bad_answer, context), indent=2))
