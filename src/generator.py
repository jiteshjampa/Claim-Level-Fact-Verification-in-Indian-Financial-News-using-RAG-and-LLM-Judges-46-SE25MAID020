"""
generator.py — RAG answer generation using Groq / Llama3-8B.

Builds a grounded prompt (context → question) and returns the generated answer.
temperature=0.1 for near-deterministic grounded answers.
"""

import os
import time
from typing import List, Dict

GROQ_MODEL   = "llama3-8b-8192"
MAX_TOKENS   = 500
TEMPERATURE  = 0.1


def _groq_call(messages: List[Dict], temperature: float = TEMPERATURE, max_tokens: int = MAX_TOKENS) -> str:
    """Call Groq API and return assistant text."""
    from groq import Groq
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set. Export it: export GROQ_API_KEY='gsk_...'")
    client = Groq(api_key=api_key)
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                wait = 2 ** attempt * 5
                print(f"[generator] Rate limit hit, waiting {wait}s ...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Groq API failed after 3 retries")


RAG_SYSTEM_PROMPT = """You are a precise financial analyst assistant.
Answer ONLY using information present in the provided source documents.
If the source documents do not contain enough information to answer, say so explicitly.
Do not use external knowledge. Be specific — include numbers, dates, and entities when present in sources."""


def generate_answer(query: str, chunks: List[Dict]) -> str:
    """
    Generate a grounded answer for query given retrieved chunks.

    Parameters
    ----------
    query  : user question
    chunks : list of retrieved chunk dicts from retriever

    Returns
    -------
    answer : generated answer string
    """
    context_parts = []
    for i, c in enumerate(chunks, 1):
        header = f"[SOURCE {i}]"
        if c.get("title"):
            header += f" {c['title']}"
        if c.get("date"):
            header += f" ({c['date']})"
        context_parts.append(f"{header}\n{c['text']}")

    context = "\n\n".join(context_parts)

    user_msg = (
        f"SOURCE DOCUMENTS:\n{context}\n\n"
        f"QUESTION: {query}\n\n"
        "Answer based strictly on the source documents above."
    )

    messages = [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg},
    ]

    return _groq_call(messages, temperature=TEMPERATURE)


def generate_samples(query: str, chunks: List[Dict], n: int = 3) -> List[str]:
    """
    Generate N stochastic samples of the answer (for SelfCheckGPT).
    temperature=0.7 to introduce variation.
    """
    context_parts = []
    for i, c in enumerate(chunks, 1):
        header = f"[SOURCE {i}]"
        if c.get("title"):
            header += f" {c['title']}"
        context_parts.append(f"{header}\n{c['text']}")
    context = "\n\n".join(context_parts)

    user_msg = (
        f"SOURCE DOCUMENTS:\n{context}\n\n"
        f"QUESTION: {query}\n\n"
        "Answer based strictly on the source documents above."
    )
    messages = [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg},
    ]

    samples = []
    for i in range(n):
        ans = _groq_call(messages, temperature=0.7)
        samples.append(ans)
        time.sleep(0.5)   # small pause to avoid rate limits
    return samples
