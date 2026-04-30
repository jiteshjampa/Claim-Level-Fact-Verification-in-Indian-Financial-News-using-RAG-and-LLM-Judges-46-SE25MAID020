"""
generator.py  ─  RAG Answer Generation
=======================================

WHAT HAPPENS HERE:
  retriever.py found the top-3 relevant news chunks.
  Now we feed those chunks + the question to an LLM.
  The LLM reads the chunks and writes a grounded answer.

KEY TERMS:
  PROMPT:        The full text you send to the LLM.
                 It has two parts:
                   1. System prompt  = "You are a financial analyst..."
                   2. User prompt    = "Context: [3 chunks] \nQuestion: ..."

  TEMPERATURE:   Controls how creative/random the LLM is.
                   0.0 = always gives the same answer (good for facts)
                   0.7 = adds variety (needed for SelfCheckGPT in Week 2)
                   1.0 = very creative (bad for factual QA)

  MAX_TOKENS:    Maximum length of the answer.
                   500 tokens ≈ 375 words ≈ 2-3 paragraphs

  CONTEXT WINDOW: How much the LLM can see at once.
                   Our prompt = 3 chunks (~750 words ≈ 1000 tokens)
                              + question (~20 tokens)
                              = ~1020 tokens total
                   Llama3-8B limit = 8192 tokens → we're safely inside.

  HALLUCINATION: When the LLM makes up facts not in the context.
                 Example: Context says "RBI raised rates to 6.75%"
                 LLM says "RBI raised rates to 7.0%" ← hallucination!
                 Your validator.py catches this.

FREE API — GROQ:
  Groq gives you free access to Llama3, Mixtral, Gemma.
  Speed: 500+ tokens/second (much faster than OpenAI's free tier).
  Get key: https://console.groq.com → Sign up → API Keys → Create
"""

import os
import json
from typing import List, Tuple, Dict, Optional


# ─────────────────────────────────────────────────────────────
# GROQ MODEL OPTIONS
# ─────────────────────────────────────────────────────────────
# Model name              Context    Speed    Quality
# llama3-8b-8192          8k tokens  fastest  good for facts
# llama3-70b-8192         8k tokens  slower   better reasoning
# mixtral-8x7b-32768      32k tokens fast     good for long contexts
# gemma2-9b-it            8k tokens  fast     good

DEFAULT_MODEL = "llama3-8b-8192"


# ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a precise Indian financial news analyst.
Your job is to answer questions using ONLY the provided news article excerpts.

Rules you must follow:
1. Use ONLY facts from the provided context — never add outside knowledge
2. If context doesn't answer the question, say: "The retrieved articles do not contain enough information."
3. Cite which source (Source 1, 2, or 3) each fact comes from
4. Be concise — 3-5 sentences maximum
5. Never fabricate numbers, names, dates, or statistics"""


def build_prompt(question: str, context: str) -> str:
    """
    Build the RAG prompt.

    WHY THIS STRUCTURE?
    ─────────────────────
    Standard RAG prompt = Context first, then question.
    Putting context first = LLM pays more attention to it.
    Adding "ONLY use context below" = reduces hallucinations significantly.

    Research shows this prompt structure reduces hallucination by ~30%
    compared to just asking the question directly.
    """
    return f"""Use ONLY the context below to answer the question. Do not use any outside knowledge.

=== RETRIEVED CONTEXT ===
{context}
=========================

QUESTION: {question}

ANSWER (cite sources, e.g. "According to Source 1..."):"""


# ─────────────────────────────────────────────────────────────
class RAGGenerator:
    """
    Generates answers using RAG (Retrieval-Augmented Generation).
    Wraps the Groq API with clean error handling.
    """

    def __init__(self, model: str = DEFAULT_MODEL):
        """
        HOW TO SET YOUR API KEY (step by step):
        ─────────────────────────────────────────
        Option A (recommended for development):
          Create a file called  .env  in your project root:
            GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
          Then load it: python-dotenv reads it automatically.

        Option B (terminal):
          Linux/Mac: export GROQ_API_KEY="gsk_xxx..."
          Windows:   set GROQ_API_KEY=gsk_xxx...

        Option C (hardcode — only for testing, never commit to git!):
          api_key = "gsk_xxx..."
        """
        # Try to load from .env file
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass

        self.api_key = os.environ.get("GROQ_API_KEY", "")
        if not self.api_key:
            print("WARNING: GROQ_API_KEY not set.")
            print("Get a free key at https://console.groq.com")
            print("Then set: export GROQ_API_KEY='your_key'")

        self.model = model
        self._client = None

    @property
    def client(self):
        """Lazy-load Groq client."""
        if self._client is None:
            from groq import Groq
            self._client = Groq(api_key=self.api_key)
        return self._client

    def generate(
        self,
        question: str,
        retrieved_chunks: List[Tuple[str, Dict, float]],
        temperature: float = 0.1,
        max_tokens: int = 500,
    ) -> Dict:
        """
        Generate a grounded answer using retrieved context.

        Parameters:
            question:         The user's query
            retrieved_chunks: Output from retriever.retrieve()
            temperature:      0.1 for factual answers
            max_tokens:       Max answer length in tokens

        Returns dict with answer, context, token counts.
        """
        # Format context from chunks
        context_parts = []
        for i, (chunk, meta, score) in enumerate(retrieved_chunks, 1):
            context_parts.append(
                f"[Source {i} — {meta.get('title','?')} (relevance={score:.2f})]\n{chunk}"
            )
        context_str = "\n\n".join(context_parts)

        prompt = build_prompt(question, context_str)

        # Call Groq API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        answer = response.choices[0].message.content.strip()
        usage  = response.usage

        return {
            "question":          question,
            "answer":            answer,
            "context_str":       context_str,
            "retrieved_chunks":  retrieved_chunks,
            "model":             self.model,
            "temperature":       temperature,
            # Token tracking — important for understanding LLM costs/limits
            "prompt_tokens":     usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens":      usage.total_tokens,
        }

    def generate_multiple(
        self,
        question: str,
        retrieved_chunks: List[Tuple[str, Dict, float]],
        n_samples: int = 3,
        temperature: float = 0.7,   # Higher = more variation between samples
    ) -> List[Dict]:
        """
        Generate N different answers to the same question.
        Used by selfcheck.py (SelfCheckGPT) in Week 2.

        WHY TEMPERATURE=0.7 HERE?
          With temp=0.7, the LLM adds randomness to each generation.
          If the model truly "knows" → all 3 answers stay consistent.
          If it's guessing/hallucinating → 3 answers contradict each other.
          That inconsistency IS your hallucination signal.
        """
        results = []
        for i in range(n_samples):
            print(f"  Generating sample {i+1}/{n_samples}...")
            r = self.generate(question, retrieved_chunks, temperature=temperature)
            r["sample_id"] = i
            results.append(r)
        return results


# ─────────────────────────────────────────────────────────────
# FULL PIPELINE TEST  (run: python src/generator.py)
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from retriever import FinancialRetriever
    import json

    # Step 1: Load or build index
    retriever = FinancialRetriever()
    try:
        retriever.load_index()
    except FileNotFoundError:
        print("No index found. Run 'python src/retriever.py' first.")
        exit(1)

    # Step 2: Generate answers
    generator = RAGGenerator()
    if not generator.api_key:
        print("\nSkipping API test — no GROQ_API_KEY set.")
        print("Set it and re-run this file.")
        exit(0)

    test_questions = [
        "What did the RBI decide about interest rates?",
        "How did Infosys perform in Q3?",
        "What new rules did SEBI propose for trading?",
    ]

    results = []
    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        chunks = retriever.retrieve(q)
        result = generator.generate(q, chunks)
        print(f"A: {result['answer']}")
        print(f"Tokens used: {result['total_tokens']}")
        results.append(result)

    # Save preliminary results for Week 1 demo
    os.makedirs("results", exist_ok=True)
    with open("results/preliminary_results.json", "w") as f:
        # Save only serialisable parts
        for r in results:
            r["retrieved_chunks"] = [
                {"chunk": c[:200], "title": m["title"], "score": s}
                for c, m, s in r["retrieved_chunks"]
            ]
        json.dump(results, f, indent=2)
    print("\nResults saved to results/preliminary_results.json")
