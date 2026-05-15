# Hallucination Detection in LLM-based Financial RAG Systems

**CS5202 — Generative AI & LLMs · Spring 2026 · Department of CS & AI**

Dataset: `kdave/Indian_Financial_News` (26,000 articles) · Final Evaluation — May 15, 2026

---

## Problem

LLMs hallucinate. In finance, a hallucinated fact — "RBI cut rates to support growth" when it raised them — can directly mislead retail investors or automated trading systems.

This project builds a RAG pipeline over 26,000 Indian financial news articles and adds **two complementary hallucination-detection layers**:

1. **LLM-as-Judge Validator** — labels each generated sentence as `SUPPORTED`, `UNSUPPORTED`, or `CONTRADICTED` against retrieved source chunks.
2. **SelfCheckGPT** — consistency scoring across multiple stochastic samples (Jaccard and BERTScore variants).

---

## Quick Start

```bash
# 1. Clone and enter
cd project-46-SE25MAID020

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set Groq API key (free at https://console.groq.com)
export GROQ_API_KEY="gsk_your_key_here"

# 4. Download full dataset (26k articles from HuggingFace)
python src/load_dataset.py

# 5. Run single query
python main.py --query "What did RBI do with interest rates?"

# 6. Run full evaluation (30 questions + 3 ablation studies)
python main.py --evaluate

# 7. Paper comparison table
python main.py --compare_papers

# 8. Generate all figures
python src/plot_results.py
```

---

## Architecture

```
User Query
    ↓
[1] load_dataset.py — 26,000 Indian Financial News articles (HuggingFace)
    ↓
[2] retriever.py — sentence-transformers (all-MiniLM-L6-v2) + FAISS IndexFlatIP
    ↓ top-3 most relevant chunks (250-word, 40-word overlap)
[3] generator.py — Groq / Llama3-8B (temp=0.1, grounded prompt)
    ↓ draft answer
[4] validator.py — LLM-as-Judge (2nd Groq call, temp=0.0)     ← CORE CONTRIBUTION
    ↓ SUPPORTED / UNSUPPORTED / CONTRADICTED per sentence
[5] selfcheck.py — SelfCheckGPT (N=3 samples, Jaccard + BERTScore)
    ↓
Final output + hallucination report
```

---

## Project Structure

```
project-46-SE25MAID020/
├── main.py                      ← Full pipeline demo (run this)
├── requirements.txt             ← Pinned dependencies
├── domain_note.pdf              ← 1-page domain note (Milestone 1)
├── report.pdf                   ← 4–6 page final report (Final Evaluation)
├── data/
│   └── financial_news.json      ← 26k articles (downloaded by load_dataset.py)
├── src/
│   ├── load_dataset.py          ← Download from HuggingFace
│   ├── retriever.py             ← FAISS + sentence-transformers
│   ├── generator.py             ← RAG answer generation (Groq/Llama3)
│   ├── validator.py             ← LLM-as-Judge (strict/lenient/CoT variants)
│   ├── selfcheck.py             ← SelfCheckGPT (Jaccard + BERTScore)
│   ├── evaluate.py              ← Full 30-question evaluation + 3 ablation studies
│   ├── paper_comparison.py      ← Compare vs SelfCheckGPT/Self-RAG/LRP4RAG
│   └── plot_results.py          ← Generate all report figures
├── notebooks/
│   └── exploration.ipynb        ← Data exploration
└── results/
    ├── evaluation_results.json  ← Full evaluation output
    ├── paper_comparison.json    ← Paper comparison data
    ├── fig1_hallucination_distribution.png
    ├── fig2_chunk_ablation.png
    ├── fig3_selfcheck_scatter.png
    ├── fig4_paper_comparison.png
    ├── fig5_topk_ablation.png
    └── fig6_prompt_ablation.png
```

---

## Key Results (Final Evaluation)

| Metric | Value | Target | Status |
|---|---|---|---|
| Avg retrieval similarity | 0.614 | ≥ 0.55 | ✓ |
| Avg hallucination rate | 18.3% | < 20% | ✓ |
| Avg support rate | 81.7% | > 80% | ✓ |
| Validator precision | 0.78 | > 0.75 | ✓ |
| Validator recall | 0.71 | > 0.70 | ✓ |
| SelfCheck consistency | 0.741 | — | — |
| Method agreement | 73.3% | — | — |

---

## Ablation Studies

**Chunk Size** — 250-word chunks (40-word overlap) achieve best retrieval score (0.614).

**Prompt Variant** — Strict prompt used as default; Chain-of-Thought achieves best recall (24.3% detection).

**Top-K** — K=3 achieves best balance (18.3% hallucination rate vs 26.4% at K=1).

---

## Paper Comparison

| System | Domain | Precision | Recall | F1 |
|---|---|---|---|---|
| **Our System** | Indian Financial News | **0.78** | **0.71** | **0.74** |
| SelfCheckGPT (2303.08896) | Wikipedia Biography | 0.73 | 0.69 | 0.71 |
| Self-RAG (2310.11511) | Open-domain QA | 0.82 | 0.74 | 0.78 |
| LRP4RAG (2408.15533) | Document QA | 0.76 | 0.72 | 0.74 |

Our system is training-free and works with any black-box LLM API, unlike Self-RAG (requires fine-tuning) and LRP4RAG (requires white-box model access).

---

## Groq API Key (Free)

1. Go to [https://console.groq.com](https://console.groq.com)
2. Sign up — no credit card needed
3. API Keys → Create New Key (starts with `gsk_...`)
4. `export GROQ_API_KEY="gsk_your_key"`

---

## References

1. Manakul et al. (2023). SelfCheckGPT. arXiv:2303.08896
2. Asai et al. (2023). Self-RAG. arXiv:2310.11511
3. Anonymous (2024). LRP4RAG. arXiv:2408.15533
4. Lewis et al. (2020). RAG for Knowledge-Intensive NLP. NeurIPS 2020. arXiv:2005.11401
5. Dave, K. (2023). Indian Financial News. HuggingFace: kdave/Indian_Financial_News
6. Reimers & Gurevych (2019). Sentence-BERT. EMNLP 2019. arXiv:1908.10084

---

## Contributors

| Name | Roll No |
|---|---|
| J. Jitesh Reddy | SE25MAID020 |
| M. Tridev | SE25MAID026 |
| M. Mahesh | SE25MAID037 |
| T. Varun | SE25MAID009 |

**CS5202 · Spring 2026 · Department of CS & AI**
