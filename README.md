# Financial RAG with Hallucination Validator
### Project — Generative AI | Indian Financial News QA

---

## Problem

LLMs hallucinate. In finance, a hallucinated fact ("RBI cut rates" when it raised them) 
can cause wrong investment decisions. This project builds a RAG system with an 
**LLM-as-Judge validator** that catches unsupported claims before they reach users.

---

## Quick Start

```bash
# 1. Clone and enter project
cd project-<id>-<rollno>

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your free Groq API key
export GROQ_API_KEY="gsk_your_key_here"

# 4. Download dataset
python src/load_dataset.py

# 5. Run the full pipeline
python main.py --query "What did RBI do with interest rates?"
```

---

## Project Structure

```
project/
├── main.py                     ← Full pipeline demo (run this!)
├── requirements.txt            ← Pinned dependencies
├── data/
│   └── financial_news.json     ← Indian Financial News (2000 articles)
├── src/
│   ├── load_dataset.py         ← Download from HuggingFace
│   ├── retriever.py            ← FAISS + embeddings
│   ├── generator.py            ← RAG answer generation (Groq/Llama3)
│   ├── validator.py            ← LLM-as-judge hallucination checker ★
│   └── selfcheck.py            ← SelfCheckGPT consistency scoring
├── notebooks/
│   └── exploration.ipynb       ← Data exploration
├── results/
│   └── week1_demo_output.json  ← Preliminary results
└── domain_note.pdf             ← 1-page domain note (Milestone 1)
```

---

## Architecture

```
User Query
    ↓
[retriever.py] — sentence-transformers + FAISS
    ↓ top-3 relevant chunks
[generator.py] — Groq Llama3 with grounded prompt
    ↓ draft answer
[validator.py] — LLM-as-judge (2nd Groq call)       ← CORE CONTRIBUTION
    ↓ labels each sentence: SUPPORTED / UNSUPPORTED / CONTRADICTED
[selfcheck.py] — SelfCheckGPT (3 samples + consistency score)
    ↓
Final output + hallucination report
```

---

## Dataset

**Indian Financial News** — `kdave/Indian_Financial_News` on HuggingFace  
- 26,000+ articles from Economic Times, Mint, Business Standard  
- Used: 2,000 articles for Week 1; full dataset for Week 2  

---

## Getting a Free Groq API Key

1. Go to [https://console.groq.com](https://console.groq.com)
2. Sign up (free — no credit card)
3. Go to API Keys → Create New Key
4. Copy the key (starts with `gsk_...`)
5. Run: `export GROQ_API_KEY="gsk_your_key"`

Groq gives free access to: Llama3-8B, Llama3-70B, Mixtral-8x7B

---

## Contributors
1 M Tridev -SE25MAID026
2 M. Mahesh -SE25MAID037
3 T.Varun - SE25MAID009
4 J Jitesh Reddy -SE25MAID020


