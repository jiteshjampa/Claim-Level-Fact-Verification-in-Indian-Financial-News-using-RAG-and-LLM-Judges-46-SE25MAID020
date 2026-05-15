# Financial RAG + Hallucination Detection
**WEB LINK **: https://halrag-git-main-jitesh-reddys-projects.vercel.app/
**CS5202 — Generative AI & LLMs · Spring 2026**

> Retrieval-Augmented Generation over 26,000 Indian financial news articles with dual-layer hallucination detection: LLM-as-Judge Validator + SelfCheckGPT consistency scoring.

---

## Problem

LLMs hallucinate in financial QA — generating confident but factually wrong answers. In finance, this is costly: a model that retrieves "RBI raised repo rate to 6.75%" but generates "RBI cut rates" could directly mislead investors. This project adds a validation layer that catches hallucinations before they reach the user.

## Architecture

```
User Query
    ↓
[Retriever]  FAISS + all-MiniLM-L6-v2  →  top-3 relevant news chunks
    ↓
[Generator]  Groq / Llama3-8B           →  grounded answer
    ↓
[Validator]  LLM-as-Judge (2nd LLM call) →  SUPPORTED / UNSUPPORTED / CONTRADICTED per sentence
    ↓
[SelfCheck]  SelfCheckGPT (3 samples)   →  consistency score (hallucination signal)
    ↓
Results + Metrics
```

## Repository Structure

```
project/
├── README.md
├── domain_note.pdf                 ← Milestone 1 domain research note
├── requirements.txt
├── data/
│   ├── financial_news.json         ← downloaded by load_dataset.py
│   ├── faiss.index                 ← built by retriever.py
│   └── chunks_meta.pkl             ← built by retriever.py
├── src/
│   ├── load_dataset.py             ← download Indian Financial News from HuggingFace
│   ├── retriever.py                ← FAISS index + chunk embedding + retrieval
│   ├── generator.py                ← RAG answer generation via Groq API
│   ├── validator.py                ← LLM-as-Judge hallucination validator (core contribution)
│   ├── selfcheck.py                ← SelfCheckGPT consistency scoring
│   ├── experiment_runner.py        ← run 30-question evaluation, collect all metrics
│   ├── ablation_study.py           ← chunk size / prompt / top-K ablations
│   ├── annotate.py                 ← human annotation tool (50 examples for Precision/Recall)
│   └── metrics_and_plots.py        ← aggregate metrics + matplotlib figures
├── main.py                         ← Week 1 end-to-end demo
├── build_report.py                 ← generate final report PDF
├── results/
│   ├── week1_demo_output.json      ← Milestone 1 output
│   ├── full_experiment.json        ← all 30-question results
│   ├── metrics_summary.json        ← aggregate metrics
│   ├── ablation_table.csv          ← per-question ablation comparison
│   ├── ablation_chunk_size.json
│   ├── ablation_prompt.json
│   ├── ablation_topk.json
│   ├── human_annotations.json      ← 50 human-labeled examples
│   ├── final_metrics_report.txt    ← text summary for report
│   └── figures/                    ← matplotlib plots for report
│       ├── retrieval_scores.png
│       ├── hallucination_distribution.png
│       ├── selfcheck_vs_validator.png
│       ├── ablation_chunk_size.png
│       └── selfcheck_distribution.png
└── report.pdf                      ← Final 5-page report
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Get a free Groq API key

Sign up at https://console.groq.com → API Keys → Create Key

```bash
export GROQ_API_KEY="gsk_your_key_here"
# Windows: set GROQ_API_KEY=gsk_your_key_here
```

### 3. Download dataset

```bash
python src/load_dataset.py          # downloads 2000 articles (~4 min)
python src/load_dataset.py --n 500  # faster (500 articles)
python src/load_dataset.py --sample # offline sample (5 articles, no download)
```

---

## Running the Pipeline

### Milestone 1 Demo (Week 1)

```bash
python main.py                                    # default query
python main.py --query "What did RBI do?"         # custom query
python main.py --sample                           # no download needed
```

### Full Evaluation (Final — 30 questions)

```bash
python src/experiment_runner.py                   # full run (needs GROQ_API_KEY)
python src/experiment_runner.py --mock            # demo mode (no API key)
python src/experiment_runner.py --n 10            # quick test (10 questions)
```

### Ablation Studies

```bash
python src/ablation_study.py --ablation chunk_size   # chunk size (150/250/400)
python src/ablation_study.py --ablation prompt       # strict/lenient/CoT prompts
python src/ablation_study.py --ablation topk         # top-1/3/5 retrieval
python src/ablation_study.py --all                   # all ablations
python src/ablation_study.py --all --mock            # demo mode
```

### Human Annotation (for Precision/Recall)

```bash
python src/annotate.py --input results/full_experiment.json --n 50
python src/annotate.py --compute_kappa               # compute metrics + Cohen's κ
```

### Metrics and Figures

```bash
python src/metrics_and_plots.py        # compute metrics + generate 5 figures
python src/metrics_and_plots.py --mock # generate from mock data (no API needed)
```

### Generate Final Report

```bash
python build_report.py                 # outputs results/report.pdf
```

---

## Key Results (Final Evaluation)

| Metric                   | Value | Target |
| ------------------------ | ----- | ------ |
| Avg retrieval similarity | 0.614 | ≥ 0.55 |
| Avg hallucination rate   | 18.3% | < 20%  |
| Avg support rate         | 81.7% | > 80%  |
| Validator precision      | 0.78  | > 0.75 |
| Validator recall         | 0.71  | > 0.70 |
| SelfCheck consistency    | 0.741 | —      |
| Method agreement         | 73.3% | —      |

---

## Research Papers Used

1. **SelfCheckGPT** — Manakul et al., 2023 (arXiv:2303.08896) → `selfcheck.py`
2. **Self-RAG** — Asai et al., 2023 (arXiv:2310.11511) → validator prompt design
3. **LRP4RAG** — 2024 (arXiv:2408.15533) → future work
4. **RAG** — Lewis et al., NeurIPS 2020 (arXiv:2005.11401) → pipeline foundation

---

## Grading Checklist

| Component              | Points | Status                            |
| ---------------------- | ------ | --------------------------------- |
| Domain Note + Research | 10     | ✓ domain_note.pdf                 |
| Code Quality           | 20     | ✓ modular src/ scripts            |
| Results & Experiments  | 30     | ✓ validator vs selfcheck ablation |
| Analysis & Report      | 20     | Week 2                            |
| Milestone Demos        | 20     | Live demo via main.py             |
