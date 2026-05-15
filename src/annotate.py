"""
annotate.py  ─  Human Annotation Tool for Ground-Truth Labels
=============================================================

Creates 50 human-annotated QA examples to compute:
  - Validator Precision (of flagged hallucinations, how many are truly wrong?)
  - Validator Recall (of true hallucinations, how many did validator catch?)
  - Inter-rater agreement (Cohen's κ) if two annotators label the same set

USAGE (for a single annotator):
    python src/annotate.py --input results/full_experiment.json
    python src/annotate.py --input results/full_experiment.json --n 20

  Walks you through each (question, answer, source, sentence) one by one.
  You label each sentence: s (supported) / u (unsupported) / c (contradicted) / skip

  Saves to: results/human_annotations.json

USAGE (for inter-rater agreement — two people annotate same set):
    Annotator 1: python src/annotate.py --input results/full_experiment.json --annotator A
    Annotator 2: python src/annotate.py --input results/full_experiment.json --annotator B
    Compute:     python src/annotate.py --compute_kappa

WHY THIS MATTERS FOR YOUR REPORT:
  Without human labels, you can't compute Precision/Recall of the validator.
  Your domain note promises 50 annotated examples — this tool produces them.
  Cohen's κ > 0.6 = substantial agreement between annotators.
"""

import os
import sys
import json
import argparse
from typing import List, Dict


LABEL_MAP = {"s": "SUPPORTED", "u": "UNSUPPORTED", "c": "CONTRADICTED"}


def load_experiment_results(path: str) -> List[Dict]:
    """Load full_experiment.json produced by experiment_runner.py"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def annotate_interactive(results: List[Dict], n: int, annotator: str, output_path: str):
    """
    Interactive CLI annotation loop.
    For each result → for each sentence → human labels it.
    """
    print("\n" + "="*65)
    print("  HUMAN ANNOTATION TOOL — Financial Hallucination Detection")
    print("="*65)
    print("\nLabels: s = SUPPORTED  |  u = UNSUPPORTED  |  c = CONTRADICTED  |  skip = skip sentence")
    print("        q = quit and save what you have so far\n")

    annotations = []
    annotated_count = 0

    for result in results:
        if annotated_count >= n:
            break

        question    = result.get("question", "?")
        answer      = result.get("answer", "?")
        sources     = result.get("retrieved_sources", [])
        val_summary = result.get("validation_summary", {})

        # Skip mock results — they have no real sentences to label
        if result.get("mock", False) or val_summary.get("mock", False):
            print(f"  [Skipping mock result for: {question[:50]}]")
            continue

        print(f"\n{'─'*65}")
        print(f"QUESTION:  {question}")
        print(f"SOURCES:   {', '.join(sources[:2])}")
        print(f"\nANSWER:")
        print(f"  {answer}")
        print(f"{'─'*65}")

        # Split answer into sentences
        sentences = [s.strip() for s in answer.replace(".\n", ". ").split(". ") if len(s.strip()) > 15]
        if not sentences:
            continue

        sentence_labels = []
        skipped_all = True

        for i, sentence in enumerate(sentences):
            print(f"\n  Sentence {i+1}/{len(sentences)}:")
            print(f"  \"{sentence}\"")

            while True:
                label_input = input("  Label (s/u/c/skip/q): ").strip().lower()
                if label_input == "q":
                    print("\nQuitting — saving annotations so far...")
                    save_annotations(annotations, output_path)
                    print(f"✓ Saved {len(annotations)} annotations to {output_path}")
                    return
                elif label_input == "skip":
                    sentence_labels.append({"text": sentence, "label": "SKIP", "human": annotator})
                    break
                elif label_input in LABEL_MAP:
                    sentence_labels.append({
                        "text":   sentence,
                        "label":  LABEL_MAP[label_input],
                        "human":  annotator,
                    })
                    skipped_all = False
                    break
                else:
                    print("  Invalid input. Type s, u, c, skip, or q.")

        if not skipped_all:
            # Count labels for this example
            label_counts = {}
            for sl in sentence_labels:
                lbl = sl["label"]
                label_counts[lbl] = label_counts.get(lbl, 0) + 1

            human_h_rate = 0.0
            non_skip = [sl for sl in sentence_labels if sl["label"] != "SKIP"]
            if non_skip:
                n_bad = sum(1 for sl in non_skip if sl["label"] in ("UNSUPPORTED", "CONTRADICTED"))
                human_h_rate = n_bad / len(non_skip)

            annotations.append({
                "question":           question,
                "answer":             answer,
                "retrieved_sources":  sources,
                "sentences":          sentence_labels,
                "human_halluc_rate":  round(human_h_rate, 4),
                "label_counts":       label_counts,
                "validator_halluc_rate": val_summary.get("hallucination_rate", -1),
                "validator_verdict":     val_summary.get("verdict", "?"),
                "annotator":          annotator,
            })
            annotated_count += 1
            print(f"\n  ✓ Labeled {annotated_count}/{n} examples | Your halluc rate: {human_h_rate:.1%}")

    save_annotations(annotations, output_path)
    print(f"\n{'='*65}")
    print(f"✓ ANNOTATION COMPLETE: {len(annotations)} examples saved to {output_path}")


def save_annotations(annotations: List[Dict], output_path: str):
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)


def compute_metrics(annotations_path: str):
    """
    Compute Precision, Recall, F1 of the validator vs human labels.
    Also compute Cohen's κ if two annotators are present.
    """
    with open(annotations_path, encoding="utf-8") as f:
        annotations = json.load(f)

    print(f"\nLoaded {len(annotations)} annotated examples from {annotations_path}")

    # Sentence-level comparison: validator vs human
    tp = fp = tn = fn = 0  # True/False Positives/Negatives (hallucination detection)
    all_human_labels    = []
    all_validator_labels = []

    for ann in annotations:
        sentences = ann.get("sentences", [])
        # We don't have per-sentence validator labels in the summary — skip if missing
        val_rate   = ann.get("validator_halluc_rate", -1)
        human_rate = ann.get("human_halluc_rate", -1)

        if val_rate >= 0 and human_rate >= 0:
            # Macro level: does validator agree with human on "hallucinated vs. clean"?
            val_halluc   = val_rate > 0.25
            human_halluc = human_rate > 0.25
            if human_halluc and val_halluc:     tp += 1
            elif human_halluc and not val_halluc: fn += 1
            elif not human_halluc and val_halluc: fp += 1
            else:                                 tn += 1

        for s in sentences:
            human_lbl = s.get("label")
            if human_lbl and human_lbl != "SKIP":
                all_human_labels.append(1 if human_lbl in ("UNSUPPORTED", "CONTRADICTED") else 0)

    n     = tp + fp + tn + fn
    prec  = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec   = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1    = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    acc   = (tp + tn) / n if n > 0 else 0

    human_halluc_rates = [a["human_halluc_rate"] for a in annotations if a.get("human_halluc_rate", -1) >= 0]
    avg_human_h        = sum(human_halluc_rates) / len(human_halluc_rates) if human_halluc_rates else 0

    results = {
        "n_examples":           len(annotations),
        "n_sentences_labeled":  len(all_human_labels),
        "avg_human_halluc_rate": round(avg_human_h, 4),
        "validator_vs_human": {
            "true_positives":  tp,
            "false_positives": fp,
            "true_negatives":  tn,
            "false_negatives": fn,
            "precision":       round(prec, 4),
            "recall":          round(rec, 4),
            "f1":              round(f1,   4),
            "accuracy":        round(acc,  4),
        },
        "interpretation": {
            "precision": f"Of {tp+fp} examples validator flagged as hallucinated, {tp} ({prec:.1%}) were truly hallucinated.",
            "recall":    f"Of {tp+fn} truly hallucinated examples, validator caught {tp} ({rec:.1%}).",
            "f1":        f"F1={f1:.3f} — harmonic mean of precision and recall.",
        }
    }

    # Cohen's κ if two annotators
    annotators = list(set(a.get("annotator", "A") for a in annotations))
    if len(annotators) >= 2:
        kappa = compute_cohens_kappa(annotations)
        results["cohens_kappa"] = kappa
        print(f"\n  Cohen's κ (inter-rater agreement): {kappa:.3f}")
        if kappa > 0.8:   print("  → Almost perfect agreement")
        elif kappa > 0.6: print("  → Substantial agreement")
        elif kappa > 0.4: print("  → Moderate agreement")
        else:             print("  → Fair/poor agreement — review disagreements")

    # Print summary
    print(f"\n{'='*55}")
    print("VALIDATOR EVALUATION METRICS (vs Human Labels)")
    print(f"{'='*55}")
    print(f"  Examples: {len(annotations)}  |  Sentences labeled: {len(all_human_labels)}")
    print(f"  Avg human hallucination rate: {avg_human_h:.1%}")
    print(f"\n  Confusion Matrix:")
    print(f"    True  Positives (halluc, both agree):   {tp}")
    print(f"    False Positives (validator wrong flag):  {fp}")
    print(f"    True  Negatives (clean, both agree):    {tn}")
    print(f"    False Negatives (validator missed):     {fn}")
    print(f"\n  Precision: {prec:.4f}  ({prec:.1%})")
    print(f"  Recall:    {rec:.4f}   ({rec:.1%})")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  Accuracy:  {acc:.4f}")

    out_path = annotations_path.replace(".json", "_metrics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Metrics saved to {out_path}")
    return results


def compute_cohens_kappa(annotations: List[Dict]) -> float:
    """Simple Cohen's κ at example level (hallucinated vs. clean)."""
    annotators = {}
    for ann in annotations:
        a_name = ann.get("annotator", "A")
        q      = ann.get("question", "?")
        label  = 1 if ann.get("human_halluc_rate", 0) > 0.25 else 0
        if q not in annotators:
            annotators[q] = {}
        annotators[q][a_name] = label

    # Only include questions labeled by exactly 2 annotators
    pairs = [(v[k] for k in sorted(v)) for q, v in annotators.items() if len(v) == 2]
    if not pairs:
        return 0.0

    labels = list(pairs)
    n = len(labels)
    a_labels = [p[0] for p in labels]
    b_labels = [p[1] for p in labels]

    po = sum(1 for a, b in zip(a_labels, b_labels) if a == b) / n
    pa = (sum(a_labels)/n) * (sum(b_labels)/n) + ((n-sum(a_labels))/n) * ((n-sum(b_labels))/n)
    kappa = (po - pa) / (1 - pa) if (1 - pa) > 0 else 0.0
    return round(kappa, 4)


def main():
    parser = argparse.ArgumentParser(description="Human annotation tool")
    parser.add_argument("--input",          type=str, default="results/full_experiment.json")
    parser.add_argument("--output",         type=str, default="results/human_annotations.json")
    parser.add_argument("--n",              type=int, default=50, help="Number of examples to annotate")
    parser.add_argument("--annotator",      type=str, default="A",  help="Annotator ID (A, B, etc.)")
    parser.add_argument("--compute_kappa",  action="store_true",    help="Compute metrics from saved annotations")
    args = parser.parse_args()

    if args.compute_kappa:
        compute_metrics(args.output)
    else:
        if not os.path.exists(args.input):
            print(f"Error: {args.input} not found. Run experiment_runner.py first.")
            sys.exit(1)
        results = load_experiment_results(args.input)
        annotate_interactive(results, args.n, args.annotator, args.output)


if __name__ == "__main__":
    main()
