#!/usr/bin/env python3
"""
Script to evaluate merged predictions using WER, CER, BLEU, ROUGE, and BERTScore.
Uses the same metrics computation as in model/word_classifier.py
"""

import csv
import argparse
import jiwer
import bert_score
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import numpy as np
from pathlib import Path


def compute_sentence_metrics(true_sent, pred_sent):
    """
    Compute all metrics for a single sentence pair.
    Same implementation as WordClassifier._compute_sentence_metrics()

    Args:
        true_sent: Ground truth sentence
        pred_sent: Predicted sentence

    Returns:
        Dictionary with CER, WER, BLEU, ROUGE, and BERTScore
    """
    cer = jiwer.cer(true_sent, pred_sent)
    wer = jiwer.wer(true_sent, pred_sent)

    ref_tokens = true_sent.split()
    hyp_tokens = pred_sent.split()

    bleu = sentence_bleu([ref_tokens], hyp_tokens, weights=[1])  # BLEU-1 only

    rouge = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    rouge_score = rouge.score(true_sent, pred_sent)

    bert = bert_score.score([pred_sent], [true_sent], lang='en', verbose=False)

    return {
        "cer": cer,
        "wer": wer,
        "bleu": bleu,
        "rouge": rouge_score['rouge1'].fmeasure,
        "bert": bert[2].item(),
    }


def evaluate_predictions(csv_file, prediction_column="merged_prediction"):
    """
    Evaluate predictions from a CSV file.

    Args:
        csv_file: Path to CSV file with columns: true_sentence, merged_prediction
        prediction_column: Name of the column containing predictions to evaluate

    Returns:
        Dictionary with mean and std of all metrics
    """
    print(f"Reading predictions from: {csv_file}")
    print(f"Evaluating column: {prediction_column}\n")

    # Read CSV file
    results = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        # Verify required columns exist
        if 'true_sentence' not in reader.fieldnames:
            raise ValueError("CSV must contain 'true_sentence' column")
        if prediction_column not in reader.fieldnames:
            raise ValueError(f"CSV must contain '{prediction_column}' column")

        for row in reader:
            true_sent = row['true_sentence']
            pred_sent = row[prediction_column]

            # Skip empty predictions
            if not pred_sent or not pred_sent.strip():
                print(f"Warning: Skipping empty prediction for: {true_sent[:50]}...")
                continue

            metrics = compute_sentence_metrics(true_sent, pred_sent)
            results.append(metrics)

    if not results:
        raise ValueError("No valid predictions found in CSV")

    print(f"Evaluated {len(results)} predictions\n")

    # Compute mean and std for each metric
    metric_names = ["cer", "wer", "bleu", "rouge", "bert"]
    summary = {}

    for metric in metric_names:
        values = [r[metric] for r in results]
        summary[f"{metric}_mean"] = np.mean(values)
        summary[f"{metric}_std"] = np.std(values)
        summary[f"{metric}_values"] = values

    return summary, results


def print_results(summary, prediction_type="Merged"):
    """Print evaluation results in a formatted table."""
    print("=" * 60)
    print(f"{prediction_type} Prediction Evaluation Results")
    print("=" * 60)
    print(f"{'Metric':<15} {'Mean':<12} {'Std':<12}")
    print("-" * 60)

    metrics = [
        ("CER", "cer"),
        ("WER", "wer"),
        ("BLEU-1", "bleu"),
        ("ROUGE-1", "rouge"),
        ("BERTScore", "bert"),
    ]

    for label, metric in metrics:
        mean = summary[f"{metric}_mean"]
        std = summary[f"{metric}_std"]
        print(f"{label:<15} {mean:<12.4f} {std:<12.4f}")

    print("=" * 60)


def compare_predictions(csv_file):
    """
    Compare merged predictions against greedy and top beam predictions if available.

    Args:
        csv_file: Path to CSV file
    """
    # Check what columns are available
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames

    # Evaluate merged predictions
    print("\n" + "=" * 60)
    print("EVALUATING MERGED PREDICTIONS")
    print("=" * 60 + "\n")
    merged_summary, merged_results = evaluate_predictions(csv_file, "merged_prediction")
    print_results(merged_summary, "Merged")

    # Evaluate greedy predictions if available
    if "greedy_prediction" in columns:
        print("\n" + "=" * 60)
        print("EVALUATING GREEDY PREDICTIONS (for comparison)")
        print("=" * 60 + "\n")
        greedy_summary, greedy_results = evaluate_predictions(csv_file, "greedy_prediction")
        print_results(greedy_summary, "Greedy")

        # Print comparison
        print("\n" + "=" * 60)
        print("IMPROVEMENT: Merged vs Greedy")
        print("=" * 60)
        print(f"{'Metric':<15} {'Improvement':<15} {'Direction'}")
        print("-" * 60)

        for metric in ["cer", "wer", "bleu", "rouge", "bert"]:
            merged_val = merged_summary[f"{metric}_mean"]
            greedy_val = greedy_summary[f"{metric}_mean"]

            # For CER and WER, lower is better
            # For BLEU, ROUGE, BERT, higher is better
            if metric in ["cer", "wer"]:
                improvement = greedy_val - merged_val
                direction = "↓ (better)" if improvement > 0 else "↑ (worse)"
            else:
                improvement = merged_val - greedy_val
                direction = "↑ (better)" if improvement > 0 else "↓ (worse)"

            print(f"{metric.upper():<15} {improvement:+.4f}{'':>9} {direction}")
        print("=" * 60)

    # Evaluate top beam prediction if available
    if "top_beam_prediction" in columns:
        print("\n" + "=" * 60)
        print("EVALUATING TOP BEAM PREDICTIONS (for comparison)")
        print("=" * 60 + "\n")
        beam_summary, beam_results = evaluate_predictions(csv_file, "top_beam_prediction")
        print_results(beam_summary, "Top Beam")

        # Print comparison
        print("\n" + "=" * 60)
        print("IMPROVEMENT: Merged vs Top Beam")
        print("=" * 60)
        print(f"{'Metric':<15} {'Improvement':<15} {'Direction'}")
        print("-" * 60)

        for metric in ["cer", "wer", "bleu", "rouge", "bert"]:
            merged_val = merged_summary[f"{metric}_mean"]
            beam_val = beam_summary[f"{metric}_mean"]

            if metric in ["cer", "wer"]:
                improvement = beam_val - merged_val
                direction = "↓ (better)" if improvement > 0 else "↑ (worse)"
            else:
                improvement = merged_val - beam_val
                direction = "↑ (better)" if improvement > 0 else "↓ (worse)"

            print(f"{metric.upper():<15} {improvement:+.4f}{'':>9} {direction}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate merged predictions using speech recognition metrics"
    )
    parser.add_argument(
        "csv_file",
        type=str,
        help="Path to CSV file with predictions (e.g., merged_predictions_TIMESTAMP.csv)"
    )
    parser.add_argument(
        "--no-compare",
        action="store_true",
        help="Only evaluate merged predictions, skip comparison with greedy/beam"
    )

    args = parser.parse_args()

    # Check if file exists
    if not Path(args.csv_file).exists():
        print(f"Error: File not found: {args.csv_file}")
        return

    if args.no_compare:
        # Only evaluate merged predictions
        summary, results = evaluate_predictions(args.csv_file, "merged_prediction")
        print_results(summary, "Merged")
    else:
        # Evaluate and compare all available prediction types
        compare_predictions(args.csv_file)


if __name__ == "__main__":
    main()
