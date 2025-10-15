#!/usr/bin/env python3
"""
Script to merge top beam predictions from multiple CSV files using Anthropic API.
"""

import csv
import os
import glob
from pathlib import Path
from anthropic import Anthropic
from datetime import datetime


def read_predictions(predictions_dir):
    """
    Read all prediction CSV files and organize them by sentence index.

    Args:
        predictions_dir: Path to directory containing prediction CSV files

    Returns:
        List of dictionaries, each containing true_sentence and list of top_beam_predictions
    """
    # Find all CSV files
    csv_files = sorted(glob.glob(os.path.join(predictions_dir, "test_predictions_*.csv")))

    if not csv_files:
        raise ValueError(f"No prediction CSV files found in {predictions_dir}")

    print(f"Found {len(csv_files)} prediction files:")
    for f in csv_files:
        print(f"  - {os.path.basename(f)}")

    # Read all files and organize by index
    sentences_data = []

    # Read first file to initialize the structure
    with open(csv_files[0], 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sentences_data.append({
                'true_sentence': row['true_sentence'],
                'top_beam_predictions': [row['top_beam_prediction']],
                'greedy_predictions': [row['greedy_prediction']]
            })

    # Read remaining files and append predictions
    for csv_file in csv_files[1:]:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                if idx >= len(sentences_data):
                    print(f"Warning: {csv_file} has more rows than first file")
                    break
                sentences_data[idx]['top_beam_predictions'].append(row['top_beam_prediction'])
                sentences_data[idx]['greedy_predictions'].append(row['greedy_prediction'])

    print(f"\nCollected predictions for {len(sentences_data)} sentences")
    print(f"Each sentence has {len(csv_files)} predictions to merge")

    return sentences_data


def merge_predictions_with_claude(predictions_list, greedy_list, api_key):
    """
    Use Anthropic API to merge multiple predictions into one sentence.

    Args:
        predictions_list: List of beam prediction strings to merge
        greedy_list: List of corresponding greedy prediction strings
        api_key: Anthropic API key

    Returns:
        Merged sentence string
    """
    client = Anthropic(api_key=api_key)

    # Create prompt for Claude with beam and greedy predictions paired
    predictions_text = "\n".join([
        f"Candidate {i+1}:\n  Beam: {beam}\n  Greedy: {greedy}"
        for i, (beam, greedy) in enumerate(zip(predictions_list, greedy_list))
    ])

#     prompt = f"""You are given multiple predictions of the same sentence from different speech recognition models decoding very noisy audio. Your task is to merge these predictions into a single, coherent sentence that best represents the intended meaning. In effect, you are resolving the epistemic uncertainty across all these speech recognition models by observing the individual sentence predictions. Note that all predictions have the correct sentence length, so make sure your sentence matches that length.

# Here are the predictions:
# {predictions_text}

# Please provide the best merged sentence. Output only the merged sentence, nothing else."""
    
    prompt = f"""
Your task is to perform automatic speech recognition. You are given some candidates of an
unknown transcription generated from different speech recognition models via beam search. Your job is to come up with a transcription that is most accurate, relying
on the context that the candidates provide. You will additionally be given some the corresponding greedy most likely words from each model. Both the beam search transcription candidates and greedy word predictions may or may not contain errors. The greedy word predictions are additional helpful information that may help you come up with
the correct transcription. Make sure your transcription based on the query beam search candidates is
contextually and grammatically correct. Focus on key differences in the candidates that change
the meaning or correctness. Avoid selections with repetitive or nonsensical phrases. In cases of
ambiguity, select the option that is most coherent and contextually sound. Note that all the candidate sentences have the correct number of words, so your final transcription should use the same number of words. Respond with your
final transcription only, without any introductory text.

Here are the predictions:
{predictions_text}

Output only your final merged transcription and nothing else.
    """.strip()

    message = client.messages.create(
        # model="claude-3-5-sonnet-20241022",
        # model="claude-sonnet-4-5-20250929",
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return message.content[0].text.strip()


def main():
    """Main function to process predictions and merge them."""
    # Configuration
    predictions_dir = "predictions"  # Change this to your predictions directory path
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    # Read all predictions
    print("Reading prediction files...")
    sentences_data = read_predictions(predictions_dir)

    # Process each sentence
    print("\nMerging predictions using Claude API...")
    results = []

    for idx, data in enumerate(sentences_data):
        print(f"Processing sentence {idx+1}/{len(sentences_data)}...", end='\r')

        merged_sentence = merge_predictions_with_claude(
            data['top_beam_predictions'],
            data['greedy_predictions'],
            api_key
        )

        results.append({
            'true_sentence': data['true_sentence'],
            'merged_prediction': merged_sentence,
            'individual_predictions': data['top_beam_predictions'],
            'greedy_predictions': data['greedy_predictions']
        })

    print(f"\nCompleted merging {len(results)} sentences")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"merged_predictions_{timestamp}.csv"

    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['true_sentence', 'merged_prediction', 'individual_predictions'])

        for result in results:
            writer.writerow([
                result['true_sentence'],
                result['merged_prediction'],
                '|'.join(result['individual_predictions'])
            ])

    print(f"\nResults saved to: {output_file}")

    # Print sample results
    print("\nSample results (first 3):")
    for i, result in enumerate(results[:3]):
        print(f"\n--- Sentence {i+1} ---")
        print(f"True: {result['true_sentence']}")
        print(f"Merged: {result['merged_prediction']}")
        print(f"Individual predictions:")
        for j, (beam, greedy) in enumerate(zip(result['individual_predictions'], result['greedy_predictions'])):
            print(f"  Candidate {j+1}:")
            print(f"    Beam: {beam}")
            print(f"    Greedy: {greedy}")


if __name__ == "__main__":
    main()
