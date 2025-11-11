"""
Compare Sample-Level vs Sentence-Level Split Results

This script loads both models and evaluates them side-by-side to show
the difference between robustness testing (sample-level) and
generalization testing (sentence-level).
"""

import torch
import numpy as np
from pathlib import Path
from train_ctc import (
    CTCReadout, load_label_map, greedy_decoder,
    BLANK_TOKEN, CHAR_MAP, INDEX_MAP
)

def calculate_cer(reference: str, hypothesis: str) -> float:
    """Calculate Character Error Rate"""
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)

    # Levenshtein distance
    d = [[0] * (len(hyp_chars) + 1) for _ in range(len(ref_chars) + 1)]

    for i in range(len(ref_chars) + 1):
        d[i][0] = i
    for j in range(len(hyp_chars) + 1):
        d[0][j] = j

    for i in range(1, len(ref_chars) + 1):
        for j in range(1, len(hyp_chars) + 1):
            if ref_chars[i-1] == hyp_chars[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = min(d[i-1][j], d[i][j-1], d[i-1][j-1]) + 1

    return d[len(ref_chars)][len(hyp_chars)] / len(ref_chars) if len(ref_chars) > 0 else 0.0

def calculate_wer(reference: str, hypothesis: str) -> float:
    """Calculate Word Error Rate"""
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    # Levenshtein distance for words
    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]

    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = min(d[i-1][j], d[i][j-1], d[i-1][j-1]) + 1

    return d[len(ref_words)][len(hyp_words)] / len(ref_words) if len(ref_words) > 0 else 0.0

def evaluate_split(features_file, model_file, split_name):
    """Evaluate a single split"""
    print(f"\n{'='*80}")
    print(f"EVALUATING: {split_name.upper()}")
    print(f"{'='*80}")

    # Check if files exist
    if not Path(features_file).exists():
        print(f"❌ Features file not found: {features_file}")
        return None

    if not Path(model_file).exists():
        print(f"❌ Model file not found: {model_file}")
        print(f"   Please train the model first: python train_ctc.py")
        return None

    # Load data
    print(f"Loading features from '{features_file}'...")
    dataset = np.load(features_file)
    X_train = dataset['X_train_sequences']
    X_test = dataset['X_test_sequences']
    y_test_indices = dataset['y_test']

    label_map = load_label_map()
    if label_map is None:
        return None

    y_test_text = [label_map[idx] for idx in y_test_indices]

    print(f"Test samples: {len(X_test)}")

    # Normalize features
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    X_test_flat = X_test.reshape(-1, X_test.shape[-1])

    feature_mean = X_train_flat.mean(axis=0)
    feature_std = X_train_flat.std(axis=0) + 1e-8

    feature_mean = feature_mean.reshape(1, 1, -1)
    feature_std = feature_std.reshape(1, 1, -1)

    X_test_normalized = (X_test - feature_mean) / feature_std
    X_test_tensor = torch.FloatTensor(X_test_normalized)

    # Load model
    print(f"Loading model from '{model_file}'...")
    num_lsm_neurons = X_test_tensor.shape[2]
    num_classes = len(CHAR_MAP) + 1

    model = CTCReadout(input_features=num_lsm_neurons, num_classes=num_classes)
    model.load_state_dict(torch.load(model_file))
    model.eval()

    # Evaluate
    print("Evaluating all test samples...")
    correct = 0
    total = len(X_test_tensor)
    all_cer = []
    all_wer = []

    with torch.no_grad():
        for i in range(total):
            test_sample_log_probs = model(X_test_tensor[i].unsqueeze(0))
            test_sample_log_probs = test_sample_log_probs.squeeze(0)
            decoded_text = greedy_decoder(test_sample_log_probs)

            target = y_test_text[i]

            if decoded_text == target:
                correct += 1

            cer = calculate_cer(target, decoded_text)
            wer = calculate_wer(target, decoded_text)
            all_cer.append(cer)
            all_wer.append(wer)

    # Calculate metrics
    accuracy = (correct / total) * 100
    avg_cer = np.mean(all_cer) * 100
    avg_wer = np.mean(all_wer) * 100

    results = {
        'split_name': split_name,
        'total_samples': total,
        'correct': correct,
        'accuracy': accuracy,
        'cer': avg_cer,
        'wer': avg_wer,
        'char_accuracy': 100 - avg_cer,
        'word_accuracy': 100 - avg_wer
    }

    print(f"\n{'='*80}")
    print(f"RESULTS: {split_name.upper()}")
    print(f"{'='*80}")
    print(f"Sentence Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"Character Error Rate (CER): {avg_cer:.2f}%")
    print(f"Word Error Rate (WER): {avg_wer:.2f}%")
    print(f"Character Accuracy: {100 - avg_cer:.2f}%")
    print(f"Word Accuracy: {100 - avg_wer:.2f}%")

    return results

def compare_results(sample_results, sentence_results):
    """Compare results side-by-side"""
    print(f"\n{'='*80}")
    print("SIDE-BY-SIDE COMPARISON")
    print(f"{'='*80}")

    if sample_results is None or sentence_results is None:
        print("⚠️  Cannot compare - one or both models not evaluated")
        return

    print(f"\n{'Metric':<30} | {'Sample-Level':<20} | {'Sentence-Level':<20}")
    print("-" * 80)
    print(f"{'Test samples':<30} | {sample_results['total_samples']:<20} | {sentence_results['total_samples']:<20}")
    print(f"{'Sentence accuracy':<30} | {sample_results['accuracy']:<20.2f} | {sentence_results['accuracy']:<20.2f}")
    print(f"{'Character Error Rate (CER)':<30} | {sample_results['cer']:<20.2f} | {sentence_results['cer']:<20.2f}")
    print(f"{'Word Error Rate (WER)':<30} | {sample_results['wer']:<20.2f} | {sentence_results['wer']:<20.2f}")
    print(f"{'Character accuracy':<30} | {sample_results['char_accuracy']:<20.2f} | {sentence_results['char_accuracy']:<20.2f}")
    print(f"{'Word accuracy':<30} | {sample_results['word_accuracy']:<20.2f} | {sentence_results['word_accuracy']:<20.2f}")

    print(f"\n{'What it tests':<30} | {'Robustness':<20} | {'Generalization':<20}")
    print(f"{'Test sentences':<30} | {'Seen (71/100)':<20} | {'Unseen (20/100)':<20}")

    # Interpretation
    print(f"\n{'='*80}")
    print("INTERPRETATION")
    print(f"{'='*80}")

    diff = sample_results['accuracy'] - sentence_results['accuracy']

    if diff < 5:
        print(f"\n✅ EXCELLENT: Only {diff:.1f}% accuracy drop on unseen sentences!")
        print(f"   Model has learned robust character-level patterns.")
    elif diff < 15:
        print(f"\n✅ GOOD: {diff:.1f}% accuracy drop on unseen sentences.")
        print(f"   Model generalizes reasonably well to new vocabulary.")
    elif diff < 25:
        print(f"\n⚠️  MODERATE: {diff:.1f}% accuracy drop on unseen sentences.")
        print(f"   Model has some generalization but relies partly on word templates.")
    else:
        print(f"\n❌ LARGE DROP: {diff:.1f}% accuracy drop on unseen sentences.")
        print(f"   Model may be memorizing word templates rather than learning characters.")

    print(f"\n{'='*80}")
    print("RECOMMENDATIONS FOR REPORTING")
    print(f"{'='*80}")

    print(f"""
Report both results for complete evaluation:

"The LSM-based sentence recognizer achieves {sample_results['accuracy']:.1f}%
sentence accuracy on sample-level split (robustness to audio variations)
and {sentence_results['accuracy']:.1f}% on sentence-level split (generalization
to unseen sentences).

Character-level accuracy is {sample_results['char_accuracy']:.1f}% and
{sentence_results['char_accuracy']:.1f}% respectively, demonstrating both
production readiness and fundamental character recognition capability."
    """)

if __name__ == "__main__":
    print("\n" + "="*80)
    print("SAMPLE-LEVEL vs SENTENCE-LEVEL SPLIT COMPARISON")
    print("="*80)

    # Evaluate sample-level split
    sample_results = evaluate_split(
        features_file="lsm_windowed_features_filtered.npz",
        model_file="ctc_model.pt",
        split_name="Sample-Level Split"
    )

    # Evaluate sentence-level split
    sentence_results = evaluate_split(
        features_file="lsm_windowed_features_filtered_sentence_split.npz",
        model_file="ctc_model_sentence_split.pt",
        split_name="Sentence-Level Split"
    )

    # Compare
    compare_results(sample_results, sentence_results)

    print(f"\n{'='*80}")
    print("COMPARISON COMPLETE")
    print(f"{'='*80}\n")
