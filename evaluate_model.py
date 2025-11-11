import torch
import numpy as np
from pathlib import Path
import itertools
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

def evaluate_model():
    print("="*80)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*80)

    # Load data
    print("\nLoading data...")
    filtered_file = "lsm_windowed_features_filtered.npz"
    feature_file = "lsm_windowed_features.npz"

    if Path(filtered_file).exists():
        print(f"✅ Loading from '{filtered_file}'")
        dataset = np.load(filtered_file)
    elif Path(feature_file).exists():
        print(f"✅ Loading from '{feature_file}'")
        dataset = np.load(feature_file)
    else:
        print("❌ Error: No feature file found.")
        return

    X_train = dataset['X_train_sequences']
    y_train_indices = dataset['y_train']
    X_test = dataset['X_test_sequences']
    y_test_indices = dataset['y_test']

    label_map = load_label_map()
    if label_map is None:
        return

    y_test_text = [label_map[idx] for idx in y_test_indices]

    print(f"Test samples: {len(X_test)}")
    print(f"Unique test sentences: {len(set(y_test_text))}")

    # Normalize features (FIXED VERSION)
    print("\nNormalizing features (CORRECTED)...")
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    X_test_flat = X_test.reshape(-1, X_test.shape[-1])

    feature_mean = X_train_flat.mean(axis=0)
    feature_std = X_train_flat.std(axis=0) + 1e-8

    # FIX: Properly reshape for broadcasting
    feature_mean = feature_mean.reshape(1, 1, -1)
    feature_std = feature_std.reshape(1, 1, -1)

    X_train_normalized = (X_train - feature_mean) / feature_std
    X_test_normalized = (X_test - feature_mean) / feature_std

    print(f"  Original range: [{X_train.min():.2f}, {X_train.max():.2f}]")
    print(f"  Normalized range: [{X_train_normalized.min():.2f}, {X_train_normalized.max():.2f}]")

    X_test_tensor = torch.FloatTensor(X_test_normalized)

    # Load model
    print("\nLoading trained model...")
    num_lsm_neurons = X_test_tensor.shape[2]
    num_classes = len(CHAR_MAP) + 1

    model = CTCReadout(input_features=num_lsm_neurons, num_classes=num_classes)

    # Check if model checkpoint exists
    if Path("ctc_model.pt").exists():
        model.load_state_dict(torch.load("ctc_model.pt"))
        print("✅ Loaded saved model from 'ctc_model.pt'")
    else:
        print("⚠️  No saved model found. Training model first...")
        print("   Please run: python train_ctc.py")
        print("   Then run this script again.")
        return

    model.eval()

    # Evaluate all test samples
    print("\n" + "="*80)
    print("EVALUATING ALL TEST SAMPLES")
    print("="*80)

    all_cer = []
    all_wer = []
    perfect_matches = 0
    all_predictions = []

    with torch.no_grad():
        for i in range(len(X_test_tensor)):
            # Get prediction
            test_sample_log_probs = model(X_test_tensor[i].unsqueeze(0))
            test_sample_log_probs = test_sample_log_probs.squeeze(0)
            decoded_text = greedy_decoder(test_sample_log_probs)

            target = y_test_text[i]

            # Calculate metrics
            cer = calculate_cer(target, decoded_text)
            wer = calculate_wer(target, decoded_text)

            all_cer.append(cer)
            all_wer.append(wer)

            if decoded_text == target:
                perfect_matches += 1

            all_predictions.append({
                'index': i,
                'target': target,
                'prediction': decoded_text,
                'cer': cer,
                'wer': wer,
                'perfect': decoded_text == target
            })

    # Summary statistics
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)

    avg_cer = np.mean(all_cer) * 100
    avg_wer = np.mean(all_wer) * 100
    sentence_accuracy = (perfect_matches / len(X_test_tensor)) * 100

    print(f"\n📊 Overall Metrics:")
    print(f"  Character Error Rate (CER): {avg_cer:.2f}%")
    print(f"  Word Error Rate (WER): {avg_wer:.2f}%")
    print(f"  Sentence Accuracy: {sentence_accuracy:.2f}% ({perfect_matches}/{len(X_test_tensor)})")
    print(f"  Character Accuracy: {100 - avg_cer:.2f}%")
    print(f"  Word Accuracy: {100 - avg_wer:.2f}%")

    # Show best and worst predictions
    print("\n" + "="*80)
    print("BEST PREDICTIONS (Perfect or Near-Perfect)")
    print("="*80)

    best_predictions = sorted(all_predictions, key=lambda x: x['cer'])[:10]
    for pred in best_predictions:
        status = "✅ PERFECT" if pred['perfect'] else f"CER: {pred['cer']*100:.1f}%"
        print(f"\n[{pred['index']}] {status}")
        print(f"  Target:     '{pred['target']}'")
        print(f"  Prediction: '{pred['prediction']}'")
        if pred['wer'] > 0:
            print(f"  WER: {pred['wer']*100:.1f}%")

    print("\n" + "="*80)
    print("WORST PREDICTIONS")
    print("="*80)

    worst_predictions = sorted(all_predictions, key=lambda x: x['cer'], reverse=True)[:10]
    for pred in worst_predictions:
        print(f"\n[{pred['index']}] CER: {pred['cer']*100:.1f}%, WER: {pred['wer']*100:.1f}%")
        print(f"  Target:     '{pred['target']}'")
        print(f"  Prediction: '{pred['prediction']}'")

    # Error analysis
    print("\n" + "="*80)
    print("ERROR ANALYSIS")
    print("="*80)

    cer_ranges = {
        'Perfect (0%)': 0,
        'Excellent (<5%)': 0,
        'Good (5-10%)': 0,
        'Fair (10-20%)': 0,
        'Poor (20-50%)': 0,
        'Very Poor (>50%)': 0
    }

    for cer in all_cer:
        cer_pct = cer * 100
        if cer_pct == 0:
            cer_ranges['Perfect (0%)'] += 1
        elif cer_pct < 5:
            cer_ranges['Excellent (<5%)'] += 1
        elif cer_pct < 10:
            cer_ranges['Good (5-10%)'] += 1
        elif cer_pct < 20:
            cer_ranges['Fair (10-20%)'] += 1
        elif cer_pct < 50:
            cer_ranges['Poor (20-50%)'] += 1
        else:
            cer_ranges['Very Poor (>50%)'] += 1

    print("\nDistribution of Character Error Rates:")
    for range_name, count in cer_ranges.items():
        pct = (count / len(all_cer)) * 100
        bar = '█' * int(pct / 2)
        print(f"  {range_name:20s}: {count:3d} ({pct:5.1f}%) {bar}")

    # Save detailed results
    output_file = "evaluation_results.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("COMPREHENSIVE MODEL EVALUATION RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Character Error Rate (CER): {avg_cer:.2f}%\n")
        f.write(f"Word Error Rate (WER): {avg_wer:.2f}%\n")
        f.write(f"Sentence Accuracy: {sentence_accuracy:.2f}%\n\n")
        f.write("="*80 + "\n")
        f.write("ALL PREDICTIONS\n")
        f.write("="*80 + "\n\n")

        for pred in all_predictions:
            status = "✅ PERFECT" if pred['perfect'] else f"❌ CER: {pred['cer']*100:.1f}%"
            f.write(f"[{pred['index']:3d}] {status}\n")
            f.write(f"  Target:     '{pred['target']}'\n")
            f.write(f"  Prediction: '{pred['prediction']}'\n")
            f.write(f"  CER: {pred['cer']*100:.2f}%, WER: {pred['wer']*100:.2f}%\n\n")

    print(f"\n✅ Detailed results saved to '{output_file}'")

    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)

if __name__ == "__main__":
    evaluate_model()
