"""
Quick script to verify and compare sample-level vs sentence-level splits

This loads the spike dataset and shows you exactly what sentences appear in
train vs test for both split methods.
"""

import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

def load_label_map(filepath="sentence_label_map.txt"):
    """Load sentence text for each label ID"""
    if not Path(filepath).exists():
        print(f"❌ Error: Label map not found at '{filepath}'")
        return None

    label_map = {}
    with open(filepath, "r") as f:
        next(f)  # Skip header
        for line in f:
            idx, text = line.strip().split(",", 1)
            label_map[int(idx)] = text
    return label_map

def analyze_sample_level_split():
    """Analyze the current sample-level split"""
    print("\n" + "="*80)
    print("SAMPLE-LEVEL SPLIT (Current Method)")
    print("="*80)

    # Load data
    data = np.load("sentence_spike_trains.npz")
    X_spikes = data['X_spikes']
    y_labels = data['y_labels']

    # Load label map
    label_map = load_label_map()
    if label_map is None:
        return

    # Split exactly as current code does
    X_train, X_test, y_train, y_test = train_test_split(
        X_spikes, y_labels, test_size=0.2, random_state=42
    )

    print(f"\nDataset overview:")
    print(f"  Total samples: {len(X_spikes)}")
    print(f"  Train samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")

    # Analyze sentence distribution
    unique_train_sentences = set(y_train)
    unique_test_sentences = set(y_test)
    overlap_sentences = unique_train_sentences & unique_test_sentences

    print(f"\nSentence distribution:")
    print(f"  Unique train sentences: {len(unique_train_sentences)}")
    print(f"  Unique test sentences: {len(unique_test_sentences)}")
    print(f"  ⚠️  OVERLAP (sentences in both): {len(overlap_sentences)}")

    # Show examples of overlap
    print(f"\nExamples of sentences appearing in BOTH train and test:")
    for i, sentence_id in enumerate(sorted(overlap_sentences)[:5]):
        train_count = np.sum(y_train == sentence_id)
        test_count = np.sum(y_test == sentence_id)
        sentence_text = label_map.get(sentence_id, "Unknown")
        print(f"  [{sentence_id}] \"{sentence_text}\"")
        print(f"       Train: {train_count} samples, Test: {test_count} samples")

    if len(overlap_sentences) > 5:
        print(f"  ... and {len(overlap_sentences) - 5} more")

    print(f"\n✅ What this tests:")
    print(f"   - Robustness to audio variations (pitch, speed, noise)")
    print(f"   - Same vocabulary, different recording conditions")

    print(f"\n❌ What this DOESN'T test:")
    print(f"   - Generalization to completely new sentences")
    print(f"   - Recognition of unseen vocabulary")

def analyze_sentence_level_split():
    """Analyze the proposed sentence-level split"""
    print("\n" + "="*80)
    print("SENTENCE-LEVEL SPLIT (Proposed Method)")
    print("="*80)

    # Load data
    data = np.load("sentence_spike_trains.npz")
    X_spikes = data['X_spikes']
    y_labels = data['y_labels']

    # Load label map
    label_map = load_label_map()
    if label_map is None:
        return

    # Split by sentence (proposed method)
    unique_sentence_ids = np.unique(y_labels)
    train_sentence_ids, test_sentence_ids = train_test_split(
        unique_sentence_ids, test_size=0.2, random_state=42
    )

    # Create train/test split
    train_mask = np.isin(y_labels, train_sentence_ids)
    test_mask = np.isin(y_labels, test_sentence_ids)

    X_train = X_spikes[train_mask]
    y_train = y_labels[train_mask]
    X_test = X_spikes[test_mask]
    y_test = y_labels[test_mask]

    print(f"\nDataset overview:")
    print(f"  Total samples: {len(X_spikes)}")
    print(f"  Train samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")

    # Analyze sentence distribution
    unique_train_sentences = set(y_train)
    unique_test_sentences = set(y_test)
    overlap_sentences = unique_train_sentences & unique_test_sentences

    print(f"\nSentence distribution:")
    print(f"  Unique train sentences: {len(unique_train_sentences)}")
    print(f"  Unique test sentences: {len(unique_test_sentences)}")
    print(f"  ✅ OVERLAP (sentences in both): {len(overlap_sentences)}")

    if len(overlap_sentences) == 0:
        print(f"\n✅ VERIFIED: No sentence overlap!")
        print(f"   Test set contains COMPLETELY UNSEEN sentences.")

    # Show example train sentences
    print(f"\nExample TRAIN sentences:")
    for i, sentence_id in enumerate(sorted(train_sentence_ids)[:5]):
        count = np.sum(y_train == sentence_id)
        sentence_text = label_map.get(sentence_id, "Unknown")
        print(f"  [{sentence_id}] \"{sentence_text}\" ({count} samples)")
    print(f"  ... and {len(train_sentence_ids) - 5} more")

    # Show example test sentences
    print(f"\nExample TEST sentences (NEVER seen during training):")
    for i, sentence_id in enumerate(sorted(test_sentence_ids)[:5]):
        count = np.sum(y_test == sentence_id)
        sentence_text = label_map.get(sentence_id, "Unknown")
        print(f"  [{sentence_id}] \"{sentence_text}\" ({count} samples)")
    print(f"  ... and {len(test_sentence_ids) - 5} more")

    print(f"\n✅ What this tests:")
    print(f"   - TRUE generalization to new sentences")
    print(f"   - Recognition of unseen word combinations")
    print(f"   - Character-level learning (not word memorization)")

    print(f"\n⚠️  Expected:")
    print(f"   - Lower accuracy than sample-level split (70-85% vs 85-95%)")
    print(f"   - More challenging, but stronger proof of learning")

def compare_splits():
    """Compare both split methods side-by-side"""
    print("\n" + "="*80)
    print("COMPARISON: Sample-Level vs Sentence-Level")
    print("="*80)

    # Load data
    data = np.load("sentence_spike_trains.npz")
    X_spikes = data['X_spikes']
    y_labels = data['y_labels']

    label_map = load_label_map()
    if label_map is None:
        return

    print(f"\n{'Aspect':<30} | {'Sample-Level':<25} | {'Sentence-Level':<25}")
    print("-" * 80)

    # Sample-level analysis
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        X_spikes, y_labels, test_size=0.2, random_state=42
    )
    overlap_s = len(set(y_train_s) & set(y_test_s))

    # Sentence-level analysis
    unique_sentence_ids = np.unique(y_labels)
    train_sentence_ids, test_sentence_ids = train_test_split(
        unique_sentence_ids, test_size=0.2, random_state=42
    )
    train_mask = np.isin(y_labels, train_sentence_ids)
    test_mask = np.isin(y_labels, test_sentence_ids)
    X_train_l = X_spikes[train_mask]
    X_test_l = X_spikes[test_mask]
    overlap_l = len(set(train_sentence_ids) & set(test_sentence_ids))

    print(f"{'Train samples':<30} | {len(X_train_s):<25} | {len(X_train_l):<25}")
    print(f"{'Test samples':<30} | {len(X_test_s):<25} | {len(X_test_l):<25}")
    print(f"{'Train sentences':<30} | {len(set(y_train_s)):<25} | {len(train_sentence_ids):<25}")
    print(f"{'Test sentences':<30} | {len(set(y_test_s)):<25} | {len(test_sentence_ids):<25}")
    print(f"{'Sentence overlap':<30} | {overlap_s} ⚠️ MANY!{' '*12} | {overlap_l} ✅ NONE!{' '*13}")
    print(f"{'Expected accuracy':<30} | {'85-95%':<25} | {'70-85%':<25}")
    print(f"{'Tests':<30} | {'Robustness':<25} | {'Generalization':<25}")

if __name__ == "__main__":
    print("\n" + "="*80)
    print("SPLIT METHOD VERIFICATION & COMPARISON")
    print("="*80)

    try:
        # Check if dataset exists
        if not Path("sentence_spike_trains.npz").exists():
            print("\n❌ Error: sentence_spike_trains.npz not found.")
            print("   Please run: python audio_encoding.py")
            exit(1)

        if not Path("sentence_label_map.txt").exists():
            print("\n❌ Error: sentence_label_map.txt not found.")
            print("   Please run: python audio_encoding.py")
            exit(1)

        # Analyze both methods
        analyze_sample_level_split()
        analyze_sentence_level_split()
        compare_splits()

        print("\n" + "="*80)
        print("RECOMMENDATION")
        print("="*80)
        print("\n🎯 Use BOTH split methods for comprehensive evaluation:")
        print("\n  1. Sample-level split:")
        print("     - Tests robustness to audio variations")
        print("     - More realistic for production deployment")
        print("     - Run: python extract_lsm_windowed_features_filtered.py")
        print("            python train_ctc.py")

        print("\n  2. Sentence-level split:")
        print("     - Tests true generalization to new vocabulary")
        print("     - Proves character-level learning")
        print("     - Run: python extract_lsm_windowed_features_filtered_sentence_split.py")
        print("            python train_ctc.py (modify to load sentence_split.npz)")

        print("\n  Then report BOTH results in your evaluation!")
        print("="*80 + "\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
