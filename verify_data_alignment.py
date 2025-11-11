"""
Verify that LSM traces actually correspond to their labels
"""
import numpy as np
from pathlib import Path

def load_label_map(filepath="sentence_label_map_500.txt"):
    """Loads the sentence_label_map.txt file"""
    if not Path(filepath).exists():
        print(f"❌ Error: Label map not found at '{filepath}'")
        return None

    label_map = {}
    with open(filepath, "r") as f:
        next(f)  # Skip header
        for line in f:
            idx, text = line.strip().split(",", 1)
            label_map[int(idx)] = text.lower()
    return label_map

def main():
    print("="*60)
    print("VERIFYING DATA ALIGNMENT")
    print("="*60)

    # Load label map
    label_map = load_label_map()
    if label_map is None:
        return

    print(f"\n✅ Loaded label map with {len(label_map)} entries")

    # Load trace data
    trace_file = "lsm_trace_sequences_sentence_split_500.npz"
    if not Path(trace_file).exists():
        print(f"❌ Error: Trace file not found")
        return

    dataset = np.load(trace_file, allow_pickle=True)
    X_train = dataset["X_train_sequences"]
    y_train = dataset["y_train"]
    X_test = dataset["X_test_sequences"]
    y_test = dataset["y_test"]

    print(f"\n✅ Loaded trace data")
    print(f"   Train: {len(X_train)} samples, labels: {len(y_train)}")
    print(f"   Test: {len(X_test)} samples, labels: {len(y_test)}")

    # Check if labels are in range
    print(f"\n--- Label Range Check ---")
    print(f"  Train labels range: [{y_train.min()}, {y_train.max()}]")
    print(f"  Test labels range: [{y_test.min()}, {y_test.max()}]")
    print(f"  Label map keys range: [{min(label_map.keys())}, {max(label_map.keys())}]")

    # Check if all labels exist in map
    train_not_in_map = [lbl for lbl in y_train if lbl not in label_map]
    test_not_in_map = [lbl for lbl in y_test if lbl not in label_map]

    if train_not_in_map:
        print(f"\n❌ ERROR: {len(train_not_in_map)} train labels not in label map!")
        print(f"   Examples: {train_not_in_map[:5]}")
    else:
        print(f"\n✅ All train labels exist in label map")

    if test_not_in_map:
        print(f"❌ ERROR: {len(test_not_in_map)} test labels not in label map!")
        print(f"   Examples: {test_not_in_map[:5]}")
    else:
        print(f"✅ All test labels exist in label map")

    # Check sentence distribution
    print(f"\n--- Sentence Distribution ---")
    unique_train_sentences = np.unique(y_train)
    unique_test_sentences = np.unique(y_test)

    print(f"  Unique train sentences: {len(unique_train_sentences)}")
    print(f"  Unique test sentences: {len(unique_test_sentences)}")

    # Check for overlap
    overlap = set(unique_train_sentences) & set(unique_test_sentences)
    if overlap:
        print(f"\n❌ ERROR: {len(overlap)} sentences appear in BOTH train and test!")
        print(f"   Examples: {list(overlap)[:5]}")
    else:
        print(f"\n✅ No sentence overlap (good for generalization)")

    # Show examples of train and test sentences
    print(f"\n--- Example Sentences ---")
    print(f"\nFirst 5 TRAIN sentences:")
    for i in range(min(5, len(y_train))):
        label_id = y_train[i]
        text = label_map[label_id]
        print(f"  Sample {i}: ID={label_id}, Text='{text}'")

    print(f"\nFirst 5 TEST sentences:")
    for i in range(min(5, len(y_test))):
        label_id = y_test[i]
        text = label_map[label_id]
        print(f"  Sample {i}: ID={label_id}, Text='{text}'")

    # Load original spike data to verify split consistency
    spike_file = "sentence_spike_trains_500.npz"
    if Path(spike_file).exists():
        print(f"\n--- Checking Original Spike Data ---")
        spike_data = np.load(spike_file)
        X_spikes_all = spike_data['X_spikes']
        y_labels_all = spike_data['y_labels']

        print(f"  Original data: {len(X_spikes_all)} samples")
        print(f"  Train + Test: {len(y_train) + len(y_test)} samples")

        if len(X_spikes_all) == len(y_train) + len(y_test):
            print(f"✅ Sample count matches")
        else:
            print(f"❌ Sample count MISMATCH!")

        # Check if the first train label matches
        balanced_split = np.load("balanced_sentence_split_500.npz")
        train_sentence_ids = balanced_split['train_sentence_ids']

        # Find first sample in original data that belongs to train set
        train_mask = np.isin(y_labels_all, train_sentence_ids)
        first_train_idx_in_original = np.where(train_mask)[0][0]
        original_first_train_label = y_labels_all[first_train_idx_in_original]

        print(f"\n  First train sample in ORIGINAL data: ID={original_first_train_label}")
        print(f"  First train sample in TRACES: ID={y_train[0]}")

        if original_first_train_label == y_train[0]:
            print(f"✅ Labels match!")
        else:
            print(f"❌ Labels DON'T match - ALIGNMENT PROBLEM!")

    print("\n" + "="*60)

if __name__ == "__main__":
    main()
