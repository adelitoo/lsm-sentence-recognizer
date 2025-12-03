"""
Debug LSM Temporal Information Content

This script analyzes WHY the LSM is collapsing temporally.
We'll visualize the actual membrane traces to see if there's ANY temporal structure.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_label_map(filepath="sentence_label_map.txt"):
    """Load sentence labels"""
    label_map = {}
    with open(filepath, "r") as f:
        next(f)
        for line in f:
            idx, text = line.strip().split(",", 1)
            label_map[int(idx)] = text.lower()
    return label_map


def analyze_temporal_content():
    """Analyze temporal information in LSM traces"""

    print("=" * 80)
    print("🔬 LSM TEMPORAL INFORMATION ANALYSIS")
    print("=" * 80)

    # Load data
    trace_file = "lsm_trace_sequences.npz"
    if not Path(trace_file).exists():
        print(f"❌ Error: {trace_file} not found")
        return

    print(f"\n📂 Loading {trace_file}...")
    data = np.load(trace_file, allow_pickle=True)
    X_train = data["X_train_sequences"]
    y_train = data["y_train"]

    label_map = load_label_map()

    print(f"✅ Loaded {len(X_train)} samples")
    print(f"   Shape: {X_train.shape} (samples, timesteps, neurons)")

    # Select 5 different sentences
    unique_labels = np.unique(y_train)[:5]

    print("\n" + "=" * 80)
    print("📊 ANALYZING TEMPORAL STRUCTURE")
    print("=" * 80)

    fig, axes = plt.subplots(5, 2, figsize=(16, 12))

    for idx, label_id in enumerate(unique_labels):
        # Get first sample for this sentence
        sample_idx = np.where(y_train == label_id)[0][0]
        trace = X_train[sample_idx]  # Shape: (2000, 700)
        sentence = label_map[label_id]

        print(f"\n{idx+1}. Sentence: '{sentence}'")

        # Analysis 1: Average activity over time
        avg_activity = np.mean(trace, axis=1)  # Average across neurons
        std_activity = np.std(trace, axis=1)

        # Analysis 2: Check if there's variation over time
        temporal_variance = np.var(avg_activity)
        print(f"   Temporal variance: {temporal_variance:.6f}")

        # Check active regions
        threshold = np.mean(avg_activity) + 0.5 * np.std(avg_activity)
        active_steps = np.where(avg_activity > threshold)[0]
        print(f"   Active timesteps: {len(active_steps)}/2000 ({len(active_steps)/20:.1f}%)")

        # Plot 1: Average activity over time
        ax1 = axes[idx, 0]
        ax1.plot(avg_activity, linewidth=0.5)
        ax1.fill_between(range(len(avg_activity)),
                         avg_activity - std_activity,
                         avg_activity + std_activity,
                         alpha=0.3)
        ax1.set_title(f"'{sentence}'\nAvg Activity (var={temporal_variance:.2f})")
        ax1.set_xlabel("Time Step")
        ax1.set_ylabel("Mean Membrane Voltage")
        ax1.grid(True, alpha=0.3)

        # Plot 2: Heatmap of first 50 neurons
        ax2 = axes[idx, 1]
        im = ax2.imshow(trace[:, :50].T, aspect='auto', cmap='viridis',
                       interpolation='nearest')
        ax2.set_title(f"First 50 Neurons")
        ax2.set_xlabel("Time Step")
        ax2.set_ylabel("Neuron ID")
        plt.colorbar(im, ax=ax2, label="Membrane Voltage")

    plt.tight_layout()
    plt.savefig("lsm_temporal_analysis.png", dpi=150, bbox_inches='tight')
    print("\n✅ Saved: lsm_temporal_analysis.png")

    # === CRITICAL TEST: Compare different words within sentences ===
    print("\n" + "=" * 80)
    print("🎯 WORD-LEVEL TEMPORAL ANALYSIS")
    print("=" * 80)

    # Pick a sentence and analyze temporal segments
    test_idx = 0
    test_trace = X_train[test_idx]
    test_sentence = label_map[y_train[test_idx]]
    words = test_sentence.split()

    print(f"\nTest sentence: '{test_sentence}'")
    print(f"Words: {words}")
    print(f"Expected temporal distribution: ~{2000//len(words)} steps per word")

    # Divide trace into word segments
    n_words = len(words)
    steps_per_word = 2000 // n_words

    fig, axes = plt.subplots(n_words, 1, figsize=(14, 2*n_words))
    if n_words == 1:
        axes = [axes]

    for word_idx, word in enumerate(words):
        start = word_idx * steps_per_word
        end = start + steps_per_word
        segment = test_trace[start:end]

        avg_segment = np.mean(segment, axis=1)

        ax = axes[word_idx]
        ax.plot(range(start, end), avg_segment, linewidth=1)
        ax.set_title(f"Word '{word}' (steps {start}-{end})")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Avg Membrane Voltage")
        ax.grid(True, alpha=0.3)

        # Statistics
        segment_mean = np.mean(avg_segment)
        segment_std = np.std(avg_segment)
        segment_var = np.var(avg_segment)

        print(f"  '{word}': mean={segment_mean:.2f}, std={segment_std:.2f}, var={segment_var:.2f}")

    plt.tight_layout()
    plt.savefig("lsm_word_segments.png", dpi=150, bbox_inches='tight')
    print("\n✅ Saved: lsm_word_segments.png")

    # === CHECK: Are traces actually different? ===
    print("\n" + "=" * 80)
    print("🔍 TRACE SIMILARITY TEST")
    print("=" * 80)

    # Compare 5 different sentences
    sample_traces = []
    sample_labels = []

    for i in range(min(5, len(unique_labels))):
        label_id = unique_labels[i]
        sample_idx = np.where(y_train == label_id)[0][0]
        sample_traces.append(X_train[sample_idx])
        sample_labels.append(label_map[label_id])

    print("\nComputing trace similarities (cosine)...")
    from sklearn.metrics.pairwise import cosine_similarity

    # Flatten traces for comparison
    flattened = np.array([t.flatten() for t in sample_traces])
    similarities = cosine_similarity(flattened)

    print("\nSimilarity Matrix:")
    print("(1.0 = identical, 0.0 = completely different)")
    print("-" * 60)
    for i in range(len(sample_labels)):
        for j in range(len(sample_labels)):
            if i == j:
                print(f"  [{i}] vs [{j}]: 1.000 (same)")
            elif i < j:
                print(f"  [{i}] vs [{j}]: {similarities[i,j]:.3f}")
                print(f"      '{sample_labels[i][:30]}'")
                print(f"      '{sample_labels[j][:30]}'")

    avg_diff_similarity = np.mean(similarities[np.triu_indices_from(similarities, k=1)])
    print(f"\n⚠️  Average similarity between DIFFERENT sentences: {avg_diff_similarity:.3f}")
    print(f"   (Should be < 0.5 for good separability)")

    if avg_diff_similarity > 0.9:
        print("\n❌ CRITICAL PROBLEM: All traces are nearly identical!")
        print("   The LSM is NOT encoding temporal/phonetic information.")
        print("\n💡 Possible causes:")
        print("   1. LSM leak rate too high → neurons forget too quickly")
        print("   2. Weight multiplier too low → weak recurrent dynamics")
        print("   3. Insufficient reservoir connectivity")
        print("   4. Input spike encoding is too sparse/uniform")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    analyze_temporal_content()
