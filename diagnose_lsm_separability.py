"""
Diagnose LSM Feature Separability

This script analyzes WHY the LSM features alone aren't sufficient.
We'll check:
1. Sentence-level separability (cosine similarity)
2. Temporal dynamics (do features change over time?)
3. Feature variance (are neurons actually responding?)
4. Neuron utilization (how many neurons are active?)

This will tell us WHAT to fix in the LSM.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import seaborn as sns

def load_label_map(filepath="sentence_label_map.txt"):
    """Load sentence labels"""
    label_map = {}
    with open(filepath, "r") as f:
        next(f)
        for line in f:
            idx, text = line.strip().split(",", 1)
            label_map[int(idx)] = text.lower()
    return label_map


def analyze_lsm_features():
    """Main analysis function"""

    print("=" * 80)
    print("🔬 LSM FEATURE SEPARABILITY DIAGNOSIS")
    print("=" * 80)

    # Load data
    trace_file = "lsm_trace_sequences.npz"
    if not Path(trace_file).exists():
        print(f"❌ Error: {trace_file} not found")
        return

    print(f"\n📂 Loading data from {trace_file}...")
    data = np.load(trace_file, allow_pickle=True)

    X_train = data["X_train_sequences"]  # Shape: (n_samples, timesteps, features)
    y_train = data["y_train"]
    X_test = data["X_test_sequences"]
    y_test = data["y_test"]

    label_map = load_label_map()
    y_train_text = [label_map[idx] for idx in y_train]
    y_test_text = [label_map[idx] for idx in y_test]

    n_train, n_timesteps, n_features = X_train.shape

    print(f"✅ Data loaded:")
    print(f"   Train: {n_train} samples")
    print(f"   Test: {len(X_test)} samples")
    print(f"   Timesteps: {n_timesteps}")
    print(f"   LSM output neurons: {n_features}")

    # =========================================================================
    # ANALYSIS 1: Feature Variance (Are neurons responding?)
    # =========================================================================
    print("\n" + "=" * 80)
    print("📊 ANALYSIS 1: Neuron Activity & Variance")
    print("=" * 80)

    # Compute variance across all samples and time
    X_train_flat = X_train.reshape(-1, n_features)
    feature_variance = np.var(X_train_flat, axis=0)
    feature_mean = np.mean(X_train_flat, axis=0)

    # How many neurons are actually active?
    active_neurons = np.sum(feature_variance > 0.01)
    silent_neurons = n_features - active_neurons

    print(f"\n🔬 Neuron Activity:")
    print(f"   Active neurons (variance > 0.01): {active_neurons}/{n_features} ({active_neurons/n_features*100:.1f}%)")
    print(f"   Silent neurons: {silent_neurons}/{n_features} ({silent_neurons/n_features*100:.1f}%)")
    print(f"   Mean variance: {np.mean(feature_variance):.4f}")
    print(f"   Median variance: {np.median(feature_variance):.4f}")

    if silent_neurons > n_features * 0.3:
        print(f"   ⚠️  WARNING: {silent_neurons/n_features*100:.0f}% of neurons are silent!")
        print(f"      → LSM may be underutilized")

    # Plot variance distribution
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.hist(feature_variance, bins=50, edgecolor='black')
    plt.xlabel('Variance')
    plt.ylabel('Number of Neurons')
    plt.title('Distribution of Neuron Variance')
    plt.axvline(0.01, color='red', linestyle='--', label='Activity threshold')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(sorted(feature_variance, reverse=True))
    plt.xlabel('Neuron Index (sorted)')
    plt.ylabel('Variance')
    plt.title('Neuron Variance (sorted)')
    plt.yscale('log')

    plt.tight_layout()
    plt.savefig('lsm_neuron_variance.png', dpi=150, bbox_inches='tight')
    print(f"\n💾 Saved: lsm_neuron_variance.png")

    # =========================================================================
    # ANALYSIS 2: Temporal Dynamics (Do features evolve over time?)
    # =========================================================================
    print("\n" + "=" * 80)
    print("⏱️  ANALYSIS 2: Temporal Dynamics")
    print("=" * 80)

    # For each sample, measure how much features change over time
    temporal_variance = []
    for i in range(min(100, n_train)):  # Check first 100 samples
        sample = X_train[i]  # Shape: (timesteps, features)
        # Variance over time for each feature
        var_over_time = np.var(sample, axis=0).mean()
        temporal_variance.append(var_over_time)

    avg_temporal_var = np.mean(temporal_variance)

    print(f"\n🔬 Temporal Dynamics:")
    print(f"   Average temporal variance: {avg_temporal_var:.4f}")

    if avg_temporal_var < 0.01:
        print(f"   ⚠️  WARNING: Features are nearly static over time!")
        print(f"      → LSM may have too fast decay (leak too high)")
        print(f"      → Or membrane potentials saturate quickly")
    elif avg_temporal_var > 0.5:
        print(f"   ✅ Good: Features show temporal dynamics")
    else:
        print(f"   ⚠️  Moderate: Features change but may need enhancement")

    # Visualize example traces
    plt.figure(figsize=(15, 8))

    for i in range(3):
        plt.subplot(3, 1, i+1)
        # Plot first 10 neurons over time for sample i
        for neuron_idx in range(min(10, n_features)):
            plt.plot(X_train[i][:, neuron_idx], alpha=0.7, linewidth=0.8)
        plt.ylabel('Membrane Potential')
        plt.title(f"Sample {i}: '{y_train_text[i]}'")
        if i == 2:
            plt.xlabel('Time Steps')

    plt.tight_layout()
    plt.savefig('lsm_temporal_traces.png', dpi=150, bbox_inches='tight')
    print(f"💾 Saved: lsm_temporal_traces.png")

    # =========================================================================
    # ANALYSIS 3: Sentence-Level Separability (Cosine Similarity)
    # =========================================================================
    print("\n" + "=" * 80)
    print("🎯 ANALYSIS 3: Sentence-Level Separability")
    print("=" * 80)

    # Use mean pooling over time to get sentence representations
    print("\n📐 Computing sentence embeddings (mean pooling)...")
    X_train_pooled = X_train.mean(axis=1)  # Shape: (n_samples, features)
    X_test_pooled = X_test.mean(axis=1)

    # Sample a subset for similarity analysis (too slow for all pairs)
    n_sample = min(100, n_train)
    indices = np.random.choice(n_train, n_sample, replace=False)
    X_sample = X_train_pooled[indices]
    y_sample = [y_train[i] for i in indices]
    y_sample_text = [y_train_text[i] for i in indices]

    # Compute pairwise cosine similarity
    print(f"🔬 Computing pairwise similarities for {n_sample} samples...")
    similarity_matrix = cosine_similarity(X_sample)

    # Separate same-sentence vs different-sentence similarities
    same_sentence_sims = []
    diff_sentence_sims = []

    for i in range(n_sample):
        for j in range(i+1, n_sample):
            sim = similarity_matrix[i, j]
            if y_sample[i] == y_sample[j]:
                same_sentence_sims.append(sim)
            else:
                diff_sentence_sims.append(sim)

    # Statistics
    if same_sentence_sims:
        avg_same = np.mean(same_sentence_sims)
        std_same = np.std(same_sentence_sims)
    else:
        avg_same = 0
        std_same = 0

    avg_diff = np.mean(diff_sentence_sims)
    std_diff = np.std(diff_sentence_sims)

    # Separability score
    separability = avg_same - avg_diff

    print(f"\n🔬 Cosine Similarity Analysis:")
    print(f"   Same sentence:      {avg_same:.3f} ± {std_same:.3f}")
    print(f"   Different sentences: {avg_diff:.3f} ± {std_diff:.3f}")
    print(f"   Separability gap:    {separability:.3f}")

    if separability < 0.1:
        print(f"   ❌ CRITICAL: Very low separability!")
        print(f"      → Different sentences look too similar")
        print(f"      → This explains the 2.95% accuracy")
    elif separability < 0.3:
        print(f"   ⚠️  WARNING: Low separability")
        print(f"      → Need to enhance LSM dynamics")
    else:
        print(f"   ✅ Good separability")

    # Visualize similarity distributions
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    if same_sentence_sims:
        plt.hist(same_sentence_sims, bins=30, alpha=0.7, label='Same sentence', color='green')
    plt.hist(diff_sentence_sims, bins=30, alpha=0.7, label='Different sentences', color='red')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Count')
    plt.title('LSM Feature Similarity Distribution')
    plt.legend()
    plt.axvline(avg_same, color='green', linestyle='--', linewidth=2)
    plt.axvline(avg_diff, color='red', linestyle='--', linewidth=2)

    # Heatmap of similarity matrix (subset)
    plt.subplot(1, 2, 2)
    n_viz = min(50, n_sample)
    sns.heatmap(similarity_matrix[:n_viz, :n_viz], cmap='RdYlGn', center=0.5,
                vmin=0, vmax=1, square=True, cbar_kws={'label': 'Cosine Similarity'})
    plt.title(f'Pairwise Similarity Matrix ({n_viz} samples)')
    plt.xlabel('Sample Index')
    plt.ylabel('Sample Index')

    plt.tight_layout()
    plt.savefig('lsm_separability.png', dpi=150, bbox_inches='tight')
    print(f"💾 Saved: lsm_separability.png")

    # =========================================================================
    # ANALYSIS 4: PCA Visualization
    # =========================================================================
    print("\n" + "=" * 80)
    print("📊 ANALYSIS 4: PCA Visualization")
    print("=" * 80)

    print(f"\n🔬 Reducing to 2D with PCA...")
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_pooled)

    explained_var = pca.explained_variance_ratio_
    print(f"   PC1 variance: {explained_var[0]*100:.1f}%")
    print(f"   PC2 variance: {explained_var[1]*100:.1f}%")
    print(f"   Total: {sum(explained_var)*100:.1f}%")

    if sum(explained_var) < 0.3:
        print(f"   ⚠️  WARNING: Low variance captured by first 2 PCs")
        print(f"      → Features may be high-dimensional noise")

    # Plot PCA with different sentence classes
    plt.figure(figsize=(12, 8))

    # Get unique sentences for coloring
    unique_sentences = list(set(y_train))[:20]  # Plot first 20 sentence types

    for sent_id in unique_sentences:
        mask = y_train == sent_id
        plt.scatter(X_train_pca[mask, 0], X_train_pca[mask, 1],
                   alpha=0.6, s=30, label=f"Sent {sent_id}")

    plt.xlabel(f'PC1 ({explained_var[0]*100:.1f}% var)')
    plt.ylabel(f'PC2 ({explained_var[1]*100:.1f}% var)')
    plt.title('LSM Features in 2D (PCA) - Different sentence classes')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('lsm_pca_visualization.png', dpi=150, bbox_inches='tight')
    print(f"💾 Saved: lsm_pca_visualization.png")

    # =========================================================================
    # FINAL DIAGNOSIS & RECOMMENDATIONS
    # =========================================================================
    print("\n" + "=" * 80)
    print("🏥 DIAGNOSIS SUMMARY & RECOMMENDATIONS")
    print("=" * 80)

    issues = []
    recommendations = []

    # Issue 1: Silent neurons
    if silent_neurons > n_features * 0.3:
        issues.append(f"❌ {silent_neurons/n_features*100:.0f}% of neurons are silent")
        recommendations.append("→ Increase input spike rate or reduce threshold")
        recommendations.append("→ Check LSM weight initialization")

    # Issue 2: Low temporal dynamics
    if avg_temporal_var < 0.01:
        issues.append(f"❌ Features are nearly static (temp var: {avg_temporal_var:.4f})")
        recommendations.append("→ DECREASE leak from 0.1 to 0.02-0.05 (longer memory)")
        recommendations.append("→ Increase refractory period for richer dynamics")

    # Issue 3: Low separability
    if separability < 0.2:
        issues.append(f"❌ Very low separability gap: {separability:.3f}")
        recommendations.append("→ INCREASE reservoir size (1000 → 2000-3000 neurons)")
        recommendations.append("→ Increase output neurons (700 → 1000-1500)")
        recommendations.append("→ Try hierarchical LSM with multiple timescales")

    # Issue 4: High similarity
    if avg_diff > 0.8:
        issues.append(f"❌ Different sentences too similar ({avg_diff:.3f})")
        recommendations.append("→ Increase LSM complexity (more neurons)")
        recommendations.append("→ Use multiple LSM layers")

    print("\n🔍 Issues Found:")
    if issues:
        for issue in issues:
            print(f"   {issue}")
    else:
        print("   ✅ No critical issues found (but accuracy is still low)")

    print("\n💡 Recommended Parameter Changes:")
    if recommendations:
        for rec in recommendations:
            print(f"   {rec}")
    else:
        print("   → Consider using a small 1-2 layer MLP readout")
        print("   → Or use word-level CTC instead of character-level")

    print("\n📋 Next Steps:")
    print("   1. Review the generated plots:")
    print("      - lsm_neuron_variance.png")
    print("      - lsm_temporal_traces.png")
    print("      - lsm_separability.png")
    print("      - lsm_pca_visualization.png")
    print()
    print("   2. Apply recommended parameter changes in extract_lsm_traces.py")
    print()
    print("   3. Re-run pipeline:")
    print("      python extract_lsm_traces.py --multiplier 0.8")
    print("      python train_ctc_traces_linear.py")
    print()
    print("   4. Check if separability improves")

    print("\n" + "=" * 80)

    return {
        'active_neurons': active_neurons,
        'temporal_variance': avg_temporal_var,
        'separability': separability,
        'same_sim': avg_same,
        'diff_sim': avg_diff,
    }


if __name__ == "__main__":
    results = analyze_lsm_features()
