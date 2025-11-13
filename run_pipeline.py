#!/usr/bin/env python3
"""
Complete Pipeline for 2000-Sentence Dataset with Sentence-Level Split

This script runs the entire pipeline:
1. Audio encoding (spike train generation)
2. Balanced sentence-level split creation
3. LSM trace extraction with sentence-level split

Usage:
    python run_pipeline_2000.py [--multiplier 1.0] [--leak 0.1] [--skip-encoding] [--skip-split]
"""

import subprocess
import sys
from pathlib import Path
import argparse


def run_command(cmd, description):
    """Run a command and handle errors"""
    print("\n" + "="*80)
    print(f"{description}")
    print("="*80)
    print(f"Running: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, text=True)

    if result.returncode != 0:
        print(f"\n❌ Error: {description} failed with exit code {result.returncode}")
        sys.exit(1)

    print(f"\n✅ {description} completed successfully!")
    return result


def check_file_exists(filepath, description):
    """Check if a required file exists"""
    if Path(filepath).exists():
        print(f"✅ Found: {filepath}")
        return True
    else:
        print(f"❌ Missing: {filepath} - {description}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run complete 2000-sentence pipeline")
    parser.add_argument("--multiplier", type=float, default=1.0,
                       help="LSM weight multiplier (default: 1.0)")
    parser.add_argument("--leak", type=float, default=0.1,
                       help="LSM leak coefficient (default: 0.1)")
    parser.add_argument("--skip-encoding", action="store_true",
                       help="Skip audio encoding if spike trains already exist")
    parser.add_argument("--skip-split", action="store_true",
                       help="Skip balanced split creation if it already exists")
    args = parser.parse_args()

    print("\n" + "="*80)
    print("SENTENCE DATASET PIPELINE")
    print("Sentence-Level Split for True Generalization Testing")
    print("="*80)

    # Check prerequisites
    print("\n📋 Checking prerequisites...")

    if not check_file_exists("sentences/sentences.csv", "sentence dataset"):
        print("\n❌ Error: Sentence dataset not found!")
        print("Please ensure 'sentences/sentences.csv' exists")
        print("Run: python generate_sentences.py")
        sys.exit(1)

    # Step 1: Audio Encoding (Spike Train Generation)
    spike_file = "sentence_spike_trains.npz"
    label_file = "sentence_label_map.txt"

    if args.skip_encoding and Path(spike_file).exists():
        print(f"\n⏭️  Skipping audio encoding ({spike_file} already exists)")
    else:
        run_command(
            ["python", "audio_encoding.py",
             "--filterbank", "gammatone",
             "--n-filters", "128"],
            "STEP 1: Audio Encoding (Spike Train Generation)"
        )

    # Verify spike trains were created
    if not check_file_exists(spike_file, "spike train dataset"):
        print("\n❌ Error: Audio encoding failed to create spike trains")
        sys.exit(1)

    # Check how many sentences were actually processed
    import numpy as np
    data = np.load(spike_file)
    num_sentences = len(data["y_labels"])
    print(f"✅ Processed {num_sentences} sentences")

    # Step 2: Create Balanced Sentence Split
    split_file = "balanced_sentence_split.npz"
    if args.skip_split and Path(split_file).exists():
        print(f"\n⏭️  Skipping balanced split creation ({split_file} already exists)")
    else:
        run_command(
            ["python", "create_balanced_sentence_split.py"],
            "STEP 2: Create Balanced Sentence-Level Split"
        )

    # Verify split was created
    if not check_file_exists(split_file, "balanced split"):
        print("\n⚠️  Warning: Balanced split not found, will use random split")

    # Step 3: LSM Trace Extraction with Sentence-Level Split
    trace_file = "lsm_trace_sequences.npz"
    run_command(
        ["python", "extract_lsm_traces.py",
         "--multiplier", str(args.multiplier)],
        f"STEP 3: LSM Trace Extraction (multiplier={args.multiplier}, leak=0.1 hardcoded)"
    )

    # Verify traces were extracted
    if not check_file_exists(trace_file, "LSM traces"):
        print("\n❌ Error: Trace extraction failed")
        sys.exit(1)

    # Check final dataset size
    data = np.load(trace_file)
    num_train = len(data["X_train_sequences"])
    num_test = len(data["X_test_sequences"])

    # Final summary
    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETE!")
    print("="*80)
    print("\n📊 Generated files:")
    print(f"  1. {spike_file} - Spike train dataset ({num_sentences} sentences)")
    print(f"  2. {label_file} - Label mapping")
    print(f"  3. {split_file} - Train/test split")
    print(f"  4. {trace_file} - LSM traces")

    print("\n🎯 Dataset Summary:")
    print(f"  - Total sentences: {num_sentences}")
    print(f"  - Train sentences: {num_train} ({num_train/num_sentences*100:.0f}%)")
    print(f"  - Test sentences: {num_test} ({num_test/num_sentences*100:.0f}%)")
    print("  - Split type: SENTENCE-LEVEL (true generalization)")

    print("\n📝 Next steps:")
    print("  1. Run diagnostic: python diagnose_lsm_features.py")
    print("     - Check separability ratio (should be >0.4)")
    print("  2. Train model: python train_ctc_traces.py")
    print(f"  3. Expected accuracy: 85-90% character accuracy (up from 70%)")
    print()
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
