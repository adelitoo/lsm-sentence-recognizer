#!/usr/bin/env python3
"""
Complete Pipeline for 500-Sentence Dataset with Sentence-Level Split

This script runs the entire pipeline:
1. Audio encoding (spike train generation)
2. Balanced sentence-level split creation
3. LSM feature extraction with sentence-level split
4. CTC training (requires train_ctc.py modification to load 500-sentence data)

Usage:
    python run_pipeline_500.py [--multiplier 0.8] [--skip-encoding]
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
    parser = argparse.ArgumentParser(description="Run complete 500-sentence pipeline")
    parser.add_argument("--multiplier", type=float, default=0.8,
                       help="LSM weight multiplier (default: 0.8)")
    parser.add_argument("--skip-encoding", action="store_true",
                       help="Skip audio encoding if spike trains already exist")
    parser.add_argument("--skip-split", action="store_true",
                       help="Skip balanced split creation if it already exists")
    args = parser.parse_args()

    print("\n" + "="*80)
    print("500-SENTENCE DATASET PIPELINE")
    print("Sentence-Level Split for True Generalization Testing")
    print("="*80)

    # Check prerequisites
    print("\n📋 Checking prerequisites...")

    if not check_file_exists("sentences_500/sentences.csv", "500-sentence dataset"):
        print("\n❌ Error: 500-sentence dataset not found!")
        print("Please ensure 'sentences_500/sentences.csv' exists")
        print("Run: python generate_500_sentences.py")
        sys.exit(1)

    # Step 1: Audio Encoding (Spike Train Generation)
    if args.skip_encoding and Path("sentence_spike_trains_500.npz").exists():
        print("\n⏭️  Skipping audio encoding (spike trains already exist)")
    else:
        run_command(
            ["python", "audio_encoding_500.py", "--filterbank", "gammatone", "--n-filters", "128"],
            "STEP 1: Audio Encoding (Spike Train Generation)"
        )

    # Verify spike trains were created
    if not check_file_exists("sentence_spike_trains_500.npz", "spike train dataset"):
        print("\n❌ Error: Audio encoding failed to create spike trains")
        sys.exit(1)

    # Step 2: Create Balanced Sentence Split
    if args.skip_split and Path("balanced_sentence_split_500.npz").exists():
        print("\n⏭️  Skipping balanced split creation (split already exists)")
    else:
        run_command(
            ["python", "create_balanced_sentence_split_500.py"],
            "STEP 2: Create Balanced Sentence-Level Split"
        )

    # Verify split was created
    if not check_file_exists("balanced_sentence_split_500.npz", "balanced split"):
        print("\n⚠️  Warning: Balanced split not found, will use random split")

    # Step 3: LSM Feature Extraction with Sentence-Level Split
    run_command(
        ["python", "extract_lsm_windowed_features_filtered_sentence_split_500.py",
         "--multiplier", str(args.multiplier)],
        f"STEP 3: LSM Feature Extraction (multiplier={args.multiplier})"
    )

    # Verify features were extracted
    if not check_file_exists("lsm_windowed_features_filtered_sentence_split_500.npz",
                            "LSM features"):
        print("\n❌ Error: Feature extraction failed")
        sys.exit(1)

    # Final summary
    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETE!")
    print("="*80)
    print("\n📊 Generated files:")
    print("  1. sentence_spike_trains_500.npz - Spike train dataset")
    print("  2. sentence_label_map_500.txt - Label mapping")
    print("  3. balanced_sentence_split_500.npz - Train/test split")
    print("  4. lsm_windowed_features_filtered_sentence_split_500.npz - LSM features")

    print("\n🎯 Dataset Summary:")
    print("  - Total sentences: 500")
    print("  - Train sentences: ~400 (80%)")
    print("  - Test sentences: ~100 (20%)")
    print("  - Split type: SENTENCE-LEVEL (true generalization)")

    print("\n📝 Next steps:")
    print("  To train the CTC model, you need to modify train_ctc.py to support")
    print("  loading from 'lsm_windowed_features_filtered_sentence_split_500.npz'")
    print("  and 'sentence_label_map_500.txt'")
    print()
    print("  Quick modification:")
    print("    1. Add '500' variant detection in train_ctc.py")
    print("    2. Or manually specify the file in train_ctc.py")
    print("    3. Run: python train_ctc.py")
    print()
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
