"""
Complete Pipeline for 500-Sentence Dataset

This script runs the entire pipeline:
1. Generate 500 sentences (if not already generated)
2. Encode audio to spike trains
3. Extract LSM features with sentence-level split
4. Train CTC model
5. Evaluate results

Estimated time: 30-60 minutes total
"""

import subprocess
import sys
from pathlib import Path
import time

def run_command(cmd, description, timeout=None):
    """Run a command and show progress"""
    print("\n" + "="*80)
    print(f"STEP: {description}")
    print("="*80)
    print(f"Command: {cmd}")
    print()

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=False,
            text=True,
            timeout=timeout
        )

        elapsed = time.time() - start_time
        print(f"\n✅ Completed in {elapsed:.1f} seconds")
        return True

    except subprocess.TimeoutExpired:
        print(f"\n⏰ Timeout after {timeout} seconds")
        return False
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error: Command failed with return code {e.returncode}")
        return False

def check_file_exists(filepath, description):
    """Check if a file exists"""
    if Path(filepath).exists():
        print(f"✅ Found: {description} ({filepath})")
        return True
    else:
        print(f"❌ Missing: {description} ({filepath})")
        return False

def main():
    print("\n" + "="*80)
    print("COMPLETE PIPELINE FOR 500-SENTENCE DATASET")
    print("="*80)
    print("\nThis will:")
    print("  1. Generate 500 diverse sentences")
    print("  2. Convert to audio (ElevenLabs API)")
    print("  3. Encode audio to spike trains")
    print("  4. Extract LSM windowed features")
    print("  5. Train CTC model")
    print("  6. Evaluate results")
    print("\nEstimated total time: 30-60 minutes")
    print("="*80)

    response = input("\nDo you want to proceed? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Aborted.")
        return

    # === STEP 1: Generate Sentences ===
    if not check_file_exists("sentences_500/sentences.csv", "500 sentences CSV"):
        success = run_command(
            "python generate_500_sentences.py",
            "Generate 500 sentences with audio (ElevenLabs API)",
            timeout=3600  # 1 hour timeout
        )
        if not success:
            print("\n❌ Failed to generate sentences. Aborting.")
            return
    else:
        print("⏭️  Skipping sentence generation (already exists)")

    # === STEP 2: Audio Encoding ===
    # Need to update audio_encoding.py to use sentences_500 folder
    print("\n" + "="*80)
    print("NOTE: You need to update audio_encoding.py to process")
    print("      sentences_500/ instead of sentences/")
    print("="*80)

    response = input("\nHave you updated audio_encoding.py? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("\nPlease update audio_encoding.py:")
        print("  Change: AUDIO_DIR = 'sentences'")
        print("  To:     AUDIO_DIR = 'sentences_500'")
        print("\nThen run this script again.")
        return

    if not check_file_exists("sentence_spike_trains.npz", "Spike trains"):
        success = run_command(
            "python audio_encoding.py",
            "Encode audio to spike trains",
            timeout=7200  # 2 hour timeout
        )
        if not success:
            print("\n❌ Failed to encode audio. Aborting.")
            return
    else:
        print("⏭️  Skipping audio encoding (spike trains exist)")

    # === STEP 3: Extract LSM Features ===
    if not check_file_exists("lsm_windowed_features_filtered_sentence_split.npz", "LSM features"):
        success = run_command(
            "python extract_lsm_windowed_features_filtered_sentence_split.py --multiplier 1.0",
            "Extract LSM features with sentence-level split",
            timeout=7200  # 2 hour timeout
        )
        if not success:
            print("\n❌ Failed to extract features. Aborting.")
            return
    else:
        print("⏭️  Skipping feature extraction (features exist)")

    # === STEP 4: Train Model ===
    success = run_command(
        "python train_ctc.py",
        "Train CTC readout layer (5000 epochs)",
        timeout=7200  # 2 hour timeout
    )
    if not success:
        print("\n❌ Failed to train model.")
        return

    # === STEP 5: Show Results ===
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print("\nModel saved to: ctc_model_sentence_split.pt")
    print("\nNext steps:")
    print("  - Run: python evaluate_model.py")
    print("  - Run: python compare_splits.py")
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
