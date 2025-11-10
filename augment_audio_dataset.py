"""
Audio Data Augmentation for Small Datasets

This script creates augmented versions of existing audio files to increase
dataset size. Augmentations include:
- Time stretching (speed variations)
- Pitch shifting
- Adding background noise
- Volume changes

This helps when you have too few recordings for CTC training.
"""

import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import pandas as pd

# Augmentation parameters
NUM_AUGMENTATIONS_PER_FILE = 5  # Create 5 versions of each audio file (100 sentences × 5 = 500 augmented)
SAMPLE_RATE = 16000

# Augmentation ranges
TIME_STRETCH_RANGE = (0.9, 1.1)  # 90% to 110% speed
PITCH_SHIFT_RANGE = (-2, 2)  # Semitones
NOISE_LEVEL_RANGE = (0.001, 0.005)  # Very subtle noise
VOLUME_RANGE = (0.8, 1.2)  # 80% to 120% volume


def add_noise(audio, noise_level):
    """Add Gaussian noise to audio"""
    noise = np.random.randn(len(audio)) * noise_level
    return audio + noise


def change_volume(audio, factor):
    """Change volume of audio"""
    return audio * factor


def time_stretch(audio, rate):
    """Time stretch audio (change speed without changing pitch)"""
    return librosa.effects.time_stretch(audio, rate=rate)


def pitch_shift(audio, sr, n_steps):
    """Pitch shift audio"""
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)


def augment_audio_file(audio_path, output_dir, base_filename, label_text, augmentations_per_file):
    """
    Create multiple augmented versions of a single audio file

    Returns list of (filename, label_text) tuples for CSV
    """
    # Load original audio
    audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

    augmented_files = []

    for aug_idx in range(augmentations_per_file):
        # Create augmented version
        augmented = audio.copy()

        # Apply random augmentations
        # 1. Time stretch (50% chance)
        if np.random.rand() > 0.5:
            stretch_rate = np.random.uniform(*TIME_STRETCH_RANGE)
            augmented = time_stretch(augmented, stretch_rate)

        # 2. Pitch shift (50% chance)
        if np.random.rand() > 0.5:
            pitch_steps = np.random.uniform(*PITCH_SHIFT_RANGE)
            augmented = pitch_shift(augmented, SAMPLE_RATE, pitch_steps)

        # 3. Add noise (75% chance)
        if np.random.rand() > 0.25:
            noise_level = np.random.uniform(*NOISE_LEVEL_RANGE)
            augmented = add_noise(augmented, noise_level)

        # 4. Volume change (always apply)
        volume_factor = np.random.uniform(*VOLUME_RANGE)
        augmented = change_volume(augmented, volume_factor)

        # Normalize to prevent clipping
        augmented = augmented / (np.max(np.abs(augmented)) + 1e-8)

        # Save augmented audio
        output_filename = f"{base_filename}_aug{aug_idx:02d}.wav"
        output_path = output_dir / output_filename
        sf.write(output_path, augmented, SAMPLE_RATE)

        augmented_files.append((output_filename, label_text))

    return augmented_files


def main():
    print("="*60)
    print("Audio Data Augmentation")
    print("="*60)

    # Paths
    original_dir = Path("sentences")
    augmented_dir = Path("sentences_augmented")
    augmented_dir.mkdir(exist_ok=True)

    metadata_file = original_dir / "sentences.csv"

    if not metadata_file.exists():
        print(f"❌ Error: {metadata_file} not found!")
        print("Please run generate_sentences.py first.")
        return

    # Load original metadata
    print(f"\nLoading metadata from {metadata_file}")
    metadata = pd.read_csv(metadata_file)

    print(f"Original dataset: {len(metadata)} samples")
    print(f"Creating {NUM_AUGMENTATIONS_PER_FILE} augmentations per sample...")
    print(f"Target dataset size: {len(metadata) * (NUM_AUGMENTATIONS_PER_FILE + 1)} samples")
    print()

    # Copy original files and create augmented versions
    all_augmented_metadata = []

    for idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Augmenting audio"):
        original_filename = row['filename']
        label_text = row['label_text']
        original_path = original_dir / original_filename

        if not original_path.exists():
            print(f"\n⚠️  Warning: {original_path} not found, skipping...")
            continue

        # Copy original file to augmented directory
        audio, sr = librosa.load(original_path, sr=SAMPLE_RATE, mono=True)
        base_name = Path(original_filename).stem
        original_output = augmented_dir / f"{base_name}_orig.wav"
        sf.write(original_output, audio, SAMPLE_RATE)
        all_augmented_metadata.append((f"{base_name}_orig.wav", label_text))

        # Create augmented versions
        augmented_files = augment_audio_file(
            original_path,
            augmented_dir,
            base_name,
            label_text,
            NUM_AUGMENTATIONS_PER_FILE
        )
        all_augmented_metadata.extend(augmented_files)

    # Save augmented metadata
    augmented_csv = augmented_dir / "sentences.csv"
    augmented_df = pd.DataFrame(all_augmented_metadata, columns=['filename', 'label_text'])
    augmented_df.to_csv(augmented_csv, index=False)

    print("\n" + "="*60)
    print("✅ Augmentation Complete!")
    print("="*60)
    print(f"Original samples: {len(metadata)}")
    print(f"Augmented samples: {len(augmented_df)}")
    print(f"Increase: {len(augmented_df) / len(metadata):.1f}x")
    print(f"\nAugmented files saved to: {augmented_dir}/")
    print(f"Metadata saved to: {augmented_csv}")
    print("\nNext steps:")
    print("  1. Update audio_encoding.py to use 'sentences_augmented' directory")
    print("  2. Run the pipeline again with more data")
    print("="*60)


if __name__ == "__main__":
    main()
