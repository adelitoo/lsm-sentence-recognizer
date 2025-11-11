"""
Audio Data Augmentation for 200-Sentence Dataset

Creates 5 augmented versions of each sentence → 200 × 6 = 1200 total samples
"""

import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import pandas as pd

# Augmentation parameters
NUM_AUGMENTATIONS_PER_FILE = 5  # 200 sentences × 6 total (1 original + 5 augmented) = 1200 samples
SAMPLE_RATE = 16000

# Augmentation ranges
TIME_STRETCH_RANGE = (0.9, 1.1)
PITCH_SHIFT_RANGE = (-2, 2)
NOISE_LEVEL_RANGE = (0.001, 0.005)
VOLUME_RANGE = (0.8, 1.2)


def add_noise(audio, noise_level):
    """Add Gaussian noise to audio"""
    noise = np.random.randn(len(audio)) * noise_level
    return audio + noise


def change_volume(audio, factor):
    """Change volume of audio"""
    return audio * factor


def time_stretch(audio, rate):
    """Time stretch audio"""
    return librosa.effects.time_stretch(audio, rate=rate)


def pitch_shift(audio, sr, n_steps):
    """Pitch shift audio"""
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)


def augment_audio_file(audio_path, output_dir, base_filename, label_text, augmentations_per_file):
    """Create multiple augmented versions of a single audio file"""
    audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

    augmented_files = []

    # First, save original with _aug00 suffix
    original_filename = f"{base_filename}_aug00.wav"
    original_path = output_dir / original_filename
    sf.write(original_path, audio, SAMPLE_RATE)
    augmented_files.append((original_filename, label_text))

    # Create augmented versions
    for aug_idx in range(1, augmentations_per_file + 1):
        augmented = audio.copy()

        # 1. Time stretch (50% chance)
        if np.random.rand() > 0.5:
            rate = np.random.uniform(*TIME_STRETCH_RANGE)
            augmented = time_stretch(augmented, rate)

        # 2. Pitch shift (50% chance)
        if np.random.rand() > 0.5:
            n_steps = np.random.uniform(*PITCH_SHIFT_RANGE)
            augmented = pitch_shift(augmented, SAMPLE_RATE, n_steps)

        # 3. Noise (75% chance)
        if np.random.rand() > 0.25:
            noise_level = np.random.uniform(*NOISE_LEVEL_RANGE)
            augmented = add_noise(augmented, noise_level)

        # 4. Volume change (always apply)
        volume_factor = np.random.uniform(*VOLUME_RANGE)
        augmented = change_volume(augmented, volume_factor)

        # Normalize
        augmented = augmented / (np.max(np.abs(augmented)) + 1e-8)

        # Save
        output_filename = f"{base_filename}_aug{aug_idx:02d}.wav"
        output_path = output_dir / output_filename
        sf.write(output_path, augmented, SAMPLE_RATE)

        augmented_files.append((output_filename, label_text))

    return augmented_files


def main():
    print("="*80)
    print("Audio Data Augmentation for 200-Sentence Dataset")
    print("="*80)

    # Paths
    original_dir = Path("sentences_200")
    augmented_dir = Path("sentences_200_augmented")
    augmented_dir.mkdir(exist_ok=True)

    metadata_file = original_dir / "sentences.csv"

    if not metadata_file.exists():
        print(f"❌ Error: {metadata_file} not found!")
        print("Please wait for generate_200_sentences.py to complete.")
        return

    # Load original metadata
    print(f"\nLoading metadata from {metadata_file}")
    metadata = pd.read_csv(metadata_file)
    print(f"Found {len(metadata)} original sentences")

    # Process each audio file
    print(f"\nCreating {NUM_AUGMENTATIONS_PER_FILE + 1} versions of each (1 original + {NUM_AUGMENTATIONS_PER_FILE} augmented)")
    print(f"Total output: {len(metadata) * (NUM_AUGMENTATIONS_PER_FILE + 1)} audio files")
    print()

    all_augmented_files = []

    for idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Augmenting"):
        filename = row['filename']
        label_text = row['label_text']

        audio_path = original_dir / filename
        if not audio_path.exists():
            print(f"⚠️  Warning: {audio_path} not found, skipping")
            continue

        # Extract base filename (without extension)
        base_filename = Path(filename).stem

        # Augment this file
        augmented_files = augment_audio_file(
            audio_path,
            augmented_dir,
            base_filename,
            label_text,
            NUM_AUGMENTATIONS_PER_FILE
        )

        all_augmented_files.extend(augmented_files)

    # Save augmented metadata
    augmented_metadata = pd.DataFrame(all_augmented_files, columns=['filename', 'label_text'])
    output_csv = augmented_dir / "sentences.csv"
    augmented_metadata.to_csv(output_csv, index=False)

    print(f"\n{'='*80}")
    print(f"✅ Augmentation complete!")
    print(f"✅ Created {len(all_augmented_files)} total audio files")
    print(f"✅ Augmented audio saved to: {augmented_dir}/")
    print(f"✅ Metadata saved to: {output_csv}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
