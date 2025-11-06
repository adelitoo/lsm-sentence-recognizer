"""
Download and prepare LibriSpeech dataset for your LSM+CTC pipeline
Start with the small dev-clean subset (5.4 hours, ~2700 sentences)
"""
import os
import tarfile
import urllib.request
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import librosa
import soundfile as sf

# Configuration
LIBRISPEECH_URL = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
DOWNLOAD_DIR = Path("librispeech_download")
OUTPUT_DIR = Path("sentences")  # Same as your original structure
MAX_DURATION = 10.0  # Filter out very long sentences
MIN_DURATION = 2.0   # Filter out very short sentences
MAX_SAMPLES = 500    # Limit total samples for initial testing

def download_librispeech():
    """Download LibriSpeech dev-clean subset"""
    DOWNLOAD_DIR.mkdir(exist_ok=True)
    tar_path = DOWNLOAD_DIR / "dev-clean.tar.gz"
    
    if tar_path.exists():
        print(f"✅ Archive already downloaded: {tar_path}")
    else:
        print("="*60)
        print("Downloading LibriSpeech dev-clean (343 MB)...")
        print("This may take a few minutes...")
        print("="*60)
        
        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded / total_size * 100, 100)
            bar_length = 40
            filled = int(bar_length * downloaded / total_size)
            bar = '█' * filled + '-' * (bar_length - filled)
            print(f'\r[{bar}] {percent:.1f}%', end='', flush=True)
        
        urllib.request.urlretrieve(LIBRISPEECH_URL, tar_path, progress_hook)
        print("\n✅ Download complete!")
    
    # Extract
    extract_dir = DOWNLOAD_DIR / "LibriSpeech"
    if extract_dir.exists():
        print(f"✅ Already extracted to: {extract_dir}")
    else:
        print("\nExtracting archive...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(DOWNLOAD_DIR)
        print("✅ Extraction complete!")
    
    return extract_dir / "dev-clean"

def parse_transcript_file(trans_file):
    """Parse LibriSpeech transcript file"""
    transcripts = {}
    with open(trans_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                file_id, text = parts
                transcripts[file_id] = text.lower()
    return transcripts

def collect_audio_files(dataset_dir):
    """Collect all audio files and their transcripts"""
    print("\n" + "="*60)
    print("Collecting audio files and transcripts...")
    print("="*60)
    
    samples = []
    
    # LibriSpeech structure: speaker_id/chapter_id/*.flac
    for speaker_dir in tqdm(list(dataset_dir.iterdir()), desc="Processing speakers"):
        if not speaker_dir.is_dir():
            continue
            
        for chapter_dir in speaker_dir.iterdir():
            if not chapter_dir.is_dir():
                continue
            
            # Find transcript file
            trans_file = chapter_dir / f"{speaker_dir.name}-{chapter_dir.name}.trans.txt"
            if not trans_file.exists():
                continue
            
            # Parse transcripts
            transcripts = parse_transcript_file(trans_file)
            
            # Collect audio files
            for flac_file in chapter_dir.glob("*.flac"):
                file_id = flac_file.stem
                if file_id in transcripts:
                    samples.append({
                        'audio_path': flac_file,
                        'text': transcripts[file_id],
                        'speaker_id': speaker_dir.name,
                        'chapter_id': chapter_dir.name
                    })
    
    print(f"\n✅ Found {len(samples)} audio samples")
    return samples

def filter_by_duration(samples, min_dur, max_dur):
    """Filter samples by duration"""
    print(f"\nFiltering samples ({min_dur}s - {max_dur}s)...")
    filtered = []
    
    for sample in tqdm(samples, desc="Checking durations"):
        try:
            duration = librosa.get_duration(path=sample['audio_path'])
            if min_dur <= duration <= max_dur:
                sample['duration'] = duration
                filtered.append(sample)
        except Exception as e:
            print(f"Warning: Could not process {sample['audio_path']}: {e}")
    
    print(f"✅ Kept {len(filtered)}/{len(samples)} samples")
    return filtered

def convert_to_wav(samples, output_dir, max_samples):
    """Convert FLAC to WAV and create metadata CSV"""
    output_dir.mkdir(exist_ok=True)
    
    # Limit samples
    if len(samples) > max_samples:
        print(f"\nLimiting to {max_samples} samples for faster experimentation")
        samples = samples[:max_samples]
    
    metadata = []
    
    print(f"\n" + "="*60)
    print(f"Converting {len(samples)} files to WAV format...")
    print("="*60)
    
    for i, sample in enumerate(tqdm(samples, desc="Converting")):
        try:
            # Load audio
            audio, sr = librosa.load(sample['audio_path'], sr=16000, mono=True)
            
            # Save as WAV
            output_filename = f"libri_{i:05d}.wav"
            output_path = output_dir / output_filename
            sf.write(output_path, audio, sr)
            
            # Add to metadata
            metadata.append({
                'filename': output_filename,
                'label_text': sample['text'],
                'duration': sample['duration'],
                'speaker_id': sample['speaker_id']
            })
            
        except Exception as e:
            print(f"\nError processing {sample['audio_path']}: {e}")
    
    # Save metadata CSV
    df = pd.DataFrame(metadata)
    csv_path = output_dir / "sentences.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"\n✅ Saved {len(metadata)} audio files to: {output_dir}")
    print(f"✅ Saved metadata to: {csv_path}")
    
    # Print statistics
    print("\n" + "="*60)
    print("Dataset Statistics:")
    print("="*60)
    print(f"Total samples: {len(metadata)}")
    print(f"Unique sentences: {df['label_text'].nunique()}")
    print(f"Unique speakers: {df['speaker_id'].nunique()}")
    print(f"Duration range: {df['duration'].min():.1f}s - {df['duration'].max():.1f}s")
    print(f"Average duration: {df['duration'].mean():.1f}s")
    print(f"Average sentence length: {df['label_text'].str.len().mean():.0f} chars")
    print("="*60)
    
    # Show example sentences
    print("\nExample sentences:")
    for i in range(min(5, len(df))):
        print(f"  {i+1}. \"{df.iloc[i]['label_text']}\" ({df.iloc[i]['duration']:.1f}s)")

def main():
    print("="*60)
    print("LibriSpeech Dataset Preparation")
    print("="*60)
    
    # Step 1: Download
    dataset_dir = download_librispeech()
    
    # Step 2: Collect files
    samples = collect_audio_files(dataset_dir)
    
    # Step 3: Filter by duration
    samples = filter_by_duration(samples, MIN_DURATION, MAX_DURATION)
    
    # Step 4: Convert and save
    convert_to_wav(samples, OUTPUT_DIR, MAX_SAMPLES)
    
    print("\n" + "="*60)
    print("✅ Dataset preparation complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run: python audio_encoding.py --n-filters 100")
    print("2. Run: python extract_lsm_sequences.py --multiplier 1.2")
    print("3. Run: python train_ctc.py")
    print("="*60)

if __name__ == "__main__":
    main()