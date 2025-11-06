import librosa
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import zoom
from tqdm import tqdm
import warnings
import argparse
from gammatone import gtgram
import sys
import pandas as pd # <-- Make sure this is included

# --- CRITICAL: SET THIS TO YOUR MAX SENTENCE LENGTH ---
DURATION = 10.0 # e.g., 8.0 seconds. 1.0 is too short.
# ---

SAMPLE_RATE = 16000
TIME_BINS = 500 # This will be 400 after thresholding
SPIKE_THRESHOLDS = [0.70, 0.80, 0.90, 0.95]
HYSTERESIS_GAP = 0.1
MAX_SAMPLES_PER_CLASS = 1000 # Set to a high number
VISUALIZE_FIRST_SAMPLE = False
REDUNDANCY_FACTOR = 1 # Keep this at 1

np.random.seed(42)

def load_audio_file(filepath: Path) -> np.ndarray | None:
    """Load audio file"""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Load with the global DURATION
            audio, _ = librosa.load(filepath, sr=SAMPLE_RATE, duration=DURATION, mono=True)
        
        # Pad or truncate to the exact DURATION
        target_length = int(SAMPLE_RATE * DURATION)
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        else:
            audio = audio[:target_length]
        return audio
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def audio_to_mel_spectrogram(audio: np.ndarray, n_mels: int) -> np.ndarray:
    """Convert audio to mel spectrogram"""
    hop_length = max(1, int(len(audio) / TIME_BINS))
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE, n_mels=n_mels, hop_length=hop_length
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_min = mel_spec_db.min()
    mel_max = mel_spec_db.max()
    if (mel_max - mel_min) < 1e-8:
        return np.zeros((n_mels, TIME_BINS), dtype=np.float32)
    mel_spec_norm = (mel_spec_db - mel_min) / (mel_max - mel_min + 1e-8)
    if mel_spec_norm.shape[1] != TIME_BINS:
        try:
            zoom_factor = TIME_BINS / mel_spec_norm.shape[1]
            mel_spec_norm = zoom(mel_spec_norm, (1, zoom_factor), order=1)
        except ValueError as e:
            print(f"Warning: Zoom failed (shape {mel_spec_norm.shape}, factor {zoom_factor}): {e}")
            return np.zeros((n_mels, TIME_BINS), dtype=np.float32)
    return mel_spec_norm[:, :TIME_BINS]

def audio_to_gammatone_spectrogram(audio: np.ndarray, n_filters: int) -> np.ndarray:
    """Convert audio to gammatone spectrogram"""
    hop_time = len(audio) / (SAMPLE_RATE * TIME_BINS)
    # Ensure hop_time is valid
    if hop_time <= 0:
        hop_time = 0.01 # A small default if audio is empty
        
    gtg = gtgram.gtgram(
        wave=audio,
        fs=SAMPLE_RATE,
        window_time=0.025,
        hop_time=hop_time,
        channels=n_filters,
        f_min=50
    )
    gtg_db = 20 * np.log10(gtg + 1e-9) # Add epsilon to avoid log(0)
    gtg_db = np.maximum(gtg_db, gtg_db.max() - 80.0) # Clip to 80 dB dynamic range
    gtg_min = gtg_db.min()
    gtg_max = gtg_db.max()
    if (gtg_max - gtg_min) < 1e-8:
        return np.zeros((n_filters, TIME_BINS), dtype=np.float32)
    gtg_norm = (gtg_db - gtg_min) / (gtg_max - gtg_min + 1e-8)
    if gtg_norm.shape[1] != TIME_BINS:
        try:
            zoom_factor = TIME_BINS / gtg_norm.shape[1]
            gtg_norm = zoom(gtg_norm, (1, zoom_factor), order=1)
        except ValueError as e:
            print(f"Warning: Zoom failed (shape {gtg_norm.shape}, factor {zoom_factor}): {e}")
            return np.zeros((n_filters, TIME_BINS), dtype=np.float32)
    return gtg_norm[:, :TIME_BINS]


def convert_mels_to_spikes_hysteresis(mel_spec, thresholds, hysteresis_gap=0.05):
    """
    Converts mel spectrogram to spikes using hysteresis.
    """
    n_mels, n_time = mel_spec.shape
    n_thresholds = len(thresholds)
    # The output shape is expanded in time
    spikes = np.zeros((n_mels, n_time * n_thresholds), dtype=np.uint8)
    
    for t_idx, threshold in enumerate(sorted(thresholds, reverse=True)):
        active = np.zeros(n_mels, dtype=bool)
        lower_bound = threshold - hysteresis_gap
        
        for time_bin in range(n_time):
            rising = (mel_spec[:, time_bin] > threshold) & ~active
            falling = (mel_spec[:, time_bin] < lower_bound) & active
            
            active[rising] = True
            active[falling] = False
            
            output_time = time_bin + (n_time * t_idx) # This logic was wrong before, fixed
            
            # This logic interleaves: [T1_t0, T2_t0, T3_t0, T1_t1, T2_t1, ...]
            output_time_interleaved = time_bin * n_thresholds + t_idx
            
            if output_time_interleaved < spikes.shape[1]:
                spikes[:, output_time_interleaved] = active.astype(np.uint8)
                
    return spikes

def create_pure_redundancy(spike_train: np.ndarray, redundancy_factor: int) -> np.ndarray:
    """Create pure redundancy by simply repeating each neuron's spike train."""
    if redundancy_factor <= 1:
        return spike_train
    redundant = np.repeat(spike_train, redundancy_factor, axis=0)
    return redundant

def visualize_conversion(mel, base_spikes, redundant_spikes, filename, n_filters):
    """Visualize the conversion with pure redundancy"""
    # (This function is unchanged, but you can copy it from your old file if needed)
    pass

def visualize_hysteresis_channel(spectrogram, channel_index, thresholds, hysteresis_gap, filename):
    """Generates a detailed plot to debug the hysteresis logic for a single channel."""
    # (This function is unchanged, but you can copy it from your old file if needed)
    pass

def create_dataset(n_filters: int, filterbank: str):
    """Create dataset from a metadata CSV for sentences."""
    
    BASE_DATASET_PATH = Path("sentences")
    METADATA_FILE = BASE_DATASET_PATH / "sentences.csv"
    
    all_spike_trains = []
    all_labels = []
    all_spike_counts = []

    print("="*60)
    print("CREATING SENTENCE DATASET (from CSV)")
    print("="*60)
    
    # --- Load Metadata ---
    if not METADATA_FILE.exists():
        print(f"ERROR: Metadata file not found.")
        print(f"Please run 'generate_sentences.py' first to create: {METADATA_FILE.resolve()}")
        print("="*60)
        return

    print(f"Loading metadata from '{METADATA_FILE}'")
    try:
        metadata = pd.read_csv(METADATA_FILE)
        if 'filename' not in metadata.columns or 'label_text' not in metadata.columns:
             raise ValueError("CSV must have 'filename' and 'label_text' columns.")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
        
    # --- Create Numeric Labels for Unique Sentences ---
    unique_labels = metadata['label_text'].unique()
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    print(f"Found {len(metadata)} total samples across {len(unique_labels)} unique sentences.")
    
    # --- Print Configuration ---
    print("\n" + "="*60)
    print("Configuration:")
    print(f"  Filterbank: {filterbank}")
    print(f"  Filters: {n_filters}")
    print(f"  Audio Duration: {DURATION}s")
    print(f"  Base Time bins: {TIME_BINS}")
    print(f"  Thresholds: {len(SPIKE_THRESHOLDS)}")
    print(f"  => Final Time Bins: {TIME_BINS * len(SPIKE_THRESHOLDS)}")
    print(f"  Redundancy factor: {REDUNDANCY_FACTOR}x")
    print(f"  Input neurons per sample: {n_filters * REDUNDANCY_FACTOR}")
    print(f"  Encoding: Hysteresis (Gap: {HYSTERESIS_GAP})")
    print("="*60 + "\n")

    # --- Process Files from CSV ---
    file_list = metadata.to_dict('records')
    for row in tqdm(file_list, desc="Processing audio files"):
        filename = row['filename']
        label_text = row['label_text']
        label_idx = label_to_id[label_text]
        
        audio_file = BASE_DATASET_PATH / filename
        
        if not audio_file.exists():
            print(f"  Warning: File not found, skipping: {audio_file}")
            continue

        audio_data = load_audio_file(audio_file)
        if audio_data is None:
            continue
        
        if filterbank == 'mel':
            spectrogram = audio_to_mel_spectrogram(audio_data, n_filters)
        else:
            spectrogram = audio_to_gammatone_spectrogram(audio_data, n_filters)

        base_spike_train = convert_mels_to_spikes_hysteresis(
            spectrogram,
            SPIKE_THRESHOLDS,
            HYSTERESIS_GAP
        )
        
        redundant_spike_train = create_pure_redundancy(base_spike_train, REDUNDANCY_FACTOR)
        
        num_spikes = np.sum(redundant_spike_train)
        all_spike_counts.append(num_spikes)
        all_spike_trains.append(redundant_spike_train)
        all_labels.append(label_idx)

        if VISUALIZE_FIRST_SAMPLE and len(all_spike_trains) == 1:
            visualize_conversion(spectrogram, base_spike_train,
                               redundant_spike_train, audio_file.name, n_filters)

    if not all_spike_trains:
        print("\n" + "="*60)
        print("ERROR: No audio files were successfully processed.")
        print(f"Please check your 'sentences.csv' file and audio paths.")
        print("="*60)
        return
        
    X_spikes = np.array(all_spike_trains, dtype=np.uint8)
    y_labels = np.array(all_labels, dtype=np.int32)
    
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    print(f"Total samples: {len(X_spikes)}")
    print(f"Unique classes (sentences): {len(unique_labels)}")
    print(f"Shape: {X_spikes.shape}")
    print(f"  Input neurons: {X_spikes.shape[1]} ({n_filters} filters × {REDUNDANCY_FACTOR} copies)")
    print(f"  Time bins: {X_spikes.shape[2]} ({TIME_BINS} base × {len(SPIKE_THRESHOLDS)} thresholds)")
    print(f"Spike statistics:")
    print(f"  Avg per sample: {np.mean(all_spike_counts):.1f}")
    print(f"  Std: {np.std(all_spike_counts):.1f}")
    print(f"  Min/Max: {np.min(all_spike_counts)} / {np.max(all_spike_counts)}")
    print("="*60)
    
    output_filename = "sentence_spike_trains.npz"
    np.savez_compressed(output_filename, X_spikes=X_spikes, y_labels=y_labels)
    
    # Save label mapping
    label_map_filename = "sentence_label_map.txt"
    with open(label_map_filename, "w", encoding="utf-8") as f:
        f.write("label_id,label_text\n")
        for label, idx in label_to_id.items():
            f.write(f"{idx},{label}\n")
            
    print(f"\n✅ Saved dataset to '{output_filename}'")
    print(f"✅ Saved label map to '{label_map_filename}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a spike train dataset from audio files.")
    parser.add_argument("--n-filters", type=int, default=128,
                        help="Number of filters to use in the filterbank (default: 128).")
    parser.add_argument("--filterbank", type=str, default="gammatone", choices=["mel", "gammatone"],
                        help="Type of filterbank to use (default: gammatone).")
    parser.add_argument("--debug-hysteresis", type=str, default=None,
                        help="Path to a .wav file to debug. Runs visualization and exits.")
    parser.add_argument("--debug-channel", type=int, default=50,
                        help="The filter channel to visualize for the debug plot (default: 50).")

    args = parser.parse_args()

    if args.debug_hysteresis:
        print("--- Running in Hysteresis Debug Mode ---")
        audio_file = Path(args.debug_hysteresis)
        if not audio_file.exists():
            print(f"Error: Debug file not found at '{audio_file}'")
            sys.exit(1)

        # Load and process a single file
        audio_data = load_audio_file(audio_file)
        if audio_data is not None:
            print(f"Generating spectrogram for '{audio_file.name}'...")
            if args.filterbank == 'mel':
                spectrogram = audio_to_mel_spectrogram(audio_data, args.n_filters)
            else:
                spectrogram = audio_to_gammatone_spectrogram(audio_data, args.n_filters)
            
            # Check if channel index is valid
            if args.debug_channel >= spectrogram.shape[0]:
                print(f"Error: --debug-channel ({args.debug_channel}) is out of bounds. "
                      f"Max channel is {spectrogram.shape[0] - 1}.")
                sys.exit(1)

            print(f"Visualizing channel {args.debug_channel}...")
            visualize_hysteresis_channel(
                spectrogram,
                args.debug_channel,
                SPIKE_THRESHOLDS,
                HYSTERESIS_GAP,
                audio_file.name
            )
        else:
            print("Could not load audio file.")
    else:
        # Run the full dataset creation
        create_dataset(n_filters=args.n_filters, filterbank=args.filterbank)