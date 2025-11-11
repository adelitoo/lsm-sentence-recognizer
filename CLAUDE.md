# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a sentence recognition system using a Spiking Neural Network (SNN) implemented as a Liquid State Machine (LSM). The system converts spoken sentences into spike trains, processes them through an LSM reservoir, and uses Connectionist Temporal Classification (CTC) to decode the output.

**Key Technologies:**
- **snnpy** (snn_reservoir_py): Core SNN/LSM simulation library (CPU-only)
- **PyTorch**: CTC readout layer training (GPU-accelerated)
- **Librosa & Gammatone**: Audio feature extraction
- **Chatterbox TTS**: Text-to-speech audio generation (offline, included in repo)

## Pipeline Architecture

The pipeline is split into **CPU preprocessing** and **GPU training** stages:

```
[CPU STAGE - This Machine]
generate_500_sentences.py → audio_encoding_500.py → extract_lsm_traces_sentence_split_500.py

[GPU STAGE - Main Rig]
train_ctc_traces.py
```

### CPU Stage (Preprocessing)

**1. Audio Generation** (`generate_500_sentences.py`)
   - Generates 500 sentences using Chatterbox TTS
   - **Output**: `sentences_500/sentences.csv` + `sentences_500/*.wav` files (66 MB total)

**2. Spike Encoding** (`audio_encoding_500.py`)
   - Converts audio → Gammatone spectrograms → spike trains
   - Uses hysteresis thresholding with multiple thresholds [0.70, 0.80, 0.90, 0.95]
   - **Output**: `sentence_spike_trains_500.npz` + `sentence_label_map_500.txt`

**3. Train/Test Split** (`create_balanced_sentence_split_500.py`)
   - Creates sentence-level split (80/20) for true generalization testing
   - **Output**: `balanced_sentence_split_500.npz`

**4. LSM Processing** (`extract_lsm_traces_sentence_split_500.py`)
   - Feeds spike trains through LSM reservoir
   - Records continuous membrane voltage traces (not windowed features)
   - **Output**: `lsm_trace_sequences_sentence_split_500.npz` (350 MB)

**4b. LSM Windowed Features** (`extract_lsm_windowed_features_filtered_sentence_split_500.py`) - OPTIONAL
   - Alternative to traces: computes windowed spike counts from LSM output
   - Uses 50ms windows with 10ms stride, applies bandpass filtering
   - **Output**: `lsm_windowed_features_filtered_sentence_split_500.npz` (7 MB)
   - Note: Windowed features are required for char-level and word-level training

### GPU Stage (Training)

You have **THREE training options** depending on your approach:

**5a. Trace-Based Training** (`train_ctc_traces.py`) - RECOMMENDED
   - Trains on continuous membrane voltages (full temporal resolution)
   - Bidirectional GRU (3 layers, 128 hidden)
   - **Input**: `lsm_trace_sequences_sentence_split_500.npz` (350 MB)
   - Best performance, captures fine-grained dynamics

**5b. Character-Level Training** (`train_ctc.py`)
   - Trains on windowed spike features, decodes to characters
   - Bidirectional GRU (2 layers, 32 hidden)
   - **Input**: `lsm_windowed_features_filtered_sentence_split_500.npz` (7 MB)
   - Lighter model, good for character recognition

**5c. Word-Level Training** (`train_ctc_word_level.py`)
   - Trains on windowed spike features, decodes to words
   - Bidirectional GRU (2 layers, 32 hidden)
   - **Input**: `lsm_windowed_features_filtered_sentence_split_500.npz` (7 MB)
   - Vocabulary-based, requires word boundaries

## Deployment Workflow

### On CPU Machine (This Machine)

Run the complete preprocessing pipeline:
```bash
python run_pipeline_500.py --multiplier 0.8
```

This generates all necessary data files. Then push to GitHub:
```bash
git add .
git commit -m "Update preprocessed data"
git push
```

### On GPU Machine (Main Rig)

Pull the repo and train with your preferred approach:

```bash
git pull

# Option 1: Trace-based training (recommended, best accuracy)
python train_ctc_traces.py

# Option 2: Character-level training (lighter, decodes to chars)
python train_ctc.py

# Option 3: Word-level training (vocabulary-based)
python train_ctc_word_level.py
```

All training scripts automatically use CUDA if available.

**Note**: Character and word-level training require `lsm_windowed_features_filtered_sentence_split_500.npz`. If this file is not in the repo, generate it on the CPU machine first:
```bash
python extract_lsm_windowed_features_filtered_sentence_split_500.py --multiplier 0.8
```

## Key Hyperparameters

**LSM Configuration** (`extract_lsm_traces_sentence_split_500.py`):
- 1000 reservoir neurons, 700 output neurons
- Small-world topology (p=0.1, k=200)
- Critical weight multiplier: 0.8 (tunable via `--multiplier`)
- Records continuous membrane voltages at 2000 timesteps

**Audio Encoding** (`audio_encoding_500.py`):
- Duration: 10 seconds per sentence
- Gammatone filterbank: 128 filters
- Time bins: 500 → 2000 after multi-threshold encoding
- Spike thresholds: [0.70, 0.80, 0.90, 0.95] with hysteresis (gap=0.1)

**CTC Models**:

*Trace-based (`train_ctc_traces.py`)*:
- Bidirectional 3-layer GRU (hidden_size=128, dropout=0.2)
- Input: (batch, 2000 timesteps, 700 membrane channels)
- Character vocabulary: space + a-z + apostrophe (29 classes with blank)
- 5000 epochs, Adam (lr=0.0005, weight_decay=1e-5)

*Character-level (`train_ctc.py`)*:
- Bidirectional 2-layer GRU (hidden_size=32)
- Input: (batch, ~200 windows, num_features)
- Character vocabulary: space + a-z + apostrophe (29 classes with blank)
- 1000 epochs, Adam (lr=0.0005)

*Word-level (`train_ctc_word_level.py`)*:
- Bidirectional 2-layer GRU (hidden_size=32)
- Input: (batch, ~200 windows, num_features)
- Word vocabulary: ~200-300 unique words (varies by dataset)
- 1000 epochs, Adam (lr=0.0005)

## Data Files

**Essential for Trace-Based Training** (checked into repo):
- `lsm_trace_sequences_sentence_split_500.npz` - LSM membrane traces (350 MB)
- `sentence_label_map_500.txt` - Text labels for each sentence
- `balanced_sentence_split_500.npz` - Train/test split indices

**Essential for Char/Word Training** (generate with step 4b):
- `lsm_windowed_features_filtered_sentence_split_500.npz` - Windowed spike features (7 MB)

**Intermediate Files** (can regenerate, but included for convenience):
- `sentences_500/*.wav` - Audio files (66 MB)
- `sentence_spike_trains_500.npz` - Spike train encoding (1 MB)

## Individual Pipeline Steps

If you need to run steps individually:

```bash
# 1. Generate audio (if not using included audio)
python generate_500_sentences.py

# 2. Encode audio to spike trains
python audio_encoding_500.py --n-filters 128 --filterbank gammatone

# 3. Create balanced sentence split
python create_balanced_sentence_split_500.py

# 4a. Extract LSM traces (for trace-based training)
python extract_lsm_traces_sentence_split_500.py --multiplier 0.8

# 4b. Extract LSM windowed features (for char/word training)
python extract_lsm_windowed_features_filtered_sentence_split_500.py --multiplier 0.8

# 5. Train CTC model (on GPU machine) - choose one:
python train_ctc_traces.py         # Trace-based (recommended)
python train_ctc.py                 # Character-level
python train_ctc_word_level.py      # Word-level
```

## Architecture Notes

### Three Approaches Explained

**1. Trace-Based (Recommended)**
- Uses continuous membrane voltages from 700 LSM output neurons
- Full temporal resolution (2000 timesteps)
- No information loss from windowing
- Larger model (3-layer GRU, 128 hidden) to handle 700 channels
- Best accuracy but requires 350 MB data file

**2. Character-Level**
- Uses windowed spike counts (50ms windows, 10ms stride)
- Applies bandpass filtering to extract temporal features
- Smaller model (2-layer GRU, 32 hidden)
- Decodes directly to characters (space + a-z + apostrophe)
- Lightweight (7 MB data file)

**3. Word-Level**
- Same windowed features as character-level
- Builds vocabulary from training sentences (~200-300 words)
- Decodes to word tokens instead of characters
- Requires word boundaries but can be more accurate for vocabulary words
- Also lightweight (7 MB data file)

### LSM Critical Weight

The critical weight (w_critico) represents the boundary between subcritical and supercritical regimes:
- **Subcritical** (multiplier < 1.0): Dampened dynamics, less separability
- **Supercritical** (multiplier > 1.0): Chaotic dynamics, potential instability
- **Optimal range**: Typically 0.6-1.0 for this task

Theoretical calculation: `w_critico = (threshold - 2*I*refractory) / beta`

### CTC Character Mapping

- Blank token (index 0) allows variable-length outputs
- Characters: space + a-z + apostrophe (28 characters)
- Greedy decoding: argmax → collapse repeats → remove blanks

## Requirements

**CPU Stage**:
```
numpy
librosa
gammatone
snnpy (snn_reservoir_py)
```

**GPU Stage**:
```
numpy
torch (with CUDA support)
matplotlib
```

Install all with: `pip install -r requirements.txt`
- Stop generating .md files unless asked