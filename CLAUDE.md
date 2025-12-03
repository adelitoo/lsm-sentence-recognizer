# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LSM Sentence Recognizer is a speech recognition system that uses **Liquid State Machines (LSM)** as a spiking neural network reservoir to process audio and a **CTC (Connectionist Temporal Classification)** readout layer to decode text. The pipeline converts audio to spike trains, processes them through a recurrent spiking reservoir, and trains a GRU-based readout to perform character-level speech recognition.

## Common Commands

### Full Pipeline
```bash
# Run complete pipeline (encoding → split → trace extraction)
python run_pipeline.py --multiplier 1.0 --leak 0.1

# Skip steps if outputs already exist
python run_pipeline.py --skip-encoding --skip-split
```

### Individual Steps
```bash
# Step 1: Generate sentence audio (requires chatterbox-tts)
python generate_sentences.py

# Step 2: Convert audio to spike trains
python audio_encoding.py --filterbank gammatone --n-filters 128

# Step 3: Create train/test split by sentence
python create_balanced_sentence_split.py

# Step 4: Extract LSM membrane potential traces
python extract_lsm_traces.py --multiplier 0.4 --leak 0.001 --leak-variance-divisor 20

# Step 5: Train CTC readout model (multiple training variants available)
python train_ctc_traces.py              # Standard 3-layer BiGRU readout
python train_ctc_traces_linear.py       # Purist linear readout (avg pool + linear)
python train_ctc.py                     # Alternative training approach
```

### Diagnostics
```bash
# Analyze LSM feature separability (generates diagnostic plots)
python diagnose_lsm_separability.py
```

## Architecture

### Data Flow
```
Audio (.wav) → Gammatone Filterbank → Spike Trains → LSM Reservoir → Membrane Traces → CTC Readout → Text
```

### Key Components

1. **Audio Encoding** (`audio_encoding.py`)
   - Gammatone/Mel filterbank → multi-threshold hysteresis encoding
   - Output: `sentence_spike_trains.npz` (shape: samples × 128 neurons × 2000 timesteps)

2. **LSM Reservoir** (`extract_lsm_traces.py`)
   - Uses `snnpy.snn.Reservoir` from `snn_reservoir_py` package
   - 2000 neurons, 700 output neurons, small-world connectivity
   - Critical parameters: `--multiplier` (weight scaling), `--leak` (membrane decay)
   - Heterogeneous leak factors for temporal diversity
   - Output: `lsm_trace_sequences.npz` (membrane potentials over time)

3. **CTC Training** (multiple readout variants)
   - `train_ctc_traces.py`: 3-layer bidirectional GRU (128 hidden units)
   - `train_ctc_traces_linear.py`: Purist linear readout using BPE tokenizer (vocab_size=100, stride=10 for temporal downsampling)
   - Character vocabulary: space + a-z + apostrophe (29 classes including blank)
   - Uses sentence-level train/test split for true generalization testing

### Key Parameters to Tune

| Parameter | Location | Description |
|-----------|----------|-------------|
| `--multiplier` | extract_lsm_traces.py | Weight scaling relative to critical weight (try 0.4-0.9) |
| `--leak` | extract_lsm_traces.py | Membrane leak coefficient (lower = longer memory) |
| `--leak-variance-divisor` | extract_lsm_traces.py | Heterogeneous leak factor spread |
| `NUM_NEURONS` | extract_lsm_traces.py | Reservoir size (default: 2000) |
| `NUM_OUTPUT_NEURONS` | extract_lsm_traces.py | Output readout neurons (default: 700) |

### Output Files

- `sentence_spike_trains.npz` - Encoded spike trains (samples × 128 neurons × 2000 timesteps)
- `sentence_label_map.txt` - Sentence ID → text mapping
- `balanced_sentence_split.npz` - Train/test sentence IDs
- `lsm_trace_sequences.npz` - LSM membrane potential traces (samples × 2000 timesteps × 700 neurons)
- `ctc_model_traces.pt` - Trained PyTorch model (GRU variant)
- `ctc_model_sentence_split.pt` - Alternative trained model
- `lsm_raster_plot.png` - Spike activity visualization

## Important Notes

- The split is **sentence-level**: all augmented versions of a sentence stay in train or test together
- Check `diagnose_lsm_separability.py` output if accuracy is low - it identifies LSM tuning issues
- Target network activity: 1-50% of reservoir neurons active (check debugging output)
- The `snn_reservoir_py` package provides the core LSM implementation
- Audio data augmentation is controlled by `NUM_AUGMENTATIONS` in `audio_encoding.py` (default: 1, no augmentation)
- Dataset generation uses 35-word vocabulary with full alphabet coverage (a-z)
- `generate_sentences.py` uses `chatterbox-tts` for local TTS generation (requires: `pip install chatterbox-tts`)

## Environment Setup

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Linux/Mac

# Install dependencies
pip install -r requirements.txt

# For sentence generation (if not already installed)
pip install chatterbox-tts
```
