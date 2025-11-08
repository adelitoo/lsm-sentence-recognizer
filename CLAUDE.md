# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a sentence recognition system using a Spiking Neural Network (SNN) implemented as a Liquid State Machine (LSM). The system converts spoken sentences into spike trains, processes them through an LSM reservoir, and uses Connectionist Temporal Classification (CTC) to decode the output.

**Key Technologies:**
- **snnpy** (snn_reservoir_py): Core SNN/LSM simulation library
- **PyTorch**: CTC readout layer training
- **Librosa & Gammatone**: Audio feature extraction
- **ElevenLabs API**: Text-to-speech audio generation

## Pipeline Architecture

The project is structured as a **sequential pipeline** where each script produces outputs consumed by the next:

```
generate_sentences.py → audio_encoding.py → extract_lsm_sequences.py → train_ctc.py
```

### Pipeline Flow

1. **Audio Generation** (`generate_sentences.py`)
   - Generates `.mp3` files from sentences using ElevenLabs API
   - **Output**: `sentences/sentences.csv` (manifest) and `sentences/*.mp3` files
   - **Note**: ElevenLabs API key is currently hardcoded (line 5)

2. **Spike Encoding** (`audio_encoding.py`)
   - Converts audio → spectrograms (Mel or Gammatone) → spike trains
   - Uses hysteresis thresholding with multiple thresholds
   - **Output**: `sentence_spike_trains.npz` and `sentence_label_map.txt`

3. **LSM Processing** (`extract_lsm_sequences.py`)
   - Feeds spike trains through LSM reservoir
   - Records analog trace (voltage) history of output neurons
   - Calculates theoretical critical weight (w_critico) and applies multiplier
   - **Output**: `lsm_trace_sequences.npz`

4. **CTC Training** (`train_ctc.py`)
   - Trains bidirectional GRU + linear readout layer
   - Uses PyTorch CTCLoss to learn text from LSM traces
   - **Output**: Console training progress and `lsm_trace_visualization.png`

### Key Hyperparameters

**LSM Configuration** (`extract_lsm_sequences.py`):
- 1000 reservoir neurons, 400 output neurons
- Small-world topology (p=0.1, k=200)
- Critical weight calculation: `w_critico = (threshold - 2*I*refractory) / beta`
- Weight multiplier: tunable via `--multiplier` (default: 1.0)

**Audio Encoding** (`audio_encoding.py`):
- Duration: 10 seconds (DURATION constant)
- Time bins: 500 → 2000 after multi-threshold encoding
- Spike thresholds: [0.70, 0.80, 0.90, 0.95] with hysteresis

**CTC Model** (`train_ctc.py`):
- Bidirectional 2-layer GRU (hidden_size=32)
- Character vocabulary: space + a-z + apostrophe (29 classes with blank)
- 1000 epochs, Adam optimizer (lr=0.0005)

## Common Commands

### Full Pipeline

Run the entire pipeline automatically:
```bash
./run_pipeline.sh
```

This script checks for dataset preparation, then runs audio encoding → LSM extraction → CTC training sequentially.

### Individual Pipeline Steps

```bash
# 1. Generate audio data (requires ElevenLabs API key)
python generate_sentences.py

# 2. Encode audio to spike trains
python audio_encoding.py --n-filters 128 --filterbank gammatone

# Or with Mel filters:
python audio_encoding.py --filterbank mel --n-filters 128

# 3. Extract LSM reservoir traces
python extract_lsm_sequences.py --multiplier 0.6

# 4. Train CTC readout layer
python train_ctc.py
```

### Debugging Audio Encoding

Visualize hysteresis logic for a specific audio file:
```bash
python audio_encoding.py --debug-hysteresis sentences/sentence_1.mp3 --debug-channel 50
```

## Data Flow & File Dependencies

- `sentences/sentences.csv` → Required by `audio_encoding.py`
- `sentence_spike_trains.npz` + `sentence_label_map.txt` → Required by `extract_lsm_sequences.py` and `train_ctc.py`
- `lsm_trace_sequences.npz` → Required by `train_ctc.py`

**Important**: Data passes through `.npz` files. Each script expects the previous step's output file to exist.

## Architecture Notes

### LSM Reservoir Dynamics

The critical weight (w_critico) represents the boundary between subcritical and supercritical regimes:
- **Subcritical** (multiplier < 1.0): Dampened dynamics, less separability
- **Supercritical** (multiplier > 1.0): Chaotic dynamics, potential instability
- **Optimal range**: Typically 0.6-1.0 for this task

The theoretical calculation accounts for:
- Average input spike rate (I)
- Network connectivity (β = k/2)
- Membrane threshold and refractory period

### CTC Decoding

The CTC model learns alignment-free mappings:
- Blank token (index 0) allows variable-length outputs
- Greedy decoding: argmax → collapse repeats → remove blanks
- Input: LSM trace sequences (time × features)
- Output: Character sequences

### Spike Encoding Strategy

Multi-threshold hysteresis creates temporal diversity:
- Each threshold generates independent spike channels
- Hysteresis (gap=0.1) prevents flickering at boundaries
- Output interleaves thresholds: [T1_t0, T2_t0, T3_t0, T1_t1, ...]

## Additional Files

- `singleword_extraction.py`: Alternative feature extraction (not integrated into main pipeline)
- `GEMINI.md`: Previous documentation (may contain outdated information)
