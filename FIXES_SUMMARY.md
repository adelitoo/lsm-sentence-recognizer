# LSM Sentence Recognition - Problem Analysis & Fixes

## Problems Identified

### 1. **CRITICAL: Excessive Audio Padding (83% silence)**
**Problem**: Audio files are ~1.7 seconds long, but `DURATION` was set to 10 seconds
- This created 8.3s of silence/padding per sample
- 83% of the input contained no information
- CTC struggles to learn temporal alignment with this much padding

**Fix**: Changed `DURATION` from 10.0 to 3.0 seconds in `audio_encoding.py`
- Now only ~43% padding (much more reasonable)
- CTC can focus on actual audio content

### 2. **Feature Representation Mismatch**
**Problem**: Single words used aggregated features, sentences used raw traces
- **Single words**: Used `extract_features_from_spikes()` → worked well
  - Features: spike_counts, spike_variances, mean_spike_times, ISI, etc.
  - These have good discriminative power
- **Sentences**: Used `get_trace_history_output()` → didn't work
  - Raw analog traces lack separability
  - CTC couldn't distinguish between different phonemes/characters

**Solution**: Created hybrid approach with **windowed feature extraction**
- Extract the same features that worked for single words
- But do it in sliding time windows (window_size=40, stride=20)
- Preserves temporal structure (needed for CTC) + discriminative power

### 3. **CTC Training Configuration**
**Problems**:
- No learning rate scheduling
- No gradient clipping
- Fixed learning rate might be suboptimal

**Fixes** in `train_ctc.py`:
- Added learning rate scheduler (ReduceLROnPlateau)
- Added gradient clipping (max_norm=1.0)
- Increased initial LR from 0.0005 to 0.001
- Better progress tracking with best loss

## New Files Created

### 1. `extract_lsm_windowed_features.py`
Replaces `extract_lsm_sequences.py` for sentence recognition.

**Key features**:
- Sliding window extraction (configurable window_size and stride)
- Uses same features as single-word pipeline
- Output: `lsm_windowed_features.npz`

**Usage**:
```bash
python extract_lsm_windowed_features.py --multiplier 0.6 --window-size 40 --stride 20
```

### 2. `analyze_lsm_features.py`
Diagnostic tool to check feature quality.

**What it checks**:
- Feature separability (PCA visualization)
- Temporal structure
- Feature statistics (NaN, zero features, variance)

**Usage**:
```bash
python analyze_lsm_features.py
```

**Output**:
- `temporal_structure_analysis.png` - Shows how features change over time
- `feature_separability_pca.png` - Shows if different sentences are separable

## Modified Files

### 1. `audio_encoding.py`
- **Line 16**: Changed `DURATION = 10.0` → `DURATION = 3.0`
- Added warning comment about padding impact on CTC

### 2. `train_ctc.py`
- **Lines 100-114**: Auto-detect windowed features vs traces
- **Lines 166-169**: Added learning rate scheduler
- **Lines 220**: Added gradient clipping
- **Lines 224**: Added scheduler step
- **Lines 232-233**: Enhanced progress reporting

### 3. `run_pipeline.sh`
- Updated to use `extract_lsm_windowed_features.py` instead of `extract_lsm_sequences.py`
- Added feature analysis step

## How to Run

### Option 1: Full Pipeline (Recommended)
```bash
# This will re-encode audio with correct duration, extract windowed features, and train
./run_pipeline.sh
```

### Option 2: Step-by-Step

```bash
# 1. Re-encode audio with corrected duration (IMPORTANT!)
python audio_encoding.py --n-filters 128 --filterbank gammatone

# 2. Extract windowed features (NEW!)
python extract_lsm_windowed_features.py --multiplier 0.6

# 3. Analyze features (optional but recommended)
python analyze_lsm_features.py

# 4. Train CTC
python train_ctc.py
```

## Expected Results

### Before Fixes
- CTC loss stays high (~100+)
- No meaningful text decoded
- Model outputs mostly blanks or gibberish

### After Fixes
- CTC loss should decrease steadily
- Within 200-500 epochs, should see partial word recognition
- By 1000 epochs, should see recognizable words/phrases
- Learning rate should automatically decrease when loss plateaus

### Monitoring Training

Watch for:
- **Loss decreasing**: Should go from ~100 to <50 in first few hundred epochs
- **Decoded text improving**: Should progress from "" → "t" → "the" → partial sentences
- **Learning rate changes**: Scheduler will reduce LR when stuck

### If Still Not Learning

1. **Check feature quality**:
   ```bash
   python analyze_lsm_features.py
   ```
   - Look at PCA plot - classes should be somewhat separated
   - Check temporal structure - should show variation over time

2. **Try different LSM multipliers**:
   ```bash
   python extract_lsm_windowed_features.py --multiplier 0.5
   python extract_lsm_windowed_features.py --multiplier 0.7
   python extract_lsm_windowed_features.py --multiplier 0.8
   ```

3. **Adjust window parameters**:
   ```bash
   # Larger windows = more context, fewer windows
   python extract_lsm_windowed_features.py --window-size 50 --stride 25

   # Smaller windows = less context, more windows (more temporal resolution)
   python extract_lsm_windowed_features.py --window-size 30 --stride 15
   ```

4. **Check dataset**:
   - Are sentences too similar? (CTC might confuse them)
   - Are there enough samples? (200 samples / 25 classes = 8 per class might be low)
   - Consider generating more audio samples if possible

## Key Insights

### Why Windowed Features Work Better Than Traces

**Traces** (what you tried):
- ✓ Preserve temporal information
- ✗ Low discriminative power
- ✗ High-dimensional, noisy

**Windowed Features** (new approach):
- ✓ Preserve temporal information
- ✓ High discriminative power (proven to work for single words)
- ✓ Lower-dimensional, more robust
- ✓ Bridge between classification and sequence-to-sequence

### Why Duration Matters So Much

CTC learns alignment between input and output sequences. With 83% padding:
- Character "h" might align to timesteps 1-150 (mostly silence)
- Character "e" might align to timesteps 151-300 (mostly silence)
- Only ~17% of alignment is meaningful
- Model wastes capacity learning to ignore padding

With 43% padding:
- Much more of the sequence contains useful information
- Alignment is more meaningful
- Model can focus on actual temporal patterns

## Architecture Comparison

### Single Word Pipeline (WORKED)
```
Audio → Spikes → LSM → extract_features_from_spikes() → Aggregated Features → Classifier
                                                         ↑
                                                   (no time dimension)
```

### Sentence Pipeline v1 (FAILED)
```
Audio → Spikes → LSM → get_trace_history_output() → Raw Traces → CTC
                                                      ↑
                                                  (temporal, but poor separability)
```

### Sentence Pipeline v2 (NEW)
```
Audio → Spikes → LSM → extract_features_from_spikes() → Windowed Features → CTC
                        ↑ (in sliding windows)          ↑
                                                    (temporal + good separability)
```

## Next Steps

1. Run the full pipeline with corrected duration
2. Monitor training - should see improvement within first 100 epochs
3. Use `analyze_lsm_features.py` to verify feature quality
4. Experiment with hyperparameters if needed:
   - LSM multiplier (0.5-0.8)
   - Window size (30-50)
   - Stride (15-30)

Good luck!
