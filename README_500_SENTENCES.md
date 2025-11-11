# 500-Sentence Dataset Pipeline

## Overview

This pipeline implements **sentence-level splitting with vocabulary control** for the 500-sentence dataset generated using Chatterbox TTS. It ensures true generalization testing by:

1. Splitting at the **sentence level** (not sample level)
2. Ensuring test sentences are **completely unseen** during training
3. **Maximizing word overlap** between train/test vocabularies (75% target)
4. Preventing unknown words from appearing in the test set

## What's Already Implemented

Your project already has a complete implementation of sentence-level splitting with vocabulary control:

### Core Scripts

1. **`create_balanced_sentence_split.py`**
   - Creates train/test split by sentence IDs
   - Maximizes word coverage (75% target)
   - Ensures test words appear in training vocabulary

2. **`extract_lsm_windowed_features_filtered_sentence_split.py`**
   - LSM feature extraction with sentence-level split
   - Loads balanced split from `.npz` file
   - Verifies no sentence overlap

3. **`train_ctc.py`**
   - Automatically detects sentence-level vs sample-level splits
   - Loads appropriate feature files
   - Reports split type during training

### 500-Sentence Adaptations (Created)

I've created adapted versions for your 500-sentence dataset:

1. **`audio_encoding_500.py`**
   - Processes `sentences_500/` directory
   - Outputs: `sentence_spike_trains_500.npz`, `sentence_label_map_500.txt`

2. **`create_balanced_sentence_split_500.py`**
   - Creates balanced split for 500 sentences
   - Outputs: `balanced_sentence_split_500.npz`

3. **`extract_lsm_windowed_features_filtered_sentence_split_500.py`**
   - LSM extraction for 500-sentence dataset
   - Outputs: `lsm_windowed_features_filtered_sentence_split_500.npz`

4. **`run_pipeline_500.py`**
   - Complete automated pipeline
   - Runs all steps in sequence

## How to Use

### Option 1: Automated Pipeline (Recommended)

```bash
# Run the complete pipeline
python run_pipeline_500.py

# With custom LSM multiplier
python run_pipeline_500.py --multiplier 0.9

# Skip encoding if already done
python run_pipeline_500.py --skip-encoding
```

### Option 2: Step-by-Step

```bash
# Step 1: Audio encoding (spike train generation)
python audio_encoding_500.py --filterbank gammatone --n-filters 128

# Step 2: Create balanced sentence split
python create_balanced_sentence_split_500.py

# Step 3: LSM feature extraction
python extract_lsm_windowed_features_filtered_sentence_split_500.py --multiplier 0.8

# Step 4: Training (requires modification - see below)
python train_ctc.py
```

## Generated Files

After running the pipeline, you'll have:

```
sentence_spike_trains_500.npz              # Spike train dataset (500 samples)
sentence_label_map_500.txt                  # Label ID to text mapping
balanced_sentence_split_500.npz             # Train/test split (400 train, 100 test)
lsm_windowed_features_filtered_sentence_split_500.npz  # LSM features
```

## Dataset Split Details

### Sentence-Level Split (TRUE Generalization)

- **Train**: ~400 sentences (80%)
- **Test**: ~100 sentences (20%)
- **Key Property**: Test sentences are COMPLETELY UNSEEN during training
- **Vocabulary Control**: 75% of test words appear in training vocabulary

### What This Tests

✅ **True generalization** to new sentences
✅ **Compositional understanding** (combining known words in new ways)
✅ **Character-level learning** (not sentence memorization)

### What This Doesn't Test

❌ Robustness to audio variations (use sample-level split for this)
❌ Handling completely unknown words

## Training with 500-Sentence Data

Currently, `train_ctc.py` automatically loads the highest priority dataset it finds:

```python
# Priority order:
1. lsm_windowed_features_filtered_sentence_split.npz  # Sentence-level split
2. lsm_windowed_features_filtered.npz                 # Sample-level split
3. lsm_windowed_features.npz                          # Basic features
4. lsm_trace_sequences.npz                            # Raw traces
```

### To use the 500-sentence data, you have two options:

**Option A: Modify train_ctc.py (lines 103-129)**

Add detection for the 500-sentence file:

```python
sentence_split_500_file = "lsm_windowed_features_filtered_sentence_split_500.npz"
if Path(sentence_split_500_file).exists():
    print(f"✅ Loading 500-SENTENCE dataset from '{sentence_split_500_file}'")
    dataset = np.load(sentence_split_500_file)
    split_type = "sentence-level-500"
    # Also update load_label_map() to use "sentence_label_map_500.txt"
```

**Option B: Temporarily rename files**

```bash
# Backup existing files
mv sentence_spike_trains.npz sentence_spike_trains.bak
mv sentence_label_map.txt sentence_label_map.bak

# Link to 500-sentence files
ln -s sentence_spike_trains_500.npz sentence_spike_trains.npz
ln -s sentence_label_map_500.txt sentence_label_map.txt
ln -s lsm_windowed_features_filtered_sentence_split_500.npz \
      lsm_windowed_features_filtered_sentence_split.npz

# Run training
python train_ctc.py

# Restore
rm sentence_spike_trains.npz sentence_label_map.txt \
   lsm_windowed_features_filtered_sentence_split.npz
mv sentence_spike_trains.bak sentence_spike_trains.npz
mv sentence_label_map.bak sentence_label_map.txt
```

## Data Augmentation: Should You Use It?

### My Recommendation: Start WITHOUT augmentation

**Why?**
1. You have 500 diverse sentences - substantial dataset
2. LSMs are sensitive to precise spike timing
3. Audio augmentation may corrupt temporal patterns
4. Get a clean baseline first

**If you do augment later:**
- Use minimal augmentations (slight speed variations only)
- Avoid pitch shifting (changes frequency encoding)
- Avoid noise (disrupts spike thresholds)

## Expected Results

### Accuracy Expectations

| Dataset | Split Type | Expected Accuracy | What It Tests |
|---------|-----------|-------------------|---------------|
| 500 sentences | Sentence-level | 70-85% | True generalization |
| 500 sentences (augmented) | Sentence-level | 75-90% | Robustness + generalization |
| 200 sentences (augmented) | Sample-level | 85-95% | Robustness to audio variations |

Lower accuracy on sentence-level split is **expected and good** - it proves the model is truly learning, not memorizing.

## Vocabulary Coverage Example

After creating the balanced split, you'll see output like:

```
Train vocabulary: 150 words
Test vocabulary: 80 words
Word coverage: 60/80 = 75.0%

Uncovered test words (20): ['afternoon', 'excellent', 'purple', ...]
```

This means:
- 75% of test words appeared in training (learnable)
- 25% are truly novel (challenging but still character-level)

## Architecture Summary

```
generate_500_sentences.py (Chatterbox TTS)
         ↓
    500 .wav files
         ↓
audio_encoding_500.py (Gammatone → Spikes)
         ↓
sentence_spike_trains_500.npz (500 samples)
         ↓
create_balanced_sentence_split_500.py
         ↓
balanced_sentence_split_500.npz (400 train, 100 test sentence IDs)
         ↓
extract_lsm_windowed_features_filtered_sentence_split_500.py
         ↓
lsm_windowed_features_filtered_sentence_split_500.npz
         ↓
train_ctc.py (needs modification)
         ↓
    Trained model
```

## Key Differences: Sample-Level vs Sentence-Level

| Aspect | Sample-Level | Sentence-Level |
|--------|--------------|----------------|
| **Split Unit** | Individual audio samples | Unique sentence IDs |
| **Test Overlap** | Same sentences in train/test | NO overlap |
| **What It Tests** | Audio robustness | True generalization |
| **Expected Acc** | 85-95% | 70-85% |
| **Use Case** | Production deployment | Research validation |

## Troubleshooting

### "Balanced split file not found"
Run: `python create_balanced_sentence_split_500.py`

### "Dataset not found"
Run: `python audio_encoding_500.py`

### "Test accuracy is 0%"
This likely means `train_ctc.py` is loading the wrong label map. Ensure:
- `sentence_label_map_500.txt` exists
- `train_ctc.py` loads the correct label map

### "Loss is NaN"
Try reducing the LSM weight multiplier:
```bash
python extract_lsm_windowed_features_filtered_sentence_split_500.py --multiplier 0.6
```

## References

See these files for implementation details:
- `verify_split_method.py` - Compare sample vs sentence splits
- `compare_splits.py` - Detailed split comparison
- `CLAUDE.md` - Full project documentation

## Questions?

The implementation is complete and follows these principles:
1. Sentence-level split ensures no test leakage
2. Balanced vocabulary selection maximizes success rate
3. Character-level CTC learning proves true understanding

You're ready to run the pipeline and train!
