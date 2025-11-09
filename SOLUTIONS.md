# Solutions to CTC Learning Problems

## 📊 Problems Identified from Pipeline Run

### Problem 1: CATASTROPHICALLY SMALL DATASET (Critical - Priority 1)

**Current State**:
```
Total samples: 10
Training samples: 8
Test samples: 2
```

**Why This Breaks CTC**:
- CTC needs to learn 29 character classes (a-z + space + ' + blank)
- Only 8 training samples = 0.28 samples per character class
- CTC learns temporal alignment patterns, needs LOTS of examples
- Typical CTC systems use 1000+ samples minimum
- Your model is memorizing noise, not learning characters

**Evidence**: Decoded "the aw stps oth bard" vs actual "the train moves along the track"

---

### Problem 2: 70% of Features Are Useless

**Current State**:
```
⚠️  WARNING: 1357 features are always zero!
⚠️  WARNING: 1439 features have very low variance!
Feature dimension: 2000 features per window
```

**Why This Matters**:
- Out of 2000 features, 1439 contribute almost nothing
- LSM reservoir neurons are mostly inactive (subcritical regime)
- Model wastes capacity learning from noise
- Increases computational cost and overfitting risk

---

### Problem 3: Feature-to-Sample Ratio Too High

**Current State**:
```
Features per window: 2000
Training samples: 8
Windows per sample: 99
Total training windows: 792
```

**The Problem**:
- 2000 features vs 792 windows = 2.5x more features than data
- Severely underdetermined system
- Curse of dimensionality in full effect

---

## 🔧 SOLUTIONS (Ordered by Priority)

### Solution 1: INCREASE DATASET SIZE (CRITICAL - DO THIS FIRST!)

You have 3 options to fix this:

#### Option A: Generate More Sentences (Best for Real Use)

1. **Edit `generate_sentences.py`** - Add more sentences:

```python
sentences = [
    # Keep your existing 10...
    "The cat sleeps on the mat",
    "The dog runs in the park",
    # ... etc ...

    # ADD AT LEAST 40-90 MORE:
    "A quick brown fox jumps",
    "She sells sea shells",
    "The weather is nice today",
    "I like to eat apples",
    "Music makes me happy",
    # ... add 85+ more varied sentences
]
```

**Target**: 50-100 unique sentences

2. **Generate multiple recordings per sentence** to get voice/timing variation:
   - Use multiple voices in ElevenLabs
   - Or re-generate the same sentence multiple times (slight timing differences)
   - Target: 10 recordings × 50 sentences = 500 samples

---

#### Option B: Audio Data Augmentation (FASTEST - Do This Now!)

I've created `augment_audio_dataset.py` for you. This will:
- Create 10+ augmented versions of each audio file
- Apply time stretching, pitch shifting, noise, volume changes
- Turn your 10 samples into 110+ samples immediately

**Run this now**:

```bash
# Generate augmented dataset
python augment_audio_dataset.py

# This creates sentences_augmented/ with 110 samples (10 original + 100 augmented)
```

**Then update `audio_encoding.py`** line 149 to use augmented data:

```python
# Change this line:
BASE_DATASET_PATH = Path("sentences")
# To:
BASE_DATASET_PATH = Path("sentences_augmented")
```

**Then re-run pipeline**:

```bash
python audio_encoding.py
python extract_lsm_windowed_features_filtered.py --multiplier 0.8
python train_ctc.py
```

---

#### Option C: Use Public Dataset (Best for Research)

Use LibriSpeech or Common Voice dataset:
- LibriSpeech: ~1000 hours of read English
- Common Voice: Multiple languages, user-contributed

This requires adapting your pipeline to work with their format.

---

### Solution 2: FILTER DEAD FEATURES

I've created `extract_lsm_windowed_features_filtered.py` which:
- Removes features with zero variance
- Removes features with very low variance (< 0.01)
- Reduces dimensionality from 2000 → ~500-700 useful features

**Usage**:

```bash
# Extract with feature filtering
python extract_lsm_windowed_features_filtered.py \
    --multiplier 0.8 \
    --variance-threshold 0.01

# Then train (will auto-detect filtered features)
python train_ctc.py
```

**Why higher multiplier (0.8 vs 0.6)?**
- More LSM neurons will be active
- Fewer dead features
- Better feature diversity
- Still stable (below critical = 1.0)

---

### Solution 3: Adjust LSM Parameters for More Activity

**Current issue**: Too many inactive neurons suggests LSM is too subcritical

**Try these multipliers** (after fixing data size):

```bash
# More activity (fewer dead neurons)
python extract_lsm_windowed_features_filtered.py --multiplier 0.8

# Even more activity (if 0.8 still has dead features)
python extract_lsm_windowed_features_filtered.py --multiplier 0.9

# Balance point
python extract_lsm_windowed_features_filtered.py --multiplier 0.85
```

Monitor the feature analysis output - you want < 500 dead features.

---

### Solution 4: Adjust Windowing Parameters

**Current**: window_size=40, stride=20 → 99 windows

**Try larger windows** for more context per window:

```bash
# Larger windows = more temporal context, fewer windows
python extract_lsm_windowed_features_filtered.py \
    --window-size 60 \
    --stride 30 \
    --multiplier 0.8

# This gives ~66 windows instead of 99
# Each window has more information
```

---

## 📋 RECOMMENDED ACTION PLAN

### Immediate Actions (Do Today):

1. **Generate more data** (Pick ONE):

   **Option A (5 minutes)**: Run augmentation
   ```bash
   python augment_audio_dataset.py
   # Edit audio_encoding.py line 149: Path("sentences_augmented")
   ```

   **Option B (30 minutes)**: Add more sentences to generate_sentences.py
   - Add 40-90 more sentences
   - Re-run `python generate_sentences.py`

2. **Re-encode with augmented data**:
   ```bash
   python audio_encoding.py
   ```

3. **Extract with feature filtering**:
   ```bash
   python extract_lsm_windowed_features_filtered.py --multiplier 0.8
   ```

4. **Analyze feature quality**:
   ```bash
   python analyze_lsm_features.py
   ```
   - Check that dead features < 500
   - Check PCA shows class separation

5. **Train CTC**:
   ```bash
   python train_ctc.py
   ```

---

### Expected Improvements

**Before (with 10 samples)**:
- Decoded: "the aw stps oth bard"
- Actual: "the train moves along the track"
- Loss: ~0.025 but nonsense output

**After (with 100+ samples + filtered features)**:
- Should start seeing real words by epoch 200-500
- Should get partial sentences by epoch 1000
- Should get recognizable (if imperfect) sentences by epoch 2000

---

### Monitoring Progress

**Good signs**:
- Loss decreasing steadily
- Decoded text shows actual English words
- PCA plot shows class separation
- < 500 dead features

**Bad signs**:
- Loss stuck or increasing
- Decoded text is gibberish or empty
- PCA plot shows no separation
- > 1000 dead features

If bad signs persist after data augmentation:
1. Try multiplier 0.85-0.9 (more LSM activity)
2. Try larger window sizes (60-80)
3. Check if sentences are too similar (need more diversity)

---

## 🎯 Why These Solutions Work

### Data Augmentation
- CTC needs to see character patterns in different contexts
- Audio variations (speed, pitch, noise) create diverse inputs
- LSM will produce different but related features
- Model learns character invariances, not memorization

### Feature Filtering
- Removes noise features that confuse learning
- Reduces model complexity (fewer parameters)
- Focuses learning on discriminative features
- Improves generalization

### Higher LSM Multiplier
- More neurons active = more feature diversity
- Better coverage of input space
- Less information loss through reservoir
- Still stable (< 1.0 = subcritical)

### Larger Windows
- More temporal context per window
- Better capture of phoneme/character patterns
- Reduces total number of alignment steps for CTC
- Each window is more informative

---

## 🚀 Quick Start (TL;DR)

```bash
# 1. Augment data (takes 2 minutes)
python augment_audio_dataset.py

# 2. Edit audio_encoding.py line 149:
#    BASE_DATASET_PATH = Path("sentences_augmented")

# 3. Run improved pipeline
python audio_encoding.py
python extract_lsm_windowed_features_filtered.py --multiplier 0.8
python analyze_lsm_features.py  # Check quality
python train_ctc.py

# 4. Watch for real words in decoded output by epoch 500
```

This should give you 110 samples, ~600 useful features, and actual learning!

---

## 📈 Success Metrics

After implementing solutions, you should see:

**Epoch 200-500**:
- Loss < 2.0
- Decoded output: Real words appear ("the", "cat", "dog")

**Epoch 500-1000**:
- Loss < 1.0
- Decoded output: Multiple words, partial phrases

**Epoch 1000-2000**:
- Loss < 0.5
- Decoded output: Recognizable sentences (may have errors)

**Example good progression**:
```
Epoch 200: "t c"
Epoch 500: "the cat"
Epoch 1000: "the cat slps on"
Epoch 2000: "the cat sleeps on the mat"
```

If you don't see this progression, you need more data or more diverse sentences.
