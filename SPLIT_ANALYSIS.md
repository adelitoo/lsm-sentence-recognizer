# Train/Test Split Analysis: Current vs Sentence-Level

## 🔍 Current Split Method (Sample-Level)

### Found in: `extract_lsm_windowed_features_filtered.py` lines 176-178

```python
# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_spikes, y_labels, test_size=0.2, random_state=42
)
```

### What This Does:

1. **Shuffles all 600 samples randomly**
2. **Splits 80/20**: 480 training, 120 test
3. **Random assignment**: Each sample has 20% chance of being in test set

### Example Distribution:

```
Sentence: "the cat sleeps on the mat" (sentence_id = 0)
Has 6 samples total:
  - sample_0_original.wav
  - sample_0_aug1.wav (pitch shifted)
  - sample_0_aug2.wav (time stretched)
  - sample_0_aug3.wav (noisy)
  - sample_0_aug4.wav (volume changed)
  - sample_0_aug5.wav (combined augmentations)

After random split:
  Training set: 4-5 of these samples
  Test set:     1-2 of these samples
```

### What This Tests:

✅ **Robustness to audio variations**
   - Model sees "cat" in training with pitch shift
   - Model tested on "cat" with noise
   - Tests: Can it recognize same words under different conditions?

✅ **Temporal alignment robustness**
   - Same phoneme sequence at different speeds
   - CTC alignment across variations

❌ **NOT testing vocabulary generalization**
   - Model has seen the word "cat" many times in training
   - Not testing if it can recognize "cat" as a *new* word

### Pros:
- More realistic for production (same vocabulary, different speakers/conditions)
- Easier to achieve high accuracy (85-95% reasonable)
- Tests what matters in real deployment

### Cons:
- Doesn't prove the model learned character-level patterns
- Could be memorizing word templates rather than character recognition
- Can't claim true generalization to unseen vocabulary

---

## 🆕 Proposed: Sentence-Level Split (True Generalization)

### What This Will Do:

1. **Split unique sentences first** (before augmentation)
   - 100 sentences → 80 training, 20 test
2. **All versions of a sentence stay together**
   - If "the cat sleeps" is in test, ALL 6 versions go to test
   - Training never sees any version of test sentences

### Example Distribution:

```
Training sentences (80 total):
  - "the cat sleeps on the mat" (sentence_id = 0, 6 samples)
  - "the dog runs in the park" (sentence_id = 1, 6 samples)
  - "a bird sings in the tree" (sentence_id = 2, 6 samples)
  - ... 77 more sentences
  Total: 80 × 6 = 480 samples

Test sentences (20 total - COMPLETELY NEW):
  - "snow falls on the ground" (sentence_id = 80, 6 samples)
  - "rain drops make puddles" (sentence_id = 81, 6 samples)
  - ... 18 more sentences
  Total: 20 × 6 = 120 samples
```

### What This Tests:

✅ **True vocabulary generalization**
   - If test has "snow" but training doesn't, can model recognize 's', 'n', 'o', 'w'?
   - Tests character-level understanding, not word memorization

✅ **Character composition**
   - Can model combine known characters into new words?
   - E.g., seen "cat" and "map", can it recognize "cap"?

✅ **Proof of learning**
   - Much stronger evidence the model learned general character recognition
   - Not just memorizing word templates

### Pros:
- Proves true generalization capability
- More scientifically rigorous
- Demonstrates character-level learning

### Cons:
- More challenging (expect 70-85% accuracy, not 85-95%)
- May see more errors on rare character combinations
- Requires larger training set for good performance

---

## 📊 Expected Accuracy Comparison

### Current (Sample-Level Split):
```
Character Error Rate (CER):   2-5%
Word Error Rate (WER):         8-15%
Sentence Accuracy:             85-95%

Why: Model has seen these words many times, just different audio variations
```

### New (Sentence-Level Split):
```
Character Error Rate (CER):   5-10%
Word Error Rate (WER):         15-25%
Sentence Accuracy:             70-85%

Why: Model must generalize to completely new word combinations
```

### Example Predictions:

**Sample-Level (current):**
```
Test: "the cat sleeps on the mat" (has seen this sentence, different audio)
Predicted: "the cat sleeps on the mat" ✅ (high confidence)
```

**Sentence-Level (new):**
```
Test: "snow falls on the ground" (never seen this sentence)
Predicted: "snow falls on the ground" ✅ (if model learned well)
OR
Predicted: "sno falls on the ground" ❌ (misses 'w' - rare character combo)
```

---

## 🧪 Which Should You Use?

### Use Sample-Level Split If:
- **Production deployment** with known vocabulary
- **Real-world scenario** testing (same words, different speakers/conditions)
- **Proving robustness** to audio variations
- **Want higher accuracy** for confidence

### Use Sentence-Level Split If:
- **Research/academic work** requiring rigor
- **Proving generalization** to new vocabulary
- **Testing if model truly learned characters** vs memorizing words
- **Want stronger claims** about model capabilities

---

## 💡 Recommendation:

**Do BOTH!**

1. **Keep sample-level split** for production evaluation
2. **Add sentence-level split** for generalization testing

Report both results:
```
"The model achieves 87% sentence accuracy on sample-level split
(testing robustness to audio variations) and 76% on sentence-level
split (testing generalization to unseen vocabulary), demonstrating
both robustness and true character-level learning."
```

This is the most honest and complete evaluation! 🎯

---

## 🔧 Implementation Details

### Files to Create:

1. **`extract_lsm_windowed_features_filtered_sentence_split.py`**
   - New version with sentence-level splitting
   - Ensures all augmentations of same sentence stay together

2. **`train_ctc_sentence_split.py`**
   - Loads sentence-split features
   - Trains model on truly unseen sentences
   - Saves model as `ctc_model_sentence_split.pt`

3. **`evaluate_model_sentence_split.py`**
   - Evaluates on unseen sentences
   - Provides CER/WER/accuracy metrics

### Key Code Changes:

```python
# BEFORE (current - sample-level):
X_train, X_test, y_train, y_test = train_test_split(
    X_spikes, y_labels, test_size=0.2, random_state=42
)

# AFTER (new - sentence-level):
# 1. Get unique sentence IDs (0-99)
unique_sentence_ids = np.unique(y_labels)

# 2. Split sentence IDs 80/20
train_sentence_ids, test_sentence_ids = train_test_split(
    unique_sentence_ids, test_size=0.2, random_state=42
)

# 3. Get all samples for train sentences
train_mask = np.isin(y_labels, train_sentence_ids)
X_train = X_spikes[train_mask]
y_train = y_labels[train_mask]

# 4. Get all samples for test sentences
test_mask = np.isin(y_labels, test_sentence_ids)
X_test = X_spikes[test_mask]
y_test = y_labels[test_mask]
```

This ensures all 6 versions of each sentence stay together!

---

## 📈 Next Steps

I will now create:
1. ✅ This analysis document (done!)
2. 🔨 `extract_lsm_windowed_features_filtered_sentence_split.py`
3. 🔨 Updated training/evaluation scripts for sentence-level split
4. 📊 Comparison script to run both evaluations

After running both, you'll have a complete picture of your model's capabilities! 🚀
