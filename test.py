
import numpy as np

# Load trace data
data = np.load('lsm_trace_sequences_sentence_split_500.npz')
X_train = data['X_train_sequences']
y_train = data['y_train']

print(f"X_train shape: {X_train.shape}")  # Should be (400, 2000, 700)
print(f"Number of channels: {X_train.shape[2]}")
print(f"First 5 train label indices: {y_train[:5]}")

# Load label map
with open('sentence_label_map_500.txt') as f:
  next(f)  # skip header
  for i, line in enumerate(f):
      if i < 5:  # First 5
          idx, text = line.strip().split(',', 1)
          print(f"Label {idx}: {text}")
      if i >= 4:
          break
