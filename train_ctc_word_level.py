import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
import itertools
from collections import Counter

# --- 1. Build Word Vocabulary ---
BLANK_TOKEN = 0

def build_word_vocabulary(label_map_file="sentence_label_map_500.txt"):
    """Build word vocabulary from sentences"""
    with open(label_map_file) as f:
        next(f)  # skip header
        sentences = [line.strip().split(',', 1)[1] for line in f]

    # Count all words
    word_counts = Counter()
    for sent in sentences:
        word_counts.update(sent.lower().split())

    # Create vocabulary (sorted by frequency for stability)
    vocab_words = [word for word, _ in word_counts.most_common()]

    # Create mappings (reserve 0 for blank)
    WORD_MAP = {word: i + 1 for i, word in enumerate(vocab_words)}
    INDEX_MAP = {i + 1: word for i, word in enumerate(vocab_words)}
    INDEX_MAP[BLANK_TOKEN] = "<blank>"

    return WORD_MAP, INDEX_MAP, vocab_words


def load_label_map(filepath="sentence_label_map_500.txt"):
    """Loads the sentence label map"""
    if not Path(filepath).exists():
        print(f"❌ Error: Label map not found at '{filepath}'")
        return None

    label_map = {}
    with open(filepath, "r") as f:
        next(f)  # Skip header
        for line in f:
            idx, text = line.strip().split(",", 1)
            label_map[int(idx)] = text.lower()
    print(f"✅ Loaded label map with {len(label_map)} entries.")
    return label_map


def encode_text_words(text: str, word_map: dict) -> torch.LongTensor:
    """Encodes a text string into word tokens"""
    words = text.split()
    encoded = [word_map[word] for word in words if word in word_map]
    return torch.LongTensor(encoded)


def greedy_decoder_words(log_probs: torch.Tensor, index_map: dict) -> str:
    """Decodes word-level CTC output"""
    # Find most likely word at each time step
    indices = torch.argmax(log_probs, dim=1)

    # Collapse repeats and remove blanks
    deduped = [key for key, _ in itertools.groupby(indices)]

    decoded_words = []
    for idx in deduped:
        i = idx.item()
        if i != BLANK_TOKEN:
            decoded_words.append(index_map[i])

    return " ".join(decoded_words)


# --- 2. Word-Level CTC Model ---
class WordLevelCTCReadout(nn.Module):
    """Word-level CTC model"""
    def __init__(self, input_features, num_word_classes):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_features,
            hidden_size=128,
            num_layers=3,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        self.linear = nn.Linear(128 * 2, num_word_classes)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        x = self.linear(gru_out)
        return F.log_softmax(x, dim=2)


# --- 3. Training Function ---
def train():
    print("="*60)
    print("🧠 Starting WORD-LEVEL CTC Training...")
    print("="*60)

    # Build vocabulary
    print("\nBuilding word vocabulary...")
    WORD_MAP, INDEX_MAP, vocab_words = build_word_vocabulary()
    print(f"✅ Vocabulary size: {len(vocab_words)} unique words")
    print(f"   Most common: {vocab_words[:10]}")

    # Load data
    print("\nLoading data...")
    sentence_split_file = "lsm_windowed_features_filtered_sentence_split_500.npz"

    if not Path(sentence_split_file).exists():
        print(f"❌ Error: Features not found at '{sentence_split_file}'")
        print("Please run feature extraction first.")
        return

    print(f"✅ Loading from '{sentence_split_file}'")
    dataset = np.load(sentence_split_file)
    split_type = "sentence-level"

    X_train = dataset['X_train_sequences']
    y_train_indices = dataset['y_train']
    X_test = dataset['X_test_sequences']
    y_test_indices = dataset['y_test']

    label_map = load_label_map()
    if label_map is None:
        return

    # Get text for each label
    y_train_text = [label_map[idx] for idx in y_train_indices]
    y_test_text = [label_map[idx] for idx in y_test_indices]

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Sample train label: '{y_train_text[0]}'")

    # Normalize features
    print("\nNormalizing features...")
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    X_test_flat = X_test.reshape(-1, X_test.shape[-1])

    feature_mean = X_train_flat.mean(axis=0).reshape(1, 1, -1)
    feature_std = (X_train_flat.std(axis=0) + 1e-8).reshape(1, 1, -1)

    X_train_normalized = (X_train - feature_mean) / feature_std
    X_test_normalized = (X_test - feature_mean) / feature_std

    # --- Setup Device (GPU/CPU) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Using device: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_normalized).to(device)
    X_test_tensor = torch.FloatTensor(X_test_normalized).to(device)

    # Model configuration
    num_samples, num_timesteps, num_features = X_train_tensor.shape
    num_word_classes = len(WORD_MAP) + 1  # +1 for blank

    print("\n--- Model Configuration ---")
    print(f"LSM Features: {num_features}")
    print(f"Time Steps: {num_timesteps}")
    print(f"Word Vocabulary: {len(WORD_MAP)} words")
    print(f"Num Classes (w/ blank): {num_word_classes}")

    # Initialize model
    model = WordLevelCTCReadout(input_features=num_features, num_word_classes=num_word_classes).to(device)
    loss_fn = nn.CTCLoss(blank=BLANK_TOKEN, reduction='mean', zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=200, min_lr=1e-7
    )

    # Prepare CTC targets (word-level)
    y_train_encoded = [encode_text_words(text, WORD_MAP) for text in y_train_text]
    y_train_targets = torch.cat(y_train_encoded).to(device)
    y_train_target_lengths = torch.LongTensor([len(seq) for seq in y_train_encoded]).to(device)
    X_train_input_lengths = torch.LongTensor([num_timesteps] * num_samples).to(device)

    # Training loop
    num_epochs = 5000
    print(f"\nStarting training for {num_epochs} epochs...")

    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        log_probs = model(X_train_tensor)
        log_probs_for_loss = log_probs.permute(1, 0, 2)

        # Calculate loss
        loss = loss_fn(
            log_probs_for_loss,
            y_train_targets,
            X_train_input_lengths,
            y_train_target_lengths
        )

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(loss)

        if loss.item() < best_loss:
            best_loss = loss.item()

        # Print progress
        if (epoch + 1) % 20 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, Best: {best_loss:.4f}, LR: {current_lr:.6f}")

            # Check prediction
            model.eval()
            with torch.no_grad():
                test_log_probs = model(X_test_tensor[0].unsqueeze(0)).squeeze(0)
                decoded_text = greedy_decoder_words(test_log_probs, INDEX_MAP)

                print(f"  Test Sample 0 Target: '{y_test_text[0]}'")
                print(f"  Test Sample 0 Decoded: '{decoded_text}'\n")

    print("✅ Training complete.")

    # Save model
    torch.save(model.state_dict(), "ctc_model_word_level.pt")
    print(f"✅ Model saved to 'ctc_model_word_level.pt'")

    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION ON ALL TEST SAMPLES")
    print("="*60)

    model.eval()
    correct = 0
    total = len(X_test_tensor)

    with torch.no_grad():
        for i in range(total):
            test_log_probs = model(X_test_tensor[i].unsqueeze(0)).squeeze(0)
            decoded_text = greedy_decoder_words(test_log_probs, INDEX_MAP)

            if decoded_text == y_test_text[i]:
                correct += 1

            # Show first 5 and last 5
            if i < 5 or i >= total - 5:
                match_symbol = "✅" if decoded_text == y_test_text[i] else "❌"
                print(f"{match_symbol} Sample {i}:")
                print(f"  Target:     '{y_test_text[i]}'")
                print(f"  Prediction: '{decoded_text}'")

    accuracy = (correct / total) * 100
    print(f"\n{'='*60}")
    print(f"Test Set Accuracy: {correct}/{total} = {accuracy:.2f}%")
    print(f"Split Type: WORD-LEVEL (instead of character-level)")
    print(f"{'='*60}")


if __name__ == "__main__":
    train()
