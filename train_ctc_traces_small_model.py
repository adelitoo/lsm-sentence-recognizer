"""
CTC Training with SMALLER MODEL for limited data (500 sentences)

Key changes from original:
- Smaller GRU (1-2 layers, 64 hidden units)
- More dropout (0.3-0.4)
- Stronger weight decay
- Lower learning rate
- Early stopping
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
import itertools
from torch.utils.data import Dataset, DataLoader

# --- 1. Define the Character Set (Vocabulary) ---
BLANK_TOKEN = 0
CHAR_LIST = " " + "abcdefghijklmnopqrstuvwxyz'"
CHAR_MAP = {char: i + 1 for i, char in enumerate(CHAR_LIST)}
INDEX_MAP = {i + 1: char for i, char in enumerate(CHAR_LIST)}
INDEX_MAP[BLANK_TOKEN] = "<b>"


def load_label_map(filepath="sentence_label_map_500.txt"):
    """Loads the 'sentence_label_map.txt' file into a dictionary."""
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


def encode_text(text: str) -> torch.LongTensor:
    """Encodes a text string into a tensor of integer labels."""
    encoded = [CHAR_MAP[char] for char in text if char in CHAR_MAP]
    return torch.LongTensor(encoded)


def greedy_decoder(log_probs: torch.Tensor) -> str:
    """Decodes the output of the CTC model into text."""
    indices = torch.argmax(log_probs, dim=1)
    deduped = [key for key, _ in itertools.groupby(indices)]
    decoded_text = ""
    for idx in deduped:
        i = idx.item()
        if i != BLANK_TOKEN:
            decoded_text += INDEX_MAP[i]
    return decoded_text.strip()


class TraceDataset(Dataset):
    """Custom Dataset to handle (trace, encoded_text) pairs."""
    def __init__(self, x_data, y_encoded_list):
        self.x_data = x_data
        self.y_data = y_encoded_list

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]


def ctc_collate_fn(batch):
    """Custom collate function to batch variable-length CTC data."""
    x_batch = torch.stack([item[0] for item in batch])
    y_batch_list = [item[1] for item in batch]
    y_targets_concat = torch.cat(y_batch_list)
    y_target_lengths = torch.LongTensor([len(seq) for seq in y_batch_list])
    num_timesteps = x_batch.shape[1]
    input_lengths = torch.LongTensor([num_timesteps] * len(batch))

    return x_batch, y_targets_concat, input_lengths, y_target_lengths


class CTCReadoutSmall(nn.Module):
    """
    SMALLER readout model for limited data

    Key changes:
    - Fewer layers (2 instead of 3)
    - Smaller hidden size (64 instead of 128)
    - More dropout (0.3 instead of 0.2)
    """

    def __init__(self, input_features, num_classes):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_features,
            hidden_size=64,  # REDUCED from 128
            num_layers=2,    # REDUCED from 3
            batch_first=True,
            bidirectional=True,
            dropout=0.3,     # INCREASED from 0.2
        )
        self.dropout = nn.Dropout(0.3)  # Additional dropout
        self.linear = nn.Linear(64 * 2, num_classes)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        gru_out = self.dropout(gru_out)  # Additional dropout
        x = self.linear(gru_out)
        return F.log_softmax(x, dim=2)


def train():
    print("=" * 60)
    print("🧠 CTC Training with SMALLER MODEL (for limited data)")
    print("=" * 60)

    # --- Load Trace Data ---
    print("\nLoading trace data...")
    trace_file = "lsm_trace_sequences_sentence_split_500.npz"

    if not Path(trace_file).exists():
        print(f"❌ Error: Trace file not found at '{trace_file}'")
        return

    print(f"✅ Loading from '{trace_file}'")
    dataset = np.load(trace_file, allow_pickle=True)

    X_train = dataset["X_train_sequences"]
    y_train = dataset["y_train"]
    X_test = dataset["X_test_sequences"]
    y_test = dataset["y_test"]

    label_map = load_label_map()
    if label_map is None:
        return

    y_train_text = [label_map[idx] for idx in y_train]
    y_test_text = [label_map[idx] for idx in y_test]

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # --- Normalize Features ---
    print("\nNormalizing traces...")
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    X_test_flat = X_test.reshape(-1, X_test.shape[-1])

    feature_mean = X_train_flat.mean(axis=0).reshape(1, 1, -1)
    feature_std = (X_train_flat.std(axis=0) + 1e-8).reshape(1, 1, -1)

    X_train_normalized = (X_train - feature_mean) / feature_std
    X_test_normalized = (X_test - feature_mean) / feature_std

    # --- Setup Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Using device: {device}")

    # --- Convert to Tensors ---
    X_train_tensor = torch.FloatTensor(X_train_normalized)
    X_test_tensor = torch.FloatTensor(X_test_normalized)

    num_samples, num_timesteps, num_membrane_channels = X_train_tensor.shape
    num_classes = len(CHAR_MAP) + 1

    print("\n--- Model Configuration ---")
    print(f"Membrane Channels: {num_membrane_channels}")
    print(f"Time Steps: {num_timesteps}")
    print(f"Num Classes: {num_classes}")
    print(f"Model: SMALL (2 layers, 64 hidden, dropout=0.3)")

    # --- Initialize SMALL Model ---
    model = CTCReadoutSmall(input_features=num_membrane_channels, num_classes=num_classes).to(device)

    loss_fn = nn.CTCLoss(blank=BLANK_TOKEN, reduction="mean", zero_infinity=True)

    # LOWER learning rate and HIGHER weight decay
    optimizer = optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=100, min_lr=1e-7
    )

    # --- Prepare DataLoader ---
    BATCH_SIZE = 8  # SMALLER batch size
    y_train_encoded = [encode_text(text) for text in y_train_text]

    train_dataset = TraceDataset(X_train_tensor, y_train_encoded)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=ctc_collate_fn,
        num_workers=2,
        pin_memory=True
    )

    # --- Training Loop with Early Stopping ---
    num_epochs = 3000
    print(f"\nStarting training for {num_epochs} epochs (Batch Size: {BATCH_SIZE})")

    best_loss = float("inf")
    patience = 500  # Early stopping patience
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for X_batch, y_targets, X_input_lengths, y_target_lengths in train_loader:
            X_batch = X_batch.to(device)
            y_targets = y_targets.to(device)
            X_input_lengths = X_input_lengths.to(device)
            y_target_lengths = y_target_lengths.to(device)

            optimizer.zero_grad()
            log_probs = model(X_batch)
            log_probs_for_loss = log_probs.permute(1, 0, 2)

            loss = loss_fn(
                log_probs_for_loss,
                y_targets,
                X_input_lengths,
                y_target_lengths,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        scheduler.step(avg_epoch_loss)

        # Track best loss and early stopping
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), "ctc_model_traces_best.pt")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"\n⚠️  Early stopping at epoch {epoch + 1}")
            break

        # Print progress
        if (epoch + 1) % 20 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}, "
                f"Best: {best_loss:.4f}, LR: {current_lr:.6f}, Patience: {patience_counter}"
            )

            model.eval()
            with torch.no_grad():
                test_sample = X_test_tensor[0].unsqueeze(0).to(device)
                test_sample_log_probs = model(test_sample).squeeze(0)
                decoded_text = greedy_decoder(test_sample_log_probs.cpu())

                print(f"  Test Sample 0 Target: '{y_test_text[0]}'")
                print(f"  Test Sample 0 Decoded: '{decoded_text}'\n")

    print("✅ Training complete.")

    # Load best model for evaluation
    model.load_state_dict(torch.load("ctc_model_traces_best.pt"))
    print(f"✅ Loaded best model (loss: {best_loss:.4f})")

    # Evaluate
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)

    model.eval()
    correct = 0
    total = len(X_test_tensor)

    with torch.no_grad():
        for i in range(total):
            test_sample = X_test_tensor[i].unsqueeze(0).to(device)
            test_sample_log_probs = model(test_sample).squeeze(0)
            decoded_text = greedy_decoder(test_sample_log_probs.cpu())

            if decoded_text == y_test_text[i]:
                correct += 1

            if i < 5 or i >= total - 5:
                match_symbol = "✅" if decoded_text == y_test_text[i] else "❌"
                print(f"{match_symbol} Sample {i}:")
                print(f"  Target:     '{y_test_text[i]}'")
                print(f"  Prediction: '{decoded_text}'")

    accuracy = (correct / total) * 100
    print(f"\n{'=' * 60}")
    print(f"Test Accuracy: {correct}/{total} = {accuracy:.2f}%")
    print(f"Model: SMALL (2 layers, 64 hidden, dropout=0.3)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    train()
