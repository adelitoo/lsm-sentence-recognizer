"""
CTC Training with LINEAR Readout Only (No GRU)

This tests whether the LSM features alone are sufficient,
or if the GRU was doing the heavy lifting.

Key difference from train_ctc_traces.py:
- Replaces 3-layer BiGRU with single linear layer
- Forces LSM to provide all temporal/context information
- If accuracy drops significantly → GRU was doing the work
- If accuracy stays high → LSM is doing the work
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

# --- 1. Character Set ---
BLANK_TOKEN = 0
CHAR_LIST = " " + "abcdefghijklmnopqrstuvwxyz'"
CHAR_MAP = {char: i + 1 for i, char in enumerate(CHAR_LIST)}
INDEX_MAP = {i + 1: char for i, char in enumerate(CHAR_LIST)}
INDEX_MAP[BLANK_TOKEN] = "<b>"


def load_label_map(filepath="sentence_label_map.txt"):
    """Loads the 'sentence_label_map.txt' file into a dictionary."""
    if not Path(filepath).exists():
        print(f"❌ Error: Label map not found at '{filepath}'")
        return None

    label_map = {}
    with open(filepath, "r") as f:
        next(f)
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


def char_error_rate(predicted: str, target: str) -> float:
    """Calculate Character Error Rate (CER)"""
    if len(target) == 0:
        return 1.0 if len(predicted) > 0 else 0.0

    m, n = len(predicted), len(target)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if predicted[i-1] == target[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    return dp[m][n] / len(target)


def char_accuracy(predicted: str, target: str) -> float:
    """Calculate Character-level Accuracy (1 - CER)"""
    return max(0.0, 1.0 - char_error_rate(predicted, target))


# --- Custom Dataset & Collate Function ---
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


# --- 2. SIMPLE LINEAR READOUT (NO GRU!) ---
class CTCReadoutLinear(nn.Module):
    """
    MINIMAL readout model - ONLY a linear layer.

    This forces the LSM to provide all temporal and context information.
    No recurrent processing, no temporal modeling in the readout.

    Compare this to the 3-layer BiGRU in train_ctc_traces.py!
    """

    def __init__(self, input_features, num_classes):
        super().__init__()

        print("\n⚠️  WARNING: Using LINEAR readout only (no GRU)")
        print("   This tests if LSM features alone are sufficient.")
        print("   If accuracy drops significantly, GRU was doing the work.\n")

        # ONLY a single linear layer - no temporal modeling!
        self.linear = nn.Linear(input_features, num_classes)

    def forward(self, x):
        # Input x shape: (Batch_Size, Time_Steps, Membrane_Channels)

        # Direct linear projection - no recurrence, no context modeling
        x = self.linear(x)

        # Apply Log Softmax for CTC Loss
        return F.log_softmax(x, dim=2)


# Alternative: Add ONE small MLP layer (still much simpler than 3-layer BiGRU)
class CTCReadoutMLP(nn.Module):
    """
    Simple MLP readout - one hidden layer, no recurrence.
    Still much simpler than the 3-layer BiGRU.
    """

    def __init__(self, input_features, num_classes, hidden_size=64):
        super().__init__()

        print(f"\n⚠️  Using simple MLP readout (1 hidden layer, {hidden_size} units)")
        print("   No recurrence - LSM must provide temporal features.\n")

        self.fc1 = nn.Linear(input_features, hidden_size)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Input x shape: (Batch_Size, Time_Steps, Membrane_Channels)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return F.log_softmax(x, dim=2)


# --- 3. The Main Training Function ---
def train():
    print("=" * 60)
    print("🧪 EXPERIMENT: Testing LSM vs GRU Contribution")
    print("=" * 60)
    print("\nThis script uses a LINEAR readout (no GRU) to test:")
    print("  - If accuracy stays high → LSM is doing the work ✅")
    print("  - If accuracy drops → GRU was doing the work ❌")
    print("=" * 60)

    # --- Load Trace Data ---
    trace_file = "lsm_trace_sequences.npz"

    if not Path(trace_file).exists():
        print(f"❌ Error: Trace file not found at '{trace_file}'")
        return

    print(f"\n✅ Loading MEMBRANE POTENTIAL TRACES from '{trace_file}'")
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
    print("\nNormalizing membrane potential traces...")
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    feature_mean = X_train_flat.mean(axis=0).reshape(1, 1, -1)
    feature_std = (X_train_flat.std(axis=0) + 1e-8).reshape(1, 1, -1)

    X_train_normalized = (X_train - feature_mean) / feature_std
    X_test_normalized = (X_test - feature_mean) / feature_std

    # --- Setup Device ---
    device = torch.device("x" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Using device: {device}")

    # --- Convert to PyTorch Tensors ---
    X_train_tensor = torch.FloatTensor(X_train_normalized)
    X_test_tensor = torch.FloatTensor(X_test_normalized)

    # --- Model Configuration ---
    num_timesteps = X_train_tensor.shape[1]
    num_channels = X_train_tensor.shape[2]
    num_classes = len(CHAR_MAP) + 1

    print("\n--- Model Configuration ---")
    print(f"Membrane Channels: {num_channels}")
    print(f"Time Steps: {num_timesteps}")
    print(f"Num Classes (w/ blank): {num_classes}")
    print(f"Data type: Continuous membrane voltages from LSM")

    # --- Choose Readout Model ---
    print("\n--- Readout Architecture ---")
    print("Choose readout model:")
    print("  1. Linear only (strictest test)")
    print("  2. Simple MLP (1 hidden layer)")

    # For automatic testing, use linear (option 1)
    USE_MLP = False  # Set to True to use MLP instead

    if USE_MLP:
        model = CTCReadoutMLP(
            input_features=num_channels,
            num_classes=num_classes,
            hidden_size=64
        ).to(device)
        model_name = "MLP"
    else:
        model = CTCReadoutLinear(
            input_features=num_channels,
            num_classes=num_classes
        ).to(device)
        model_name = "Linear"

    loss_fn = nn.CTCLoss(blank=BLANK_TOKEN, reduction="mean", zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Higher LR for simpler model
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=30, min_lr=1e-6, verbose=True
    )

    # --- Prepare DataLoader ---
    BATCH_SIZE = 16
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

    # --- Training Loop ---
    num_epochs = 2000  # May need fewer epochs with simpler model
    print(f"\n🚀 Starting training for {num_epochs} epochs...")

    best_loss = float("inf")
    best_char_acc = 0.0
    patience = 100
    patience_counter = 0
    best_model_state = None

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

            loss = loss_fn(log_probs_for_loss, y_targets, X_input_lengths, y_target_lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        scheduler.step(avg_epoch_loss)

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        # --- Evaluate ---
        if (epoch + 1) % 20 == 0:
            current_lr = optimizer.param_groups[0]["lr"]

            model.eval()
            with torch.no_grad():
                total_cer = 0.0
                total_char_acc = 0.0

                for i in range(min(5, len(X_test_tensor))):
                    test_sample = X_test_tensor[i].unsqueeze(0).to(device)
                    test_log_probs = model(test_sample).squeeze(0)
                    decoded_text = greedy_decoder(test_log_probs.cpu())

                    cer = char_error_rate(decoded_text, y_test_text[i])
                    acc = char_accuracy(decoded_text, y_test_text[i])
                    total_cer += cer
                    total_char_acc += acc

                avg_cer = total_cer / min(5, len(X_test_tensor))
                avg_char_acc = total_char_acc / min(5, len(X_test_tensor))

                if avg_char_acc > best_char_acc:
                    best_char_acc = avg_char_acc

                print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_epoch_loss:.4f}, Best: {best_loss:.4f}, LR: {current_lr:.6f}")
                print(f"  Char Acc: {avg_char_acc*100:.1f}% (Best: {best_char_acc*100:.1f}%) | CER: {avg_cer*100:.1f}%")

                # Show first sample
                test_sample = X_test_tensor[0].unsqueeze(0).to(device)
                test_log_probs = model(test_sample).squeeze(0)
                decoded_text = greedy_decoder(test_log_probs.cpu())
                print(f"  Sample: '{y_test_text[0]}'")
                print(f"  Pred:   '{decoded_text}'\n")

            if patience_counter >= patience:
                print(f"\n⏹️  Early stopping at epoch {epoch+1}")
                if best_model_state:
                    model.load_state_dict(best_model_state)
                    print(f"   ✅ Restored best model")
                break

    print("✅ Training complete.")
    torch.save(model.state_dict(), f"ctc_model_traces_{model_name.lower()}.pt")
    print(f"✅ Model saved to 'ctc_model_traces_{model_name.lower()}.pt'")

    # --- Final Evaluation ---
    print("\n" + "=" * 60)
    print(f"FINAL EVALUATION ({model_name.upper()} READOUT)")
    print("=" * 60)

    model.eval()
    correct = 0
    total = len(X_test_tensor)
    total_cer = 0.0
    total_char_acc = 0.0

    with torch.no_grad():
        for i in range(total):
            test_sample = X_test_tensor[i].unsqueeze(0).to(device)
            test_log_probs = model(test_sample).squeeze(0)
            decoded_text = greedy_decoder(test_log_probs.cpu())

            if decoded_text == y_test_text[i]:
                correct += 1

            cer = char_error_rate(decoded_text, y_test_text[i])
            acc = char_accuracy(decoded_text, y_test_text[i])
            total_cer += cer
            total_char_acc += acc

            if i < 5 or i >= total - 5:
                match = "✅" if decoded_text == y_test_text[i] else "❌"
                print(f"{match} Sample {i}:")
                print(f"  Target:     '{y_test_text[i]}'")
                print(f"  Prediction: '{decoded_text}'")
                print(f"  Char Acc:   {acc*100:.1f}%")

    word_acc = (correct / total) * 100
    avg_char_acc = (total_char_acc / total) * 100
    avg_cer = (total_cer / total) * 100

    print(f"\n{'=' * 60}")
    print(f"RESULTS WITH {model_name.upper()} READOUT:")
    print(f"  Word Accuracy: {correct}/{total} = {word_acc:.2f}%")
    print(f"  Char Accuracy: {avg_char_acc:.2f}%")
    print(f"  CER: {avg_cer:.2f}%")
    print(f"{'=' * 60}")

    # --- Comparison ---
    print("\n" + "=" * 60)
    print("📊 COMPARISON TO 3-LAYER BiGRU")
    print("=" * 60)
    print(f"Your original results (3-layer BiGRU):")
    print(f"  Word Accuracy: 31.00%")
    print(f"  Char Accuracy: 89.77%")
    print(f"  CER: 10.23%")
    print()
    print(f"Current results ({model_name} readout):")
    print(f"  Word Accuracy: {word_acc:.2f}%")
    print(f"  Char Accuracy: {avg_char_acc:.2f}%")
    print(f"  CER: {avg_cer:.2f}%")
    print()

    # Interpretation
    char_acc_drop = 89.77 - avg_char_acc

    if char_acc_drop < 10:
        print("🎉 CONCLUSION: LSM IS DOING THE WORK!")
        print("   The LSM features alone are sufficient for high accuracy.")
        print("   Character accuracy dropped by less than 10%.")
        print("   ✅ Your supervisor's requirement is satisfied!")
    elif char_acc_drop < 30:
        print("⚠️  CONCLUSION: SHARED WORK")
        print("   Both LSM and GRU contribute significantly.")
        print("   Consider enhancing LSM (bigger reservoir, longer memory)")
        print("   to shift more work to the LSM.")
    else:
        print("❌ CONCLUSION: GRU WAS DOING THE HEAVY LIFTING")
        print("   Character accuracy dropped significantly.")
        print("   The LSM features alone are NOT sufficient.")
        print("   You need to enhance the LSM to satisfy your supervisor.")

    print("=" * 60)


if __name__ == "__main__":
    train()
