import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
import itertools
# --- MODIFIED: Added imports for DataLoader ---
from torch.utils.data import Dataset, DataLoader

# --- 1. Define the Character Set (Vocabulary) ---
# The CTCLoss "blank" token is conventionally 0
BLANK_TOKEN = 0
# Create a mapping from integers to characters
# We add 1 because 0 is reserved for BLANK
CHAR_LIST = " " + "abcdefghijklmnopqrstuvwxyz'"
CHAR_MAP = {char: i + 1 for i, char in enumerate(CHAR_LIST)}
INDEX_MAP = {i + 1: char for i, char in enumerate(CHAR_LIST)}
INDEX_MAP[BLANK_TOKEN] = "<b>" # Representation for "blank"


def load_label_map(filepath="sentence_label_map.txt"):
    """Loads the 'sentence_label_map.txt' file into a dictionary."""
    if not Path(filepath).exists():
        print(f"❌ Error: Label map not found at '{filepath}'")
        return None
    
    label_map = {}
    with open(filepath, "r") as f:
        next(f) # Skip header
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
    # Find the most likely character index at each time step
    # Shape of log_probs: (Time, Num_Classes)
    indices = torch.argmax(log_probs, dim=1)
    
    # Use itertools.groupby to collapse repeats and remove blanks
    # 1. Collapse repeats: [a, a, a, b, b] -> [a, b]
    # 2. Remove blanks: [a, <b>, <b>, b] -> [a, b]
    deduped = [key for key, _ in itertools.groupby(indices)]
    
    decoded_text = ""
    for idx in deduped:
        i = idx.item()
        if i != BLANK_TOKEN:
            decoded_text += INDEX_MAP[i]
            
    return decoded_text.strip()


# --- 2. Define the PyTorch Readout Model ---
class CTCReadout(nn.Module):
    """
    A more powerful readout model using a GRU layer to better learn
    temporal patterns from the LSM state.
    """
    def __init__(self, input_features, num_classes):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_features, 
            hidden_size=256, # More capacity
            num_layers=2,      # Deeper is better
            batch_first=True,  # Input shape is (batch, seq, feature)
            bidirectional=True # Look at past and future
        )
        # The GRU output will have 2 * hidden_size features because it's bidirectional
        self.linear = nn.Linear(256 * 2, num_classes)

    def forward(self, x):
        # Input x shape: (Batch_Size, Time_Steps, LSM_Features)
        
        # Pass through the GRU
        gru_out, _ = self.gru(x)
        
        # Pass the GRU output through the linear layer
        x = self.linear(gru_out)
        
        # Apply Log Softmax for CTC Loss
        return F.log_softmax(x, dim=2)


# --- ADDED: Custom Dataset for DataLoader ---
class LsmDataset(Dataset):
    """Custom Dataset for loading LSM sequences and text."""
    def __init__(self, x_data, y_text_list):
        self.x_data = torch.FloatTensor(x_data) # Keep on CPU
        # Pre-encode text to save time
        self.y_encoded = [encode_text(text) for text in y_text_list]

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        # Return CPU tensors. Batch will be moved to GPU.
        return self.x_data[idx], self.y_encoded[idx]

# --- ADDED: Custom Collate Function for CTC ---
def ctc_collate_fn(batch):
    """
    Custom collate function to batch variable-length text sequences for CTCLoss.
    'batch' is a list of (x_sample, y_encoded_sample)
    """
    # 1. Stack all X samples
    x_samples = torch.stack([item[0] for item in batch])
    
    # 2. Handle Y targets for CTCLoss
    y_encoded_samples = [item[1] for item in batch]
    y_target_lengths = torch.LongTensor([len(seq) for seq in y_encoded_samples])
    y_targets_concatenated = torch.cat(y_encoded_samples)
    
    # 3. Get input lengths (all the same in this case)
    # x_samples shape is (Batch, Time, Features)
    num_timesteps = x_samples.shape[1]
    x_input_lengths = torch.LongTensor([num_timesteps] * len(batch))
    
    return x_samples, y_targets_concatenated, x_input_lengths, y_target_lengths


# --- 3. The Main Training Function ---
def train():
    print("="*60)
    print("🧠 Starting CTC Readout Layer Training...")
    print("="*60)
    
    # --- Training Parameters ---
    BATCH_SIZE = 32 # Adjust this based on your VRAM
    NUM_EPOCHS = 2000
    LEARNING_RATE = 0.0005
    
    # --- Detect and set the device ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ Using device: {device}")
    
    # --- Load Data (as NumPy arrays) ---
    print("Loading data...")
    dataset = np.load("lsm_trace_sequences.npz")
    X_train = dataset['X_train_sequences']
    y_train_indices = dataset['y_train']
    X_test = dataset['X_test_sequences']
    y_test_indices = dataset['y_test']
    
    label_map = load_label_map()
    if label_map is None:
        return
        
    # Get the text for each label index
    y_train_text = [label_map[idx] for idx in y_train_indices]
    y_test_text = [label_map[idx] for idx in y_test_indices]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Sample train label 0: {y_train_indices[0]} -> '{y_train_text[0]}'")

    # --- Visualize the first training sample ---
    print("\nVisualizing the first training sample...")
    plt.figure(figsize=(15, 5))
    plt.imshow(X_train[0].T, aspect='auto', cmap='viridis')
    plt.title(f"LSM Trace for Sample 0: '{y_train_text[0]}'")
    plt.xlabel("Time Steps")
    plt.ylabel("LSM Output Neurons")
    plt.colorbar(label="Trace Value")
    plt.savefig("lsm_trace_visualization.png")
    print("✅ Visualization saved to lsm_trace_visualization.png")

    # --- MODIFIED: Create DataLoader ---
    # Create CPU tensor for test data. We'll move samples to GPU one-by-one.
    X_test_tensor_cpu = torch.FloatTensor(X_test)
    
    # Create Dataset and DataLoader for training
    train_dataset = LsmDataset(X_train, y_train_text)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=ctc_collate_fn, # Use our custom batching function
        pin_memory=True,          # Helps speed up CPU-to-GPU transfer
        num_workers=4             # Spawns processes to load data (set to 0 if this causes errors)
    )
    print(f"✅ Created DataLoader with batch size {BATCH_SIZE}")

    # --- Parameters for Model & CTC ---
    num_samples, num_timesteps, num_lsm_neurons = X_train.shape
    num_classes = len(CHAR_MAP) + 1 # +1 for the BLANK token
    
    print("\n--- Model Configuration ---")
    print(f"LSM Output Features: {num_lsm_neurons}")
    print(f"Time Steps: {num_timesteps}")
    print(f"Num Classes (w/ blank): {num_classes}")
    
    # --- Initialize Model, Loss, and Optimizer ---
    model = CTCReadout(input_features=num_lsm_neurons, num_classes=num_classes)
    model.to(device) # Move the model to the GPU
    
    loss_fn = nn.CTCLoss(blank=BLANK_TOKEN, reduction='mean', zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- MODIFIED: The Training Loop (Mini-Batch) ---
    print(f"Starting training for {NUM_EPOCHS} epochs...")
    
    for epoch in range(NUM_EPOCHS):
        model.train() # Set model to training mode
        total_epoch_loss = 0.0
        
        # --- Batch Loop ---
        for i, (batch_X, batch_y_targets, batch_X_lengths, batch_y_lengths) in enumerate(train_loader):
            
            # Move this batch's data to the GPU
            batch_X = batch_X.to(device)
            batch_y_targets = batch_y_targets.to(device)
            batch_X_lengths = batch_X_lengths.to(device)
            batch_y_lengths = batch_y_lengths.to(device)
            
            optimizer.zero_grad() # Reset gradients for this batch
            
            # --- Forward Pass ---
            # Shape: (Batch_Size, Time_Steps, Num_Classes)
            log_probs = model(batch_X)
            
            # --- Prepare for CTCLoss ---
            # CTCLoss expects (Time_Steps, Batch_Size, Num_Classes)
            log_probs_for_loss = log_probs.permute(1, 0, 2)
            
            # --- Calculate Loss ---
            loss = loss_fn(
                log_probs_for_loss,  # Model output
                batch_y_targets,     # Concatenated batch labels
                batch_X_lengths,     # Length of each sequence in batch
                batch_y_lengths      # Length of each label in batch
            )
            
            # --- Backward Pass ---
            loss.backward()
            optimizer.step()
            
            total_epoch_loss += loss.item()
        # --- End of Batch Loop ---
        
        avg_epoch_loss = total_epoch_loss / len(train_loader)
        
        # --- Print Progress ---
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Avg. Loss: {avg_epoch_loss:.4f}")
            
            # --- Check a prediction (Greedy Decode) ---
            model.eval() # Set model to evaluation mode
            with torch.no_grad():
                # Get prediction for the first test sample
                # Move just this one sample to the GPU
                test_sample_gpu = X_test_tensor_cpu[0].unsqueeze(0).to(device)
                test_sample_log_probs = model(test_sample_gpu)
                
                # Squeeze to (Time_Steps, Num_Classes)
                test_sample_log_probs = test_sample_log_probs.squeeze(0)
                
                # greedy_decoder uses .item(), which pulls data from GPU to CPU
                decoded_text = greedy_decoder(test_sample_log_probs)
                
                print(f"  Test Sample 0 Target: '{y_test_text[0]}'")
                print(f"  Test Sample 0 Decoded: '{decoded_text}'\n")

    print("✅ Training complete.")

if __name__ == "__main__":
    train()