import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
import itertools

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
    A simple Linear layer to "read out" the LSM state.
    This is the classifier your professor mentioned.
    """
    def __init__(self, input_features, num_classes):
        super().__init__()
        # input_features = number of LSM output neurons (e.g., 400)
        # num_classes = number of characters + 1 (for blank)
        self.linear = nn.Linear(input_features, num_classes)

    def forward(self, x):
        # Input x shape: (Batch_Size, Time_Steps, LSM_Features)
        # e.g., (8, 400, 400)
        
        # Pass through the linear layer
        x = self.linear(x)
        
        # Apply Log Softmax (CTCLoss expects log probabilities)
        # We apply it on dimension 2 (the class dimension)
        # Output shape: (Batch_Size, Time_Steps, Num_Classes)
        return F.log_softmax(x, dim=2)


# --- 3. The Main Training Function ---
def train():
    print("="*60)
    print("🧠 Starting CTC Readout Layer Training...")
    print("="*60)
    
    # --- Load Data ---
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

    # --- Convert to PyTorch Tensors ---
    X_train_tensor = torch.FloatTensor(X_train)
    X_test_tensor = torch.FloatTensor(X_test)
    
    # --- Parameters for Model & CTC ---
    # From your lsm_sequences.npz output
    num_samples, num_timesteps, num_lsm_neurons = X_train_tensor.shape
    num_classes = len(CHAR_MAP) + 1 # +1 for the BLANK token
    
    print("\n--- Model Configuration ---")
    print(f"LSM Output Features: {num_lsm_neurons}")
    print(f"Time Steps: {num_timesteps}")
    print(f"Num Classes (w/ blank): {num_classes}")
    
    # --- Initialize Model, Loss, and Optimizer ---
    model = CTCReadout(input_features=num_lsm_neurons, num_classes=num_classes)
    
    # CTCLoss(blank=0) tells the loss function that index 0 is the blank token
    # reduction='mean' averages the loss over the batch
    loss_fn = nn.CTCLoss(blank=BLANK_TOKEN, reduction='mean', zero_infinity=True)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # --- Prepare CTC Targets ---
    # CTCLoss needs the labels in a specific format
    
    # 1. Encode all target text into integer sequences
    y_train_encoded = [encode_text(text) for text in y_train_text]
    
    # 2. Concatenate all sequences into one long tensor
    # e.g., [2,5,3] and [1,4] becomes [2,5,3,1,4]
    y_train_targets = torch.cat(y_train_encoded)
    
    # 3. Create a tensor of the *length* of each text label
    # e.g., [3, 2]
    y_train_target_lengths = torch.LongTensor([len(seq) for seq in y_train_encoded])
    
    # 4. Create a tensor of the *length* of each input sequence
    # For us, they are all the same length (400)
    X_train_input_lengths = torch.LongTensor([num_timesteps] * num_samples)

    # --- A quick warning about this tiny dataset ---
    print("\n" + "="*60)
    print("⚠️  NOTE: Your dataset has only 8 training samples.")
    print("    The model will overfit *instantly*. This is normal.")
    print("    Our goal is to prove the pipeline works,")
    print("    not to build a perfect model (yet).")
    print("="*60 + "\n")

    # --- The Training Loop ---
    num_epochs = 300
    print(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        model.train() # Set model to training mode
        optimizer.zero_grad() # Reset gradients
        
        # --- Forward Pass ---
        # Get log probabilities from the model
        # Shape: (Batch_Size, Time_Steps, Num_Classes)
        log_probs = model(X_train_tensor)
        
        # --- Prepare for CTCLoss ---
        # CTCLoss expects (Time_Steps, Batch_Size, Num_Classes)
        # So we must permute (swap) the first two dimensions
        log_probs_for_loss = log_probs.permute(1, 0, 2)
        
        # --- Calculate Loss ---
        loss = loss_fn(
            log_probs_for_loss,  # Model output
            y_train_targets,       # Concatenated text labels
            X_train_input_lengths, # Length of each LSM sequence (all 400)
            y_train_target_lengths # Length of each text label
        )
        
        # --- Backward Pass ---
        loss.backward()
        optimizer.step()
        
        # --- Print Progress ---
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")
            
            # --- Check a prediction (Greedy Decode) ---
            model.eval() # Set model to evaluation mode
            with torch.no_grad():
                # Get prediction for the first test sample
                test_sample_log_probs = model(X_test_tensor[0].unsqueeze(0))
                
                # Squeeze to (Time_Steps, Num_Classes)
                test_sample_log_probs = test_sample_log_probs.squeeze(0)
                
                decoded_text = greedy_decoder(test_sample_log_probs)
                
                print(f"  Test Sample 0 Target: '{y_test_text[0]}'")
                print(f"  Test Sample 0 Decoded: '{decoded_text}'\n")

    print("✅ Training complete.")

if __name__ == "__main__":
    train()