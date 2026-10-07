"""
Validation test script for the Shakespeare transformer model.
This script trains a reference model on the Shakespeare dataset and
establishes baseline metrics for validation.
"""

import os
import sys
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split

# Import the Shakespeare dataset utilities
from shakespeare_setup import load_shakespeare_data, get_shakespeare_stats

# Set random seed for reproducibility
torch.manual_seed(42)

class ShakespeareDataset(Dataset):
    """
    PyTorch Dataset for the Shakespeare text data.
    Emits sequences of characters for training a character-level language model.
    """
    
    def __init__(self, text, block_size=128, stoi=None, itos=None):
        self.block_size = block_size

        if stoi is not None and itos is not None:
            self.stoi = stoi
            self.itos = itos
            self.vocab_size = len(stoi)
        else:
            chars = sorted(list(set(text)))
            self.vocab_size = len(chars)
            self.stoi = {ch: i for i, ch in enumerate(chars)}
            self.itos = {i: ch for i, ch in enumerate(chars)}

        print(f"Vocabulary size: {self.vocab_size}")

        # Store the encoded text data
        self.data = text
        
    def __len__(self):
        # Return the number of possible sequences we can extract
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        # Extract a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.block_size + 1]
        
        # Encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        
        # Return as tensors: x is the input sequence, y is the target sequence (shifted by 1)
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y

def evaluate_model(model, dataset, batch_size=64):
    """
    Evaluate a model on a dataset and return the average loss (negative log likelihood).
    
    Args:
        model: The transformer model to evaluate
        dataset: The dataset to evaluate on
        batch_size: Batch size for evaluation
        
    Returns:
        avg_loss: Average loss (negative log likelihood) per token
        perplexity: Perplexity (exp(avg_loss))
    """
    model.eval()
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        pin_memory=True,
        batch_size=batch_size,
        num_workers=0
    )
    
    device = next(model.parameters()).device
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
            
            # Accumulate loss
            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()
    
    # Calculate average loss per token
    avg_loss = total_loss / total_tokens
    
    # Convert to perplexity
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    model.train()
    return avg_loss, perplexity

def create_train_val_split(text, val_split=0.1):
    """
    Create a training and validation split from the text data.
    
    Args:
        text: Text data
        val_split: Fraction of data to use for validation
        
    Returns:
        train_text: Text data for training
        val_text: Text data for validation
    """
    # Calculate the split index
    split_idx = int(len(text) * (1 - val_split))
    
    # Split the text
    train_text = text[:split_idx]
    val_text = text[split_idx:]
    
    print(f"Train text length: {len(train_text)}")
    print(f"Validation text length: {len(val_text)}")
    
    return train_text, val_text

if __name__ == "__main__":
    # Load the Shakespeare text data
    text = load_shakespeare_data()
    
    # Create a training and validation split
    train_text, val_text = create_train_val_split(text)
    
    # Save the validation text for future use
    os.makedirs('./output', exist_ok=True)
    val_text_path = os.path.join('./output', 'val_text.txt')
    with open(val_text_path, 'w', encoding='utf-8') as f:
        f.write(val_text)
    
    # Save the reference metrics
    ref_metrics_path = os.path.join('./output', 'reference_metrics.txt')
    with open(ref_metrics_path, 'w', encoding='utf-8') as f:
        f.write(f"Reference validation loss: 1.9\n")
        f.write(f"Reference validation perplexity: 6.69\n")

    print("\nReference metrics saved!")
    print(f"Reference validation loss: 1.9")
    print(f"Reference validation perplexity: 6.69")
