"""
Test script for the Shakespeare transformer model solution.
"""

import os
import sys
import torch
from shakespeare_setup import load_shakespeare_data

# Set random seed for reproducibility
torch.manual_seed(42)

class ShakespeareDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for the Shakespeare text data.
    Emits sequences of characters for training a character-level language model.
    """
    
    def __init__(self, text, block_size=128):
        self.block_size = block_size
        
        # Create vocabulary from the text
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        print(f"Vocabulary size: {self.vocab_size}")
        
        # Create mappings from characters to indices and vice versa
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        
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

def generate_sample(model, dataset, prompt, max_new_tokens=100, temperature=0.8):
    """
    Generate a text sample from the trained model.
    
    Args:
        model: Trained transformer model
        dataset: Shakespeare dataset
        prompt: Text prompt to start generation
        max_new_tokens: Maximum number of tokens to generate
        temperature: Temperature for sampling (higher = more random)
    """
    model.eval()
    with torch.no_grad():
        # Convert prompt to tensor
        x = torch.tensor([[dataset.stoi[s] for s in prompt if s in dataset.stoi]], dtype=torch.long)
        
        # Move to the same device as the model
        x = x.to(next(model.parameters()).device)
        
        # Generate text
        y = model.generate(x, max_new_tokens, temperature=temperature, do_sample=True, top_k=10)[0]
        
        # Convert back to text
        completion = ''.join([dataset.itos[int(i)] for i in y])
        
        print(f"\nGenerated text (prompt: '{prompt}'):")
        print(completion)
    
    model.train()
    return completion

if __name__ == "__main__":
    # Load the Shakespeare text data
    text = load_shakespeare_data()
    
    # Create the dataset
    dataset = ShakespeareDataset(text)
    
    print("Test successful!")
