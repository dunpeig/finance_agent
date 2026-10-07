"""
Shakespeare dataset setup for the interview problem.
This file provides the Shakespeare dataset and utilities for loading it.

Note: shakespeare/input.txt is the TRAINING split only. The held-out
validation set lives in output/val_text.txt and is NOT contained in
input.txt, so train on input.txt and evaluate on val_text.txt.

Candidates should use this file as a starting point for their solution.
"""

import os
import torch

def load_shakespeare_data(data_dir='./shakespeare'):
    """
    Load the Shakespeare text data from the data directory.
    
    Args:
        data_dir: Directory containing the Shakespeare text data
        
    Returns:
        text: The raw Shakespeare text data as a string
    """
    input_file = os.path.join(data_dir, 'input.txt')
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    print(f"Shakespeare dataset loaded: {len(text)} characters, {len(set(text))} unique characters")
    return text

def get_shakespeare_stats(text):
    """
    Get basic statistics about the Shakespeare text data.
    
    Args:
        text: The raw text data
        
    Returns:
        stats: Dictionary containing statistics about the text
    """
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    
    stats = {
        'total_chars': len(text),
        'vocab_size': vocab_size,
        'unique_chars': chars,
    }
    
    return stats

# Example usage:
if __name__ == "__main__":
    # Load the Shakespeare text data
    text = load_shakespeare_data()
    
    # Get statistics about the data
    stats = get_shakespeare_stats(text)
    
    print(f"Total characters: {stats['total_chars']}")
    print(f"Vocabulary size: {stats['vocab_size']}")
    print(f"First 100 characters of the text:")
    print(text[:100])
    
    print("\nUnique characters in the text:")
    print(''.join(stats['unique_chars']))
    
    print("\nNOTE: This is just the dataset setup. Your task is to:")
    print("1. Create a PyTorch Dataset class for the Shakespeare text")
    print("2. Implement a training loop using the minGPT framework")
    print("3. Generate text samples from your trained model")
