"""
Shakespeare validation module for the interview problem.
This module provides utilities for evaluating models on a validation set.
"""

import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader

# Import the Shakespeare dataset utilities
from shakespeare_setup import load_shakespeare_data

class ShakespeareValidation:
    """
    Utility class for validating Shakespeare transformer models.
    """

    def __init__(self, val_text_path='./output/val_text.txt', ref_metrics_path='./output/reference_metrics.txt',
                 data_dir='./shakespeare'):
        """
        Initialize the validation utility.

        Args:
            val_text_path: Path to the validation text file
            ref_metrics_path: Path to the reference metrics file
            data_dir: Path to full Shakespeare corpus (for vocabulary construction)
        """
        full_text = load_shakespeare_data(data_dir)
        chars = sorted(list(set(full_text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

        # Load the validation text
        with open(val_text_path, 'r', encoding='utf-8') as f:
            self.val_text = f.read()

        # Load the reference metrics
        self.ref_metrics = {}
        with open(ref_metrics_path, 'r', encoding='utf-8') as f:
            for line in f:
                key, value = line.strip().split(': ')
                self.ref_metrics[key] = float(value)

        print(f"Loaded validation text: {len(self.val_text)} characters")
        print(f"Vocabulary: {self.vocab_size} chars (from full corpus)")
        print(f"Loaded reference metrics: {self.ref_metrics}")
    
    def evaluate_model(self, model, dataset, batch_size=64):
        """
        Evaluate a model on the validation dataset and return the average loss (negative log likelihood).
        
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
    
    def validate_model(self, model, dataset_class):
        """
        Validate a model against the reference metrics.

        Args:
            model: The transformer model to validate
            dataset_class: The dataset class to use for creating the validation dataset

        Returns:
            results: Dictionary containing validation results
        """
        val_dataset = dataset_class(self.val_text, stoi=self.stoi, itos=self.itos)
        
        # Evaluate the model
        val_loss, val_perplexity = self.evaluate_model(model, val_dataset)
        
        # Compare with reference metrics
        ref_loss = self.ref_metrics['Reference validation loss']
        ref_perplexity = self.ref_metrics['Reference validation perplexity']
        
        loss_diff = val_loss - ref_loss
        perplexity_ratio = val_perplexity / ref_perplexity
        
        # Prepare results
        results = {
            'val_loss': val_loss,
            'val_perplexity': val_perplexity,
            'ref_loss': ref_loss,
            'ref_perplexity': ref_perplexity,
            'loss_diff': loss_diff,
            'perplexity_ratio': perplexity_ratio,
            'is_better_loss': loss_diff < 0,
            'is_better_perplexity': perplexity_ratio < 1
        }
        
        # Print results
        print(f"\nValidation Results:")
        print(f"Your model's validation loss: {val_loss:.5f}")
        print(f"Your model's perplexity: {val_perplexity:.2f}")
        print(f"Reference validation loss: {ref_loss:.5f}")
        print(f"Reference perplexity: {ref_perplexity:.2f}")
        print(f"Loss difference: {loss_diff:.5f} ({'better' if loss_diff < 0 else 'worse'} than reference)")
        print(f"Perplexity ratio: {perplexity_ratio:.2f}x ({'better' if perplexity_ratio < 1 else 'worse'} than reference)")
        
        return results
