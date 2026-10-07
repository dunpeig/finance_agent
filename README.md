# Shakespeare Transformer Training Interview Problem

## Overview

This interview problem asks you to train a small transformer model on the Shakespeare dataset. You will need to implement the transformer architecture from scratch and train it to generate Shakespeare-like text.

## Task Description

Your task is to:

1. Create a data loader for the Shakespeare dataset
2. Implement a transformer model architecture
3. Train the model on the Shakespeare dataset
4. Generate text samples from your trained model

## What's Provided

- The Shakespeare dataset has been set up for you in the `shakespeare` directory
- A basic project structure to get you started
- Utilities for loading and processing the Shakespeare text data

## What You Need to Implement

1. **Data Loader**: Create a PyTorch Dataset class that loads and processes the Shakespeare text data
2. **Transformer Model**: Implement a transformer architecture for character-level language modeling
3. **Training Loop**: Implement the training process for the transformer model
4. **Text Generation**: Use the trained model to generate Shakespeare-like text

## Getting Started

1. Examine the Shakespeare dataset provided in the `shakespeare` directory
2. Use the `shakespeare_setup.py` file to load the Shakespeare text data
3. Create your solution in the `shakespeare_model.py` file

## Evaluation Criteria

Your solution will be evaluated based on:
- Correctness of the data loader implementation
- Proper implementation of the transformer architecture
- Effectiveness of the training process
- Quality of the generated text
- Performance on a held-out validation set (measured by log likelihood)
- Code organization and clarity

## Hints

- The Shakespeare dataset is a character-level text dataset
- You may want to use a small model configuration for faster training
- Consider implementing a scaled-down version of the transformer architecture described in "Attention is All You Need"

## Validation Testing

Your model will be evaluated on a held-out validation set to measure its performance. The validation metrics include:

- **Validation Loss**: The average negative log likelihood per token on the validation set
- **Perplexity**: exp(validation_loss), a standard metric for language models

A reference model has been trained on the same dataset with the following metrics:
- Reference validation loss: 6.04
- Reference perplexity: 420.17

Your goal is to train a model that achieves similar or better performance on the validation set.
