"""
Shakespeare Transformer Model

This file is a template for your solution to the interview problem.
You need to implement:
1. A PyTorch Dataset class for the Shakespeare text data
2. A transformer model architecture
3. Training a transformer model on the Shakespeare dataset
4. Text generation using the trained model

Use the shakespeare_setup.py file to load the Shakespeare text data.
"""

import os
import math
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Import the Shakespeare dataset utilities
from shakespeare_setup import load_shakespeare_data, get_shakespeare_stats
from shakespeare_validation import ShakespeareValidation

# Set random seed for reproducibility
torch.manual_seed(42)

class ShakespeareDataset(Dataset):
    """
    PyTorch Dataset for the Shakespeare text data.
    Emits sequences of characters for training a character-level language model.
    """
    
    def __init__(self, text, block_size=128, stoi=None, itos=None):
        if block_size <= 0:
            raise ValueError("block_size must be a positive integer")
        if (stoi is None) != (itos is None):
            raise ValueError("stoi and itos must be provided together")

        self.block_size = block_size

        if stoi is None:
            chars = get_shakespeare_stats(text)["unique_chars"]
            self.stoi = {char: index for index, char in enumerate(chars)}
            self.itos = {index: char for char, index in self.stoi.items()}
        else:
            # Reuse the training vocabulary when constructing validation data.
            self.stoi = stoi
            self.itos = itos

        self.vocab_size = len(self.stoi)
        self.data = torch.tensor(self.encode(text), dtype=torch.long)

    def encode(self, text):
        """Convert a string to character-token IDs."""
        return [self.stoi[char] for char in text]

    def decode(self, token_ids):
        """Convert character-token IDs back to a string."""
        return "".join(self.itos[int(token_id)] for token_id in token_ids)
    
    def __len__(self):
        return max(0, len(self.data) - self.block_size)
    
    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size + 1]
        if len(chunk) != self.block_size + 1:
            raise IndexError("dataset index out of range")
        return chunk[:-1], chunk[1:]

class _CausalSelfAttention(torch.nn.Module):
    """Multi-head self-attention that cannot attend to future tokens."""

    def __init__(self, d_model, nhead, max_seq_length, dropout):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")

        self.nhead = nhead
        self.head_size = d_model // nhead
        self.qkv_projection = torch.nn.Linear(d_model, 3 * d_model)
        self.output_projection = torch.nn.Linear(d_model, d_model)
        self.attention_dropout = torch.nn.Dropout(dropout)
        self.residual_dropout = torch.nn.Dropout(dropout)
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(max_seq_length, max_seq_length, dtype=torch.bool),
                diagonal=1,
            ),
        )

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        q, k, v = self.qkv_projection(x).chunk(3, dim=-1)

        def split_heads(tensor):
            return tensor.view(
                batch_size, seq_len, self.nhead, self.head_size
            ).transpose(1, 2)

        q, k, v = map(split_heads, (q, k, v))
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_size)
        mask = self.causal_mask[:seq_len, :seq_len]
        scores = scores.masked_fill(mask, float("-inf"))
        weights = self.attention_dropout(F.softmax(scores, dim=-1))

        attended = torch.matmul(weights, v)
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        return self.residual_dropout(self.output_projection(attended))


class _TransformerBlock(torch.nn.Module):
    """Pre-normalized decoder block with attention and feed-forward layers."""

    def __init__(self, d_model, nhead, d_ff, max_seq_length, dropout):
        super().__init__()
        self.attention_norm = torch.nn.LayerNorm(d_model)
        self.attention = _CausalSelfAttention(
            d_model, nhead, max_seq_length, dropout
        )
        self.feed_forward_norm = torch.nn.LayerNorm(d_model)
        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_ff),
            torch.nn.GELU(),
            torch.nn.Linear(d_ff, d_model),
            torch.nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        return x + self.feed_forward(self.feed_forward_norm(x))


class TransformerModel(torch.nn.Module):
    """
    Transformer model for character-level language modeling.
    """
    
    def __init__(self, vocab_size, d_model, nhead, num_layers, d_ff, max_seq_length, dropout=0.1):
        """
        Initialize the transformer model.
        
        Args:
            vocab_size: Size of the vocabulary
            d_model: Dimension of the model
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            d_ff: Dimension of the feedforward network
            max_seq_length: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        if max_seq_length <= 0:
            raise ValueError("max_seq_length must be positive")

        self.max_seq_length = max_seq_length
        self.token_embedding = torch.nn.Embedding(vocab_size, d_model)
        self.position_embedding = torch.nn.Embedding(max_seq_length, d_model)
        self.embedding_dropout = torch.nn.Dropout(dropout)
        self.blocks = torch.nn.ModuleList([
            _TransformerBlock(
                d_model, nhead, d_ff, max_seq_length, dropout
            )
            for _ in range(num_layers)
        ])
        self.final_norm = torch.nn.LayerNorm(d_model)
        self.lm_head = torch.nn.Linear(d_model, vocab_size, bias=False)

        self.apply(self._initialize_weights)
        self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _initialize_weights(module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x, targets=None):
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len)
            targets: Target tensor of shape (batch_size, seq_len)
            
        Returns:
            logits: Output logits of shape (batch_size, seq_len, vocab_size)
            loss: Loss value if targets is provided, otherwise None
        """
        if x.ndim != 2:
            raise ValueError("x must have shape (batch_size, seq_len)")
        if targets is not None and targets.shape != x.shape:
            raise ValueError("targets must have the same shape as x")

        _, seq_len = x.shape
        if seq_len > self.max_seq_length:
            raise ValueError(
                f"sequence length {seq_len} exceeds maximum "
                f"{self.max_seq_length}"
            )

        positions = torch.arange(seq_len, device=x.device)
        hidden = self.token_embedding(x)
        hidden = hidden + self.position_embedding(positions).unsqueeze(0)
        hidden = self.embedding_dropout(hidden)

        for block in self.blocks:
            hidden = block(hidden)

        logits = self.lm_head(self.final_norm(hidden))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
        return logits, loss
    
    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=True, top_k=None):
        """
        Generate text from the model.
        
        Args:
            idx: Input tensor of shape (batch_size, seq_len)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling
            do_sample: Whether to sample from the distribution
            top_k: If specified, only sample from the top k most likely tokens
            
        Returns:
            idx: Output tensor of shape (batch_size, seq_len + max_new_tokens)
        """
        if idx.ndim != 2 or idx.size(1) == 0:
            raise ValueError("idx must have shape (batch_size, seq_len) with seq_len > 0")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive")

        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                for _ in range(max_new_tokens):
                    context = idx[:, -self.max_seq_length:]
                    logits, _ = self(context)
                    next_logits = logits[:, -1, :] / temperature

                    if top_k is not None:
                        k = min(top_k, next_logits.size(-1))
                        cutoff = torch.topk(next_logits, k).values[:, -1, None]
                        next_logits = next_logits.masked_fill(
                            next_logits < cutoff, float("-inf")
                        )

                    probabilities = F.softmax(next_logits, dim=-1)
                    if do_sample:
                        next_token = torch.multinomial(probabilities, num_samples=1)
                    else:
                        next_token = torch.argmax(
                            probabilities, dim=-1, keepdim=True
                        )
                    idx = torch.cat((idx, next_token), dim=1)
        finally:
            self.train(was_training)

        return idx


DEFAULT_TRAINING_CONFIG = {
    "block_size": 128,
    "batch_size": 32,
    "d_model": 128,
    "nhead": 4,
    "num_layers": 3,
    "d_ff": 512,
    "dropout": 0.1,
    "learning_rate": 3e-4,
    "min_learning_rate": 3e-5,
    "warmup_steps": 100,
    "weight_decay": 0.01,
    "max_steps": 4000,
    "log_interval": 50,
}


def _select_device(requested_device=None):
    if requested_device is not None:
        return torch.device(requested_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_shakespeare_model(
    data_dir='./shakespeare', output_dir='./output', config=None
):
    """
    Train a transformer model on the Shakespeare dataset.
    
    Args:
        data_dir: Directory containing the Shakespeare text data
        output_dir: Directory to save the trained model
        
    Returns:
        model: Trained transformer model
    """
    training_config = DEFAULT_TRAINING_CONFIG.copy()
    if config is not None:
        training_config.update(config)

    device = _select_device(training_config.pop("device", None))
    print(f"Training on device: {device}")

    text = load_shakespeare_data(data_dir)
    dataset = ShakespeareDataset(
        text, block_size=training_config["block_size"]
    )
    dataloader = DataLoader(
        dataset,
        batch_size=training_config["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = TransformerModel(
        vocab_size=dataset.vocab_size,
        d_model=training_config["d_model"],
        nhead=training_config["nhead"],
        num_layers=training_config["num_layers"],
        d_ff=training_config["d_ff"],
        max_seq_length=training_config["block_size"],
        dropout=training_config["dropout"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
    )

    max_steps = training_config["max_steps"]
    warmup_steps = training_config["warmup_steps"]
    min_lr_ratio = (
        training_config["min_learning_rate"]
        / training_config["learning_rate"]
    )

    def learning_rate_multiplier(step):
        if step < warmup_steps:
            return max(step + 1, 1) / max(warmup_steps, 1)
        decay_steps = max(max_steps - warmup_steps, 1)
        progress = min((step - warmup_steps) / decay_steps, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=learning_rate_multiplier
    )

    model.train()
    data_iterator = iter(dataloader)
    final_loss = None
    running_loss = None
    for step in range(1, max_steps + 1):
        try:
            x, y = next(data_iterator)
        except StopIteration:
            data_iterator = iter(dataloader)
            x, y = next(data_iterator)

        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        final_loss = loss.item()
        running_loss = (
            final_loss
            if running_loss is None
            else 0.9 * running_loss + 0.1 * final_loss
        )

        if step == 1 or step % training_config["log_interval"] == 0:
            learning_rate = optimizer.param_groups[0]["lr"]
            print(
                f"step {step}/{max_steps} | loss {final_loss:.4f} | "
                f"avg {running_loss:.4f} | lr {learning_rate:.2e}"
            )

    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, "shakespeare_model.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "step": max_steps,
            "config": training_config,
            "stoi": dataset.stoi,
            "itos": dataset.itos,
            "final_training_loss": final_loss,
        },
        checkpoint_path,
    )
    print(f"Checkpoint saved to {checkpoint_path}")
    return model


def load_shakespeare_model(checkpoint_path, device=None):
    """Reconstruct a trained model and its metadata from a checkpoint."""
    target_device = _select_device(device)
    checkpoint = torch.load(checkpoint_path, map_location=target_device)
    config = checkpoint["config"]
    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]

    model = TransformerModel(
        vocab_size=len(stoi),
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        d_ff=config["d_ff"],
        max_seq_length=config["block_size"],
        dropout=config["dropout"],
    ).to(target_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    metadata = {
        "stoi": stoi,
        "itos": itos,
        "config": config,
        "step": checkpoint.get("step"),
        "final_training_loss": checkpoint.get("final_training_loss"),
    }
    return model, metadata

def generate_sample(
    model, dataset, prompt, max_new_tokens=500, temperature=0.8, top_k=10
):
    """
    Generate a text sample from the trained model.
    
    Args:
        model: Trained transformer model
        dataset: Shakespeare dataset
        prompt: Text prompt to start generation
        max_new_tokens: Maximum number of tokens to generate
        temperature: Temperature for sampling (higher = more random)
    """
    unknown = sorted(set(prompt) - set(dataset.stoi))
    if unknown:
        raise ValueError(f"prompt contains unknown characters: {unknown!r}")
    if not prompt:
        raise ValueError("prompt must not be empty")

    device = next(model.parameters()).device
    prompt_tokens = torch.tensor(
        [dataset.encode(prompt)], dtype=torch.long, device=device
    )
    generated = model.generate(
        prompt_tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        top_k=top_k,
    )[0]
    completion = dataset.decode(generated)
    print(f"\nGenerated text (prompt: {prompt!r}):")
    print(completion)
    return completion


def _build_argument_parser():
    parser = argparse.ArgumentParser(
        description="Train, validate, or generate with the Shakespeare Transformer."
    )
    parser.add_argument("--mode", choices=("train", "validate", "generate"))
    parser.add_argument("--checkpoint", default="./output/experiment_1/shakespeare_model.pt")
    parser.add_argument("--data-dir", default="./shakespeare")
    parser.add_argument("--output-dir", default="./output/experiment_1")
    parser.add_argument("--val-text", default="./output/val_text.txt")
    parser.add_argument("--reference-metrics", default="./output/reference_metrics.txt")
    parser.add_argument("--prompt", default="ROMEO:")
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--sample-output", default=None)
    parser.add_argument("--device", default=None)
    return parser


def main(argv=None):
    parser = _build_argument_parser()
    args = parser.parse_args(argv)
    if args.mode is None:
        parser.print_help()
        return

    if args.mode == "train":
        train_shakespeare_model(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            config={"device": args.device} if args.device else None,
        )
        return

    model, metadata = load_shakespeare_model(args.checkpoint, args.device)
    if args.mode == "validate":
        validation = ShakespeareValidation(
            val_text_path=args.val_text,
            ref_metrics_path=args.reference_metrics,
            data_dir=args.data_dir,
        )
        validation.validate_model(model, ShakespeareDataset)
        return

    training_text = load_shakespeare_data(args.data_dir)
    dataset = ShakespeareDataset(
        training_text,
        block_size=metadata["config"]["block_size"],
        stoi=metadata["stoi"],
        itos=metadata["itos"],
    )
    completion = generate_sample(
        model,
        dataset,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    if args.sample_output:
        sample_directory = os.path.dirname(args.sample_output)
        if sample_directory:
            os.makedirs(sample_directory, exist_ok=True)
        with open(args.sample_output, "w", encoding="utf-8") as sample_file:
            sample_file.write(
                f"Prompt: {args.prompt}\n"
                f"Temperature: {args.temperature}\n"
                f"Top-k: {args.top_k}\n"
                f"Generated tokens: {args.max_new_tokens}\n\n"
                f"{completion}\n"
            )
        print(f"Sample saved to {args.sample_output}")


if __name__ == "__main__":
    main()
