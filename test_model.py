"""Focused tests for the decoder-only Transformer model."""

import unittest

import torch

from shakespeare_model import TransformerModel


class TransformerModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.vocab_size = 11
        self.model = TransformerModel(
            vocab_size=self.vocab_size,
            d_model=16,
            nhead=4,
            num_layers=2,
            d_ff=32,
            max_seq_length=8,
            dropout=0.0,
        )

    def test_forward_returns_expected_shapes_and_loss(self):
        x = torch.randint(self.vocab_size, (2, 6))
        targets = torch.randint(self.vocab_size, (2, 6))

        logits, loss = self.model(x, targets)

        self.assertEqual(logits.shape, (2, 6, self.vocab_size))
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))

    def test_targets_are_optional(self):
        logits, loss = self.model(torch.randint(self.vocab_size, (2, 4)))

        self.assertEqual(logits.shape, (2, 4, self.vocab_size))
        self.assertIsNone(loss)

    def test_loss_backpropagates_to_model_parameters(self):
        x = torch.randint(self.vocab_size, (2, 6))
        targets = torch.randint(self.vocab_size, (2, 6))

        _, loss = self.model(x, targets)
        loss.backward()

        self.assertIsNotNone(self.model.token_embedding.weight.grad)
        self.assertTrue(torch.isfinite(self.model.token_embedding.weight.grad).all())

    def test_rejects_sequences_longer_than_context_window(self):
        x = torch.randint(self.vocab_size, (1, 9))

        with self.assertRaises(ValueError):
            self.model(x)

    def test_rejects_incompatible_head_dimension(self):
        with self.assertRaises(ValueError):
            TransformerModel(11, 15, 4, 1, 32, 8)

    def test_future_tokens_do_not_change_earlier_logits(self):
        self.model.eval()
        first = torch.tensor([[1, 2, 3, 4, 5, 6]])
        second = torch.tensor([[1, 2, 3, 9, 8, 7]])

        first_logits, _ = self.model(first)
        second_logits, _ = self.model(second)

        self.assertTrue(torch.allclose(
            first_logits[:, :3], second_logits[:, :3], atol=1e-6
        ))

    def test_language_head_and_token_embedding_share_weights(self):
        self.assertIs(
            self.model.lm_head.weight,
            self.model.token_embedding.weight,
        )


if __name__ == "__main__":
    unittest.main()
