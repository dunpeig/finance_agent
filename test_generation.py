"""Focused tests for autoregressive text generation."""

import unittest

import torch

from shakespeare_model import (
    ShakespeareDataset,
    TransformerModel,
    generate_sample,
)


class GenerationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.dataset = ShakespeareDataset("abcabcabc", block_size=3)
        self.model = TransformerModel(3, 8, 2, 1, 16, 3, dropout=0.0)

    def test_generate_appends_requested_number_of_tokens(self):
        prompt = torch.tensor([[0, 1]])
        generated = self.model.generate(prompt, 5, do_sample=False)

        self.assertEqual(generated.shape, (1, 7))
        self.assertTrue(torch.equal(generated[:, :2], prompt))

    def test_generation_crops_context_to_model_window(self):
        prompt = torch.tensor([[0, 1, 2]])
        generated = self.model.generate(prompt, 6, do_sample=False)

        self.assertEqual(generated.shape, (1, 9))

    def test_greedy_generation_is_deterministic(self):
        prompt = torch.tensor([[0]])
        first = self.model.generate(prompt, 4, do_sample=False)
        second = self.model.generate(prompt, 4, do_sample=False)

        self.assertTrue(torch.equal(first, second))

    def test_generate_restores_training_mode(self):
        self.model.train()
        self.model.generate(torch.tensor([[0]]), 1)

        self.assertTrue(self.model.training)

    def test_rejects_invalid_sampling_parameters(self):
        prompt = torch.tensor([[0]])
        with self.assertRaises(ValueError):
            self.model.generate(prompt, 1, temperature=0)
        with self.assertRaises(ValueError):
            self.model.generate(prompt, 1, top_k=0)

    def test_generate_sample_returns_decoded_prompt_and_completion(self):
        completion = generate_sample(
            self.model, self.dataset, "ab", max_new_tokens=3
        )

        self.assertEqual(len(completion), 5)
        self.assertTrue(completion.startswith("ab"))


if __name__ == "__main__":
    unittest.main()
