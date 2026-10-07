"""Tests for character tokenization and ShakespeareDataset."""

import unittest

import torch

from shakespeare_model import ShakespeareDataset


class ShakespeareDatasetTests(unittest.TestCase):
    def setUp(self):
        self.text = "cabca"
        self.dataset = ShakespeareDataset(self.text, block_size=3)

    def test_builds_sorted_character_vocabulary(self):
        self.assertEqual(self.dataset.stoi, {"a": 0, "b": 1, "c": 2})
        self.assertEqual(self.dataset.itos, {0: "a", 1: "b", 2: "c"})
        self.assertEqual(self.dataset.vocab_size, 3)

    def test_encode_decode_round_trip(self):
        tokens = self.dataset.encode(self.text)

        self.assertEqual(tokens, [2, 0, 1, 2, 0])
        self.assertEqual(self.dataset.decode(tokens), self.text)

    def test_returns_shifted_long_tensor_pairs(self):
        x, y = self.dataset[0]

        self.assertEqual(x.dtype, torch.long)
        self.assertEqual(y.dtype, torch.long)
        self.assertTrue(torch.equal(x, torch.tensor([2, 0, 1])))
        self.assertTrue(torch.equal(y, torch.tensor([0, 1, 2])))
        self.assertTrue(torch.equal(x[1:], y[:-1]))

    def test_length_counts_all_complete_windows(self):
        self.assertEqual(len(self.dataset), len(self.text) - 3)

    def test_reuses_supplied_vocabulary(self):
        stoi = {"c": 0, "b": 1, "a": 2}
        itos = {index: char for char, index in stoi.items()}

        dataset = ShakespeareDataset(
            "abc", block_size=2, stoi=stoi, itos=itos
        )

        self.assertIs(dataset.stoi, stoi)
        self.assertIs(dataset.itos, itos)
        self.assertEqual(dataset.encode("abc"), [2, 1, 0])

    def test_rejects_incomplete_vocabulary_arguments(self):
        with self.assertRaises(ValueError):
            ShakespeareDataset("abc", block_size=2, stoi={"a": 0})


if __name__ == "__main__":
    unittest.main()
