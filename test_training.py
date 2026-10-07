"""Small integration test for training and checkpoint creation."""

import tempfile
import unittest
from pathlib import Path

import torch

from shakespeare_model import (
    TransformerModel,
    load_shakespeare_model,
    train_shakespeare_model,
)


class TrainingTests(unittest.TestCase):
    def test_tiny_training_run_saves_complete_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory, "data")
            output_dir = Path(directory, "output")
            data_dir.mkdir()
            data_dir.joinpath("input.txt").write_text(
                "abcde" * 20, encoding="utf-8"
            )
            config = {
                "block_size": 4,
                "batch_size": 2,
                "d_model": 8,
                "nhead": 2,
                "num_layers": 1,
                "d_ff": 16,
                "dropout": 0.0,
                "learning_rate": 1e-3,
                "min_learning_rate": 1e-4,
                "warmup_steps": 1,
                "max_steps": 2,
                "log_interval": 1,
                "device": "cpu",
            }

            model = train_shakespeare_model(
                str(data_dir), str(output_dir), config=config
            )

            self.assertIsInstance(model, TransformerModel)
            checkpoint_path = output_dir / "shakespeare_model.pt"
            self.assertTrue(checkpoint_path.exists())
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            self.assertIn("model_state_dict", checkpoint)
            self.assertIn("optimizer_state_dict", checkpoint)
            self.assertIn("scheduler_state_dict", checkpoint)
            self.assertEqual(checkpoint["step"], 2)
            self.assertEqual(checkpoint["config"]["max_steps"], 2)
            self.assertEqual(len(checkpoint["stoi"]), 5)
            self.assertTrue(torch.isfinite(torch.tensor(
                checkpoint["final_training_loss"]
            )))

            loaded_model, metadata = load_shakespeare_model(
                str(checkpoint_path), device="cpu"
            )
            self.assertIsInstance(loaded_model, TransformerModel)
            self.assertFalse(loaded_model.training)
            self.assertEqual(metadata["step"], 2)
            self.assertEqual(metadata["stoi"], checkpoint["stoi"])


if __name__ == "__main__":
    unittest.main()
