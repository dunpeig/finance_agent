"""Tests for the explicit command-line modes."""

import io
import unittest
from contextlib import redirect_stdout

from shakespeare_model import _build_argument_parser, main


class CommandLineTests(unittest.TestCase):
    def test_no_mode_prints_help_instead_of_training(self):
        output = io.StringIO()

        with redirect_stdout(output):
            main([])

        self.assertIn("--mode", output.getvalue())

    def test_generate_defaults_to_best_experiment_checkpoint(self):
        args = _build_argument_parser().parse_args(["--mode", "generate"])

        self.assertEqual(
            args.checkpoint,
            "./output/experiment_1/shakespeare_model.pt",
        )
        self.assertEqual(args.prompt, "ROMEO:")
        self.assertEqual(args.max_new_tokens, 500)


if __name__ == "__main__":
    unittest.main()
