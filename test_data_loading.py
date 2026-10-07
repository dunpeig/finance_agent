"""Tests for the provided Shakespeare data-loading utilities."""

import tempfile
import unittest
from pathlib import Path

from shakespeare_setup import get_shakespeare_stats, load_shakespeare_data


class ShakespeareDataLoadingTests(unittest.TestCase):
    def test_loads_input_file_from_supplied_directory(self):
        expected = "To be, or not to be.\n"
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "input.txt").write_text(expected, encoding="utf-8")

            self.assertEqual(load_shakespeare_data(directory), expected)

    def test_stats_report_sorted_vocabulary(self):
        stats = get_shakespeare_stats("cabca")

        self.assertEqual(stats["total_chars"], 5)
        self.assertEqual(stats["vocab_size"], 3)
        self.assertEqual(stats["unique_chars"], ["a", "b", "c"])


if __name__ == "__main__":
    unittest.main()
