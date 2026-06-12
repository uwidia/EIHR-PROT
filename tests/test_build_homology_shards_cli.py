from __future__ import annotations

import unittest

from scripts import build_homology_shards


class BuildHomologyShardsCliTest(unittest.TestCase):
    def test_defaults_to_all_splits(self) -> None:
        args = build_homology_shards.parse_args(["--go_aspect", "BP"])
        self.assertEqual(args.splits, ["train", "val", "test"])

    def test_can_select_test_only(self) -> None:
        args = build_homology_shards.parse_args(
            ["--go_aspect", "MF", "--splits", "test"]
        )
        self.assertEqual(args.splits, ["test"])

    def test_can_select_multiple_splits(self) -> None:
        args = build_homology_shards.parse_args(
            ["--go_aspect", "CC", "--splits", "train", "val"]
        )
        self.assertEqual(args.splits, ["train", "val"])


if __name__ == "__main__":
    unittest.main()
