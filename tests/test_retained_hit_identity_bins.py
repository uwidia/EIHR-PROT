from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from reliability_aware.utils.diamond_homology import DiamondSearchConfig
from reliability_aware.utils.retained_hit_identity_bins import (
    category_for_identity,
    create_retained_hit_identity_bins,
)


class RetainedHitIdentityBinsTest(unittest.TestCase):
    def test_boundaries_are_unrounded_and_mutually_exclusive(self) -> None:
        self.assertEqual(category_for_identity(None), "no_retained_hit")
        self.assertEqual(category_for_identity(29.999), "identity_lt30")
        self.assertEqual(category_for_identity(30.0), "identity_30_lt40")
        self.assertEqual(category_for_identity(40.0), "identity_40_lt50")
        self.assertEqual(category_for_identity(50.0), "identity_50_lt70")
        self.assertEqual(category_for_identity(70.0), "identity_70_lt95")
        self.assertEqual(category_for_identity(95.0), "identity_ge95")
        self.assertEqual(category_for_identity(100.0), "identity_ge95")

    def test_outputs_use_only_final_retained_hits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            test_fasta = root / "test.fasta"
            train_fasta = root / "train.fasta"
            hits = root / "hits.tsv"
            output_dir = root / "output"
            test_fasta.write_text(
                ">q30\nAAAA\n>q40\nAAAA\n>q50\nAAAA\n>q70\nAAAA\n"
                ">q95\nAAAA\n>qnone\nAAAA\n",
                encoding="utf-8",
            )
            train_fasta.write_text(
                ">t1\nAAAA\n>t2\nAAAA\n>t3\nAAAA\n>t4\nAAAA\n"
                ">t5\nAAAA\n>t_unretained\nAAAA\n",
                encoding="utf-8",
            )
            # q30's 99% row is excluded by coverage. q40's duplicate t2 row
            # retains the higher-bitscore 40% alignment before top-K selection.
            hits.write_text(
                "q30\tt1\t1e-10\t10\t100\t4\t4\t4\t30\n"
                "q30\tt_unretained\t1e-10\t99\t20\t4\t4\t4\t99\n"
                "q40\tt2\t1e-10\t10\t100\t4\t4\t4\t39\n"
                "q40\tt2\t1e-10\t20\t100\t4\t4\t4\t40\n"
                "q50\tt3\t1e-10\t10\t100\t4\t4\t4\t50\n"
                "q70\tt4\t1e-10\t10\t100\t4\t4\t4\t70\n"
                "q95\tt5\t1e-10\t10\t100\t4\t4\t4\t95\n",
                encoding="utf-8",
            )
            result = create_retained_hit_identity_bins(
                test_fasta=test_fasta,
                train_fasta=train_fasta,
                hits_tsv=hits,
                output_dir=output_dir,
                config=DiamondSearchConfig(top_k=10),
                command=["test"],
            )
            categories = {row["test_protein_id"]: row["category"] for row in result["rows"]}
            self.assertEqual(
                categories,
                {
                    "q30": "identity_30_lt40",
                    "q40": "identity_40_lt50",
                    "q50": "identity_50_lt70",
                    "q70": "identity_70_lt95",
                    "q95": "identity_ge95",
                    "qnone": "no_retained_hit",
                },
            )
            with (output_dir / "assignments.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            qnone = next(row for row in rows if row["test_protein_id"] == "qnone")
            self.assertEqual(qnone["max_retained_identity"], "")
            self.assertTrue(result["validation"]["categories_union_equals_test_set"])
            self.assertTrue((output_dir / "identity_lt30.fasta").is_file())

    def test_duplicate_test_fasta_ids_are_rejected_before_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            test_fasta = root / "test.fasta"
            train_fasta = root / "train.fasta"
            hits = root / "hits.tsv"
            test_fasta.write_text(
                ">duplicate\nAAAA\n>duplicate\nCCCC\n", encoding="utf-8"
            )
            train_fasta.write_text(">target\nAAAA\n", encoding="utf-8")
            hits.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate protein IDs"):
                create_retained_hit_identity_bins(
                    test_fasta=test_fasta,
                    train_fasta=train_fasta,
                    hits_tsv=hits,
                    output_dir=root / "output",
                    config=DiamondSearchConfig(top_k=10),
                    command=["test"],
                )


if __name__ == "__main__":
    unittest.main()
