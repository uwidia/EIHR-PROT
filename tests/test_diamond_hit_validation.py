import tempfile
import unittest
from pathlib import Path

from reliability_aware.utils.diamond_homology import _parse_diamond_hits


class DiamondHitValidationTests(unittest.TestCase):
    def parse(self, contents):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hits.tsv"
            path.write_text(contents)
            return _parse_diamond_hits(path)

    def test_corrupt_subject_length_reports_source_row(self):
        with self.assertRaises(ValueError) as caught:
            self.parse("\n2VAU-A\t1XX8-A\t1.25e-33\t125\t98.2\t331\t66\t354\t28.0\t99\n")
        message = str(caught.exception)
        for detail in ("hits.tsv: line 2", "2VAU-A/1XX8-A", "qlen=331", "slen=66", "nident=99", "--iterate"):
            self.assertIn(detail, message)

    def test_exact_count_boundaries_and_legacy_rows(self):
        for count in (0, 66, None):
            with self.subTest(count=count):
                row = "q\ts\t1e-9\t100\t100\t100\t66\t66\t100"
                if count is not None:
                    row += f"\t{count}"
                self.assertEqual(self.parse(row + "\n")["q"][0].nident, count)

    def test_invalid_counts_and_lengths_still_fail(self):
        for qlen, slen, count in ((100, 66, "66.0"), (100, 66, "-1"), (100, 66, "67"), (0, 66, "0"), (100, 0, "0")):
            with self.subTest(qlen=qlen, slen=slen, count=count):
                with self.assertRaisesRegex(ValueError, r"hits.tsv: line 1.*q/s"):
                    self.parse(f"q\ts\t1e-9\t100\t100\t{qlen}\t{slen}\t66\t100\t{count}\n")

    def test_wrong_column_count_reports_line(self):
        with self.assertRaisesRegex(ValueError, r"hits.tsv: line 2.*got 2"):
            self.parse("\nq\ts\n")


if __name__ == "__main__":
    unittest.main()
