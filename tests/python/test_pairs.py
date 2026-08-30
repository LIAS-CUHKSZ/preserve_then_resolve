from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dino_m2m.pairs import matching_filename, read_pairs, unique_images


class PairParserTests(unittest.TestCase):
    def _pairs(self, text: str, *, suffix: str = ".txt", **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / f"pairs{suffix}"
            path.write_text(text, encoding="utf-8")
            return read_pairs(path, **kwargs)

    def test_supported_rows_and_sorting(self) -> None:
        pairs = self._pairs(
            "# comment\n"
            "9 b.jpg c.jpg 0.5\n"
            "a.jpg b.jpg\n"
            "4\td.jpg\te.jpg\tmetadata\n"
        )
        self.assertEqual([pair.pair_index for pair in pairs], [1, 4, 9])
        self.assertEqual(str(pairs[0].left_rel), "a.jpg")
        self.assertEqual([str(path) for path in unique_images(pairs)], [
            "a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg"
        ])

    def test_duplicate_explicit_and_auto_id_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Duplicate pair index"):
            self._pairs("a.jpg b.jpg\n1 c.jpg d.jpg\n")

    def test_csv_header_and_metadata(self) -> None:
        pairs = self._pairs(
            "image_1,image_2,angular_distance_degrees\n"
            "object/a.jpg,object/b.jpg,12.5\n"
            "object/c.jpg,object/d.jpg,23.0\n",
            suffix=".csv",
        )
        self.assertEqual([pair.pair_index for pair in pairs], [1, 2])
        self.assertEqual(str(pairs[0].left_rel), "object/a.jpg")
        self.assertEqual(str(pairs[1].right_rel), "object/d.jpg")

    def test_max_pairs_and_matching_filename(self) -> None:
        pairs = self._pairs("a b\nc d\n", max_pairs=1)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(matching_filename(7, 4), "matching_0007.csv")


if __name__ == "__main__":
    unittest.main()
