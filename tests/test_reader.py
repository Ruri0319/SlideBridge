from __future__ import annotations
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ibl2svs.reader import IBLSlide

from tests.support import create_sample_ibl


class ReaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = Path(self.tempdir.name) / "sample.ibl"
        create_sample_ibl(self.path)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_reader_parses_metadata_and_block(self) -> None:
        with IBLSlide(self.path) as slide:
            self.assertEqual(slide.width, 8)
            self.assertEqual(slide.height, 8)
            self.assertAlmostEqual(slide.base_info.mpp, 0.25)
            block = slide.get_block_array(slide.blocks[0])
            self.assertEqual(block.shape, (8, 8, 3))
            self.assertEqual(slide.blocks[0].x1, 8)
            self.assertEqual(slide.blocks[0].y1, 8)

    def test_reader_builds_thumbnail(self) -> None:
        with IBLSlide(self.path) as slide:
            thumbnail = slide.get_thumbnail_image()
            self.assertIsNotNone(thumbnail)
            self.assertEqual(thumbnail.size, (1, 1))
            preview0 = slide.assemble_preview_from_layer0()
            preview1 = slide.assemble_preview_from_layer1()
            self.assertEqual(preview0.size, (2, 2))
            self.assertEqual(preview1.size, (2, 2))

    def test_reader_rejects_incomplete_tiles(self) -> None:
        broken = Path(self.tempdir.name) / "broken.ibl"
        create_sample_ibl(broken, omit_tile=(3, 3))
        with self.assertRaisesRegex(RuntimeError, "不完整"):
            IBLSlide(broken)

    def test_blocks_for_region_uses_actual_coordinates_not_grid(self) -> None:
        overlap = Path(self.tempdir.name) / "overlap.ibl"
        create_sample_ibl(
            overlap,
            grid_cols=12,
            grid_rows=1,
            img_width=8,
            img_height=8,
            tile_width=2,
            tile_height=2,
            step_x=4,
            include_preview=False,
            include_shrink=False,
        )
        with IBLSlide(overlap) as slide:
            blocks = slide.blocks_for_region(39, 0, 2, 8)
            self.assertEqual([block.grid_col for block in blocks], [8, 9, 10])

    def test_read_region_uses_overlap_ownership_boundaries(self) -> None:
        overlap = Path(self.tempdir.name) / "ownership.ibl"
        create_sample_ibl(
            overlap,
            grid_cols=3,
            grid_rows=1,
            img_width=8,
            img_height=8,
            tile_width=2,
            tile_height=2,
            step_x=6,
            include_preview=False,
            include_shrink=False,
        )
        with IBLSlide(overlap) as slide:
            region = slide.read_region(0, 0, slide.width, slide.height, decode_workers=1)
            left = slide.blocks_by_grid[(0, 0)]
            middle = slide.blocks_by_grid[(1, 0)]
            right = slide.blocks_by_grid[(2, 0)]
            left_block = slide.get_block_array(left)
            middle_block = slide.get_block_array(middle)
            right_block = slide.get_block_array(right)

            np.testing.assert_array_equal(region[4, 6], left_block[4, 6])
            np.testing.assert_array_equal(region[4, 7], middle_block[4, 1])
            np.testing.assert_array_equal(region[4, 12], middle_block[4, 6])
            np.testing.assert_array_equal(region[4, 13], right_block[4, 1])


if __name__ == "__main__":
    unittest.main()
