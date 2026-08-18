from __future__ import annotations
import json
import tempfile
import unittest
import sqlite3
from pathlib import Path

import numpy as np

from ibl2svs.reader import IBLSlide

from tests.support import create_sample_ibl, jpeg_bytes


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

    def test_reader_uses_tile_geometry_for_expected_tile_count(self) -> None:
        self.path.unlink()
        create_sample_ibl(
            self.path,
            img_width=6,
            img_height=4,
            tile_width=4,
            tile_height=3,
            include_preview=False,
            include_shrink=False,
        )

        with IBLSlide(self.path) as slide:
            block = slide.get_block_array(slide.blocks[0])
            self.assertEqual(block.shape, (4, 6, 3))

    def test_reader_uses_native_ext_images_and_scan_metadata(self) -> None:
        self.path.unlink()
        create_sample_ibl(self.path, include_preview=False, include_shrink=False)
        with sqlite3.connect(self.path) as connection:
            connection.executescript(
                """
                CREATE TABLE tbl_ext_info (
                    id INTEGER UNIQUE,
                    type INTEGER,
                    data BLOB,
                    PRIMARY KEY(id AUTOINCREMENT)
                );
                CREATE TABLE tbl_user_info (
                    id INTEGER UNIQUE,
                    deviceNo TEXT,
                    slideSource TEXT,
                    slideType INTEGER,
                    scanMode INTEGER,
                    scanLayer INTEGER,
                    scanTime INTEGER,
                    PRIMARY KEY(id AUTOINCREMENT)
                );
                """
            )
            connection.executemany(
                "INSERT INTO tbl_ext_info(type, data) VALUES (?, ?)",
                [
                    (1, jpeg_bytes((10, 20, 30), (6, 2))),
                    (2, jpeg_bytes((40, 50, 60), (3, 1))),
                    (3, jpeg_bytes((70, 80, 90), (2, 5))),
                    (6, json.dumps({"deviceNo": "EXT-01", "packetTime": "2024/04/12 09:43:56", "scanTime": 12.5}).encode("utf-8")),
                ],
            )
            connection.execute(
                "INSERT INTO tbl_user_info VALUES (?, ?, ?, ?, ?, ?, ?)",
                (1, "USER-01", "", 0, 0, 1, 123),
            )

        with IBLSlide(self.path) as slide:
            self.assertEqual(slide.get_macro_image().size, (6, 2))
            self.assertEqual(slide.get_thumbnail_image().size, (3, 1))
            self.assertEqual(slide.get_label_image().size, (2, 5))
            self.assertIsNone(slide.get_overview_image())
            metadata = slide.get_scan_metadata()
            self.assertEqual(metadata["deviceNo"], "EXT-01")
            self.assertEqual(metadata["scanTime"], "2024/04/12 09:43:56")
            self.assertEqual(metadata["scanDuration"], 12.5)
            self.assertEqual(metadata["userScanTime"], 123)

    def test_reader_uses_legacy_airimg_info_as_label(self) -> None:
        self.path.unlink()
        create_sample_ibl(self.path, include_preview=False, include_shrink=False)
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "CREATE TABLE tbl_airimg_info (id INTEGER PRIMARY KEY, data BLOB)"
            )
            connection.execute(
                "INSERT INTO tbl_airimg_info(id, data) VALUES (?, ?)",
                (1, jpeg_bytes((11, 22, 33), (7, 4))),
            )

        with IBLSlide(self.path) as slide:
            label = slide.get_label_image()
            self.assertIsNotNone(label)
            self.assertEqual(label.size, (7, 4))
            self.assertTrue(
                all(
                    abs(actual - expected) <= 2
                    for actual, expected in zip(label.getpixel((0, 0)), (11, 22, 33))
                )
            )

    def test_reader_rejects_incomplete_tiles(self) -> None:
        broken = Path(self.tempdir.name) / "broken.ibl"
        create_sample_ibl(broken, omit_tile=(3, 3))
        with self.assertRaisesRegex(RuntimeError, "不完整"):
            IBLSlide(broken)

    def test_reader_rejects_block_with_no_full_resolution_tiles(self) -> None:
        broken = Path(self.tempdir.name) / "missing-block.ibl"
        create_sample_ibl(broken)
        with sqlite3.connect(broken) as connection:
            connection.execute(
                "INSERT INTO tbl_img_info VALUES (?, 0, 0, 0, 0, 0, ?, ?, ?, ?)",
                (1, 1, 0, 0, 0),
            )

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
