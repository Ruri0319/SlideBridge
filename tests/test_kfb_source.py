from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from ibl2svs.kfb_source import KfbSlideSource
from tests.support import create_sample_kfb


class KfbSlideSourceTests(unittest.TestCase):
    def test_reads_header_levels_and_region(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "sample.kfb"
            create_sample_kfb(path)

            with KfbSlideSource(path) as source:
                self.assertEqual(source.width, 24)
                self.assertEqual(source.height, 18)
                self.assertEqual(source.tile_count, 5)
                self.assertEqual(source.tile_size, 16)
                self.assertEqual(source.base_info.max_zoom_rate, 40)
                self.assertAlmostEqual(source.base_info.mpp, 0.25, places=5)

                region = source.read_region(14, 14, 8, 4)

            self.assertEqual(region.shape, (4, 8, 3))
            np.testing.assert_allclose(region[0, 0], [20, 30, 40], atol=3)
            np.testing.assert_allclose(region[0, 4], [80, 90, 100], atol=3)
            np.testing.assert_allclose(region[3, 0], [140, 150, 160], atol=3)
            np.testing.assert_allclose(region[3, 4], [200, 210, 220], atol=3)

    def test_returns_associated_images(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "sample.kfb"
            create_sample_kfb(path)

            with KfbSlideSource(path) as source:
                macro = source.get_macro_image()
                label = source.get_label_image()
                preview = source.get_preview_image()

            self.assertIsNotNone(macro)
            self.assertIsNotNone(label)
            self.assertIsNotNone(preview)
            self.assertEqual(macro.size, (64, 24))
            self.assertEqual(label.size, (32, 40))
            self.assertEqual(preview.size, (12, 9))

    def test_preview_falls_back_to_macro_without_stitching_main_level(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "sample.kfb"
            create_sample_kfb(path, include_preview_level=False)

            with KfbSlideSource(path) as source:
                source._stitch_level = lambda scale: self.fail("must not stitch the main-resolution level")
                preview = source.get_preview_image()

            self.assertIsNotNone(preview)
            self.assertEqual(preview.size, (64, 24))


if __name__ == "__main__":
    unittest.main()
