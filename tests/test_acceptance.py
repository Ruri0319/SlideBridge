from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from ibl2svs.acceptance import compare_preview_geometry, compare_roi_quality, summarize_wsi_output
from ibl2svs.converter import convert_file
from ibl2svs.models import ConvertOptions
from ibl2svs.reader import IBLSlide
from tests.support import create_sample_ibl


TIFFFILE_AVAILABLE = importlib.util.find_spec("tifffile") is not None


class AcceptanceTests(unittest.TestCase):
    def test_compare_preview_geometry_passes_on_same_mask(self) -> None:
        layer0 = Image.new("RGB", (8, 8), (255, 255, 255))
        layer1 = Image.new("RGB", (8, 8), (255, 255, 255))
        for image in (layer0, layer1):
            for x in range(2, 6):
                for y in range(1, 7):
                    image.putpixel((x, y), (120, 10, 10))

        result = compare_preview_geometry(layer0, layer1, background=255)

        self.assertTrue(result["passed"])
        self.assertAlmostEqual(result["iou"], 1.0)
        self.assertAlmostEqual(result["xor_ratio"], 0.0)

    def test_compare_roi_quality_passes_for_identical_roi(self) -> None:
        image = Image.new("RGB", (8, 8), (120, 10, 10))

        result = compare_roi_quality(image, image)

        self.assertTrue(result["passed"])
        self.assertEqual(result["mse"], 0.0)
        self.assertEqual(result["psnr"], float("inf"))
        self.assertAlmostEqual(result["ssim"], 1.0)

    def test_debug_export_roi_writes_file(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            sample = root / "sample.ibl"
            roi = root / "roi.png"
            create_sample_ibl(sample, img_width=16, img_height=16, tile_width=4, tile_height=4)

            with IBLSlide(sample) as slide:
                slide.debug_export_roi(roi, x=0, y=0, width=8, height=8)

            self.assertTrue(roi.exists())


@unittest.skipUnless(TIFFFILE_AVAILABLE, "tifffile is required for acceptance summary test")
class AcceptanceSummaryTests(unittest.TestCase):
    def test_summarize_wsi_output_reports_pages(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            sample = root / "sample.ibl"
            output = root / "sample.ome.tif"
            create_sample_ibl(sample, img_width=1024, img_height=1024, tile_width=256, tile_height=256)

            result = convert_file(sample, output, ConvertOptions(tile_size=16))
            summary = summarize_wsi_output(output)

            self.assertTrue(result.success)
            self.assertTrue(summary["exists"])
            self.assertGreaterEqual(summary["page_count"], 2)
            self.assertTrue(summary["pages"][0]["is_tiled"])

    def test_summarize_svs_reports_series_candidate_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            sample = root / "sample.ibl"
            output = root / "sample.svs"
            create_sample_ibl(
                sample,
                grid_cols=8,
                grid_rows=8,
                img_width=1024,
                img_height=1024,
                tile_width=256,
                tile_height=256,
            )

            result = convert_file(sample, output, ConvertOptions(tile_size=16, output_format="svs"))
            summary = summarize_wsi_output(output)

            self.assertTrue(result.success)
            self.assertIn("svs_thumbnail_dimensions", summary)
            self.assertIn("svs_pyramid_dimensions", summary)
            self.assertIn("svs_label_dimensions", summary)
            self.assertIn("svs_macro_dimensions", summary)
            self.assertIn("svs_is_bigtiff", summary)
            self.assertIn("svs_photometric_pages", summary)
            self.assertIn("svs_default_resolution_tag_pages", summary)
            self.assertIn("svs_extra_series_candidate_pages", summary)
            self.assertGreaterEqual(len(summary["svs_pyramid_dimensions"]), 2)
            self.assertIsNotNone(summary["svs_label_dimensions"])
            self.assertIsNotNone(summary["svs_macro_dimensions"])
            self.assertFalse(summary["svs_is_bigtiff"])
            self.assertEqual(summary["svs_default_resolution_tag_pages"], [])


if __name__ == "__main__":
    unittest.main()
