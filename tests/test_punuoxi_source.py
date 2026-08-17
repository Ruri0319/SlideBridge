from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ibl2svs.punuoxi_source import PunuoxiFormatError, PunuoxiImageSource, PunuoxiLevel
from tests.support import create_sample_image


class PunuoxiImageSourceTests(unittest.TestCase):
    def test_real_vendor_samples_expose_native_resources_and_levels(self) -> None:
        samples = [Path(__file__).resolve().parents[1] / "punuoxi" / name for name in ("6.image", "8.image")]
        if not all(path.exists() for path in samples):
            self.skipTest("real Punuoxi samples are not available")

        for path in samples:
            with PunuoxiImageSource(path) as source:
                self.assertEqual(source.native_resource_status, "native")
                self.assertEqual(len(source.levels), 8)
                self.assertEqual(source.get_macro_image().size, (1152, 625))
                self.assertEqual(source.get_label_image().size, (300, 294))
                self.assertEqual(source.get_thumbnail_image().width, 300)
                for level_index in range(len(source.levels)):
                    tile = next(tile for tile in source.iter_native_level_jpegs(level_index) if tile is not None)
                    self.assertGreater(len(tile), 4)

    def test_reads_image_without_memory_mapping_the_file(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "sample.image"
            create_sample_image(path)

            with patch("mmap.mmap", side_effect=AssertionError("must not mmap IMAGE files")):
                with PunuoxiImageSource(path) as source:
                    region = source.read_region(0, 0, 16, 16)

            self.assertEqual(region.shape, (16, 16, 3))

    def test_reads_header_levels_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "sample.image"
            create_sample_image(path)

            with PunuoxiImageSource(path) as source:
                self.assertEqual(source.width, 512)
                self.assertEqual(source.height, 512)
                self.assertEqual(source.shape, (512, 512, 3))
                self.assertEqual(source.level_grids, [(2, 2), (1, 1)])
                self.assertEqual(source.level_dimensions, [(512, 512), (256, 256)])
                self.assertAlmostEqual(source.base_info.mpp, 0.3466053, places=5)
                metadata = source.get_scan_metadata()

            self.assertEqual(metadata["institution"], "测试医院")
            self.assertEqual(metadata["caseNo"], "CASE-001")
            self.assertEqual(metadata["deviceNo"], "TEST-PUNUOXI")
            self.assertEqual(metadata["scanTime"], "2026-01-01 12:34:56")

    def test_reads_native_rgb_resources_and_native_level_jpegs(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "native.image"
            create_sample_image(path, include_native_resources=True)

            with PunuoxiImageSource(path) as source:
                self.assertEqual(source.native_resource_status, "native")
                self.assertEqual(source.get_thumbnail_image().size, (300, 5))
                self.assertEqual(source.get_macro_image().size, (1152, 625))
                self.assertEqual(source.get_label_image().size, (300, 294))
                self.assertEqual(source.get_thumbnail_image().getpixel((0, 0)), (10, 20, 30))
                self.assertEqual(source.get_macro_image().getpixel((0, 0)), (40, 50, 60))
                self.assertEqual(source.get_label_image().getpixel((0, 0)), (70, 80, 90))
                native_tiles = list(source.iter_native_level_jpegs(0))

            self.assertEqual(len(native_tiles), 4)
            self.assertTrue(all(tile is not None for tile in native_tiles))

    def test_invalid_native_resource_pointers_use_legacy_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "fallback.image"
            create_sample_image(path)
            data = bytearray(path.read_bytes())
            data[0:8] = (200).to_bytes(8, "little")
            data[8:16] = (100).to_bytes(8, "little")
            path.write_bytes(data)

            with PunuoxiImageSource(path) as source:
                self.assertEqual(source.native_resource_status, "legacy_fallback")
                self.assertIn("偏移", source.native_resource_reason)
                self.assertIsNotNone(source.get_preview_image())

    def test_reads_region_across_tiles_and_pads_outside(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "sample.image"
            create_sample_image(path)

            with PunuoxiImageSource(path) as source:
                region = source.read_region(250, 250, 20, 20)
                outside = source.read_region(-4, -4, 8, 8)

            self.assertEqual(region.shape, (20, 20, 3))
            np.testing.assert_allclose(region[0, 0], [30, 40, 50], atol=4)
            np.testing.assert_allclose(region[0, 10], [80, 90, 100], atol=4)
            np.testing.assert_allclose(region[10, 0], [130, 140, 150], atol=4)
            np.testing.assert_allclose(region[10, 10], [180, 190, 200], atol=4)
            np.testing.assert_array_equal(outside[:4, :4], 250)

    def test_treats_zero_length_placeholder_tile_as_background(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "sparse.image"
            create_sample_image(path, empty_tiles={(0, 0, 1)})

            with PunuoxiImageSource(path) as source:
                record = source.levels[0].records[2]
                region = source.read_region(0, 256, 16, 16)

            self.assertEqual((record.column, record.row, record.jpeg_length), (0, 1, 0))
            np.testing.assert_array_equal(region, 250)

    def test_normalizes_column_major_index_to_row_major_coordinates(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "sample.image"
            create_sample_image(path)

            with PunuoxiImageSource(path) as source:
                records = source.levels[0].records

            self.assertEqual(
                [(record.column, record.row) for record in records],
                [(0, 0), (1, 0), (0, 1), (1, 1)],
            )
            self.assertEqual([record.index for record in records], [0, 2, 1, 3])

    def test_restores_transposed_pixel_axes_inside_each_tile(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "directional.image"
            create_sample_image(path, directional_tile=True)

            with PunuoxiImageSource(path) as source:
                tile = source.read_region(0, 0, 256, 256)

            np.testing.assert_allclose(tile[64, 64], [220, 30, 40], atol=5)
            np.testing.assert_allclose(tile[64, 192], [30, 210, 50], atol=5)
            np.testing.assert_allclose(tile[192, 64], [40, 60, 220], atol=5)
            np.testing.assert_allclose(tile[192, 192], [220, 210, 40], atol=5)

    def test_stitches_low_resolution_preview_without_reading_main_canvas(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "sample.image"
            create_sample_image(path)

            with PunuoxiImageSource(path) as source:
                source.read_region = lambda *args, **kwargs: self.fail("preview must use indexed level")
                preview = source.get_preview_image()

            self.assertIsNotNone(preview)
            self.assertEqual(preview.size, (256, 256))
            self.assertEqual(preview.getpixel((0, 0)), (100, 110, 120))

    def test_selects_largest_preview_level_within_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "sample.image"
            create_sample_image(path)

            with PunuoxiImageSource(path) as source:
                source._levels = [
                    PunuoxiLevel(index, side, side, 256, 0, ())
                    for index, side in enumerate((16, 8, 4, 2))
                ]
                selected = source._select_preview_level()

            self.assertEqual(selected.index, 1)
            self.assertEqual(selected.dimensions, (2048, 2048))

    def test_rejects_device_number_that_overruns_header_field(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "broken.image"
            create_sample_image(path)
            data = bytearray(path.read_bytes())
            data[0x53:0x57] = (46).to_bytes(4, "little")
            path.write_bytes(data)

            with self.assertRaisesRegex(PunuoxiFormatError, "设备号长度无效"):
                PunuoxiImageSource(path)

    def test_rejects_invalid_tail_identifier(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "broken.image"
            create_sample_image(path)
            data = bytearray(path.read_bytes())
            data[-1] = 1
            path.write_bytes(data)

            with self.assertRaises(PunuoxiFormatError):
                PunuoxiImageSource(path)


if __name__ == "__main__":
    unittest.main()
