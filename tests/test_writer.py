from __future__ import annotations

import importlib.util
from unittest import mock
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ibl2svs.converter import convert_file
from ibl2svs.models import ConvertOptions
from ibl2svs.punuoxi_source import PunuoxiImageSource
from ibl2svs.tiff_source import TiffSlideSource
from ibl2svs.writer import (
    _build_label_array,
    _compute_svs_pyramid_shapes,
    finalize_svs_with_libtiff,
    write_image,
    WriteImageError,
)
from tests.support import create_sample_ibl, create_sample_image


TIFFFILE_AVAILABLE = importlib.util.find_spec("tifffile") is not None

if TIFFFILE_AVAILABLE:
    import tifffile

try:
    if importlib.util.find_spec("openslide") is None:
        OPENSLIDE_AVAILABLE = False
    else:
        import openslide

        OPENSLIDE_AVAILABLE = True
except ImportError:
    OPENSLIDE_AVAILABLE = False


def _page_mpp(page) -> tuple[float, float]:
    x_value = page.tags["XResolution"].value
    y_value = page.tags["YResolution"].value
    x_resolution = x_value[0] / x_value[1]
    y_resolution = y_value[0] / y_value[1]
    return 10000.0 / x_resolution, 10000.0 / y_resolution


@unittest.skipUnless(TIFFFILE_AVAILABLE, "tifffile is required for writer tests")
class WriterTests(unittest.TestCase):
    def test_finalize_svs_with_libtiff_gracefully_handles_missing_tiffset(self) -> None:
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
            result = convert_file(
                sample,
                output,
                ConvertOptions(tile_size=16, output_format="svs", svs_finalize_with_libtiff=False, svs_validate_with_tiffinfo=False),
            )
            self.assertTrue(result.success)
            with mock.patch("ibl2svs.writer.subprocess.run", side_effect=FileNotFoundError()):
                self.assertEqual(finalize_svs_with_libtiff(output), "unavailable")

    def test_sparse_svs_helpers_use_true_downsample(self) -> None:
        class SlideStub:
            width = 158159
            height = 61614

        options = ConvertOptions(output_format="svs")
        shapes = _compute_svs_pyramid_shapes(SlideStub(), options)

        self.assertEqual(shapes, [(39539, 15403), (19769, 7701), (9884, 3850), (4942, 1925), (2471, 962)])

    def test_write_image_raises_on_streaming_direct_error(self) -> None:
        class SlideStub:
            width = 158159
            height = 61614

        with tempfile.TemporaryDirectory() as tempdir:
            output = Path(tempdir) / "sample.svs"
            with mock.patch("ibl2svs.writer._write_svs_streaming_direct", side_effect=RuntimeError("mock failure")):
                with self.assertRaises(WriteImageError) as ctx:
                    write_image(
                        SlideStub(),
                        output,
                        ConvertOptions(
                            output_format="svs",
                            svs_finalize_with_libtiff=False,
                            svs_validate_with_tiffinfo=False,
                        ),
                    )
        self.assertIn("mock failure", str(ctx.exception))

    def test_convert_file_writes_generic_tiff_and_reportable_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            sample = root / "sample.ibl"
            output = root / "sample.tif"
            create_sample_ibl(sample, img_width=512, img_height=512, tile_width=128, tile_height=128)

            result = convert_file(sample, output, ConvertOptions(tile_size=16))

            self.assertTrue(result.success)
            self.assertTrue(output.exists())
            self.assertGreaterEqual(result.pyramid_levels or 0, 2)
            self.assertEqual(result.output_format, "generic_tiff")
            self.assertEqual(result.backend, "tifffile-streaming")
            self.assertEqual(result.level_dimensions, [(512, 512), (256, 256)])
            self.assertGreaterEqual(result.read_decode_sec, 0.0)
            self.assertGreaterEqual(result.main_write_sec, 0.0)
            self.assertGreaterEqual(result.pyramid_sec, 0.0)
            self.assertGreaterEqual(result.peak_memory_mb, 0.0)
            self.assertGreaterEqual(result.avg_cpu_percent, 0.0)
            with tifffile.TiffFile(output) as tif:
                self.assertGreaterEqual(len(tif.pages), 2)
                self.assertEqual(tif.pages[0].shape, (512, 512, 3))
                self.assertTrue(tif.pages[0].is_tiled)
                self.assertIn("AppMag = 40", tif.pages[0].description or "")
                self.assertIn("ObjectivePower = 40", tif.pages[0].description or "")
                self.assertEqual(tif.pages[0].tags["ResolutionUnit"].value, 3)
                mpp_x, mpp_y = _page_mpp(tif.pages[0])
                self.assertAlmostEqual(mpp_x, 0.25, places=4)
                self.assertAlmostEqual(mpp_y, 0.25, places=4)
                self.assertEqual(tif.pages[1].shape, (256, 256, 3))
                self.assertTrue(tif.pages[1].is_tiled)
                self.assertEqual(tif.pages[1].subfiletype, 1)
                level_mpp_x, level_mpp_y = _page_mpp(tif.pages[1])
                self.assertAlmostEqual(level_mpp_x, 0.5, places=4)
                self.assertAlmostEqual(level_mpp_y, 0.5, places=4)
            with TiffSlideSource(output) as source:
                self.assertEqual(source.base_info.max_zoom_rate, 40)

    def test_image_conversion_preserves_restored_tile_orientation(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            sample = root / "directional.image"
            output = root / "directional.tif"
            create_sample_image(sample, directional_tile=True)

            result = convert_file(sample, output, ConvertOptions(tile_size=16))

            self.assertTrue(result.success, result.error)
            with TiffSlideSource(output) as source:
                tile = source.read_region(0, 0, 256, 256)
            self.assertAlmostEqual(int(tile[64, 64, 0]), 220, delta=8)
            self.assertAlmostEqual(int(tile[64, 192, 1]), 210, delta=8)
            self.assertAlmostEqual(int(tile[192, 64, 2]), 220, delta=8)
            self.assertAlmostEqual(int(tile[192, 192, 0]), 220, delta=8)

    def test_image_conversion_renders_empty_tiles_as_background(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            sample = root / "sparse.image"
            output = root / "sparse.tif"
            create_sample_image(sample, empty_tiles={(0, 0, 1)})

            result = convert_file(sample, output, ConvertOptions(tile_size=16))

            self.assertTrue(result.success, result.error)
            with TiffSlideSource(output) as source:
                blank = source.read_region(0, 256, 64, 64)
            self.assertEqual(int(blank.min()), 250)
            self.assertEqual(int(blank.max()), 250)

    def test_native_image_generic_tiff_preserves_levels_and_associated_pages(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            sample = root / "native.image"
            output = root / "native.tif"
            create_sample_image(sample, include_native_resources=True)

            result = convert_file(sample, output, ConvertOptions(tile_size=16))

            self.assertTrue(result.success, result.error)
            self.assertTrue(result.native_path)
            self.assertEqual(result.native_tile_mode, "lossless_transpose")
            self.assertEqual(result.native_level_dimensions, [(512, 512), (256, 256)])
            with tifffile.TiffFile(output) as tif:
                self.assertEqual(len(tif.pages), 5)
                self.assertEqual(tif.pages[0].shape, (512, 512, 3))
                self.assertEqual(tif.pages[1].shape, (256, 256, 3))
                self.assertEqual(tif.pages[2].shape, (5, 300, 3))
                self.assertEqual(tif.pages[3].shape, (625, 1152, 3))
                self.assertEqual(tif.pages[4].shape, (294, 300, 3))
                self.assertEqual(int(tif.pages[2].compression), 5)
                self.assertIn("native thumbnail", (tif.pages[2].description or "").lower())
                self.assertIn("native macro", (tif.pages[3].description or "").lower())
                self.assertIn("native label", (tif.pages[4].description or "").lower())
                self.assertTrue(np.all(tif.pages[2].asarray() == [10, 20, 30]))
                self.assertTrue(np.all(tif.pages[3].asarray() == [40, 50, 60]))
                self.assertTrue(np.all(tif.pages[4].asarray() == [70, 80, 90]))

    def test_native_image_svs_preserves_associated_images_and_uses_native_levels(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            sample = root / "native.image"
            output = root / "native.svs"
            create_sample_image(sample, include_native_resources=True)

            result = convert_file(
                sample,
                output,
                ConvertOptions(
                    output_format="svs",
                    tile_size=16,
                    svs_finalize_with_libtiff=False,
                    svs_validate_with_tiffinfo=False,
                ),
            )

            self.assertTrue(result.success, result.error)
            self.assertTrue(result.native_path)
            self.assertEqual(result.native_tile_mode, "reencoded")
            self.assertEqual(result.svs_label_dimensions, (300, 294))
            self.assertEqual(result.svs_macro_dimensions, (1152, 625))
            with tifffile.TiffFile(output) as tif:
                self.assertEqual(len(tif.pages), 5)
                self.assertEqual(tif.pages[1].shape, (5, 300, 3))
                self.assertEqual(tif.pages[2].shape, (128, 128, 3))
                self.assertEqual(tif.pages[3].shape, (294, 300, 3))
                self.assertEqual(tif.pages[4].shape, (625, 1152, 3))
                self.assertEqual(int(tif.pages[1].compression), 5)
                self.assertEqual(int(tif.pages[3].compression), 5)
                self.assertEqual(int(tif.pages[4].compression), 5)
                self.assertTrue(np.all(tif.pages[1].asarray() == [10, 20, 30]))
                self.assertTrue(np.all(tif.pages[3].asarray() == [70, 80, 90]))
                self.assertTrue(np.all(tif.pages[4].asarray() == [40, 50, 60]))

    def test_image_string_scan_time_is_rendered_on_svs_label(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            sample = Path(tempdir) / "sample.image"
            create_sample_image(sample)

            with PunuoxiImageSource(sample) as source:
                with mock.patch("ibl2svs.writer.ImageDraw.Draw") as draw_factory:
                    _build_label_array(source)

            rendered_text = [call.args[1] for call in draw_factory.return_value.text.call_args_list]
            self.assertIn("Scanned: 2026-01-01 12:34:56", rendered_text)

    def test_generic_tiff_pyramid_progress_keeps_accumulated_overall_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            sample = root / "sample.ibl"
            output = root / "sample.tif"
            create_sample_ibl(sample, img_width=512, img_height=512, tile_width=128, tile_height=128)

            progress_events: list[tuple[str, int, int, int, int]] = []

            result = convert_file(
                sample,
                output,
                ConvertOptions(tile_size=16),
                progress_callback=lambda current, level, done, total, overall_done, overall_total: progress_events.append(
                    (level, done, total, overall_done, overall_total)
                ),
            )

            self.assertTrue(result.success)
            pyramid_events = [event for event in progress_events if event[0] == "生成金字塔"]
            self.assertTrue(pyramid_events)
            first = pyramid_events[0]
            last = pyramid_events[-1]
            self.assertGreater(first[3], 0)
            self.assertGreater(first[4], first[3])
            self.assertLess(last[3], last[4])
            self.assertEqual(progress_events[-1][0], "写出文件")
            self.assertEqual(progress_events[-1][3], progress_events[-1][4])

    def test_convert_file_writes_experimental_svs(self) -> None:
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

            self.assertTrue(result.success)
            self.assertEqual(result.output_format, "svs")
            self.assertEqual(output.suffix, ".svs")
            self.assertEqual(result.backend, "svs-streaming-direct")
            self.assertFalse(bool(result.svs_is_bigtiff))
            self.assertIsNotNone(result.svs_label_dimensions)
            self.assertIsNotNone(result.svs_macro_dimensions)
            self.assertIn(result.svs_finalize_backend, {"libtiff-tiffset", "none", "unavailable"})
            self.assertIsNotNone(result.svs_photometric_pages)
            with tifffile.TiffFile(output) as tif:
                self.assertEqual(len(tif.pages), 7)
                self.assertIn("Aperio Image Library", tif.pages[0].description or "")
                self.assertIn("Aperio Image Library", tif.pages[1].description or "")
                self.assertTrue(tif.pages[0].is_tiled)
                self.assertFalse(tif.pages[1].is_tiled)
                self.assertTrue(tif.pages[2].is_tiled)
                self.assertTrue(tif.pages[3].is_tiled)
                self.assertTrue(tif.pages[4].is_tiled)
                self.assertFalse(tif.pages[5].is_tiled)
                self.assertFalse(tif.pages[6].is_tiled)
                self.assertIn("label", (tif.pages[5].description or "").lower())
                self.assertIn("macro", (tif.pages[6].description or "").lower())
                self.assertEqual(int(tif.pages[5].compression), 5)
                self.assertEqual(int(tif.pages[6].compression), 7)

    def test_convert_svs_to_generic_tiff_rebuilds_pyramid(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            sample = root / "sample.ibl"
            svs_output = root / "sample.svs"
            tiff_output = root / "rebuilt.tif"
            create_sample_ibl(sample, img_width=512, img_height=512, tile_width=128, tile_height=128)
            first = convert_file(
                sample,
                svs_output,
                ConvertOptions(tile_size=16, output_format="svs", svs_finalize_with_libtiff=False, svs_validate_with_tiffinfo=False),
            )

            result = convert_file(svs_output, tiff_output, ConvertOptions(tile_size=16))

            self.assertTrue(first.success)
            self.assertTrue(result.success)
            self.assertEqual(result.input_format, "svs")
            self.assertEqual(result.output_format, "generic_tiff")
            with tifffile.TiffFile(tiff_output) as tif:
                self.assertGreaterEqual(len(tif.pages), 2)
                self.assertEqual(tif.pages[0].shape, (512, 512, 3))
                self.assertTrue(tif.pages[0].is_tiled)
                self.assertTrue(tif.pages[1].is_tiled)
                self.assertEqual(int(tif.pages[0].compression), 7)
                self.assertIn("AppMag = 40", tif.pages[0].description or "")
                mpp_x, mpp_y = _page_mpp(tif.pages[0])
                self.assertAlmostEqual(mpp_x, 0.25, places=4)
                self.assertAlmostEqual(mpp_y, 0.25, places=4)
            with TiffSlideSource(tiff_output) as source:
                self.assertEqual(source.base_info.max_zoom_rate, 40)

    def test_convert_generic_tiff_to_svs_rebuilds_aperio_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            sample = root / "sample.ibl"
            tiff_output = root / "sample.tif"
            svs_output = root / "rebuilt.svs"
            create_sample_ibl(sample, img_width=512, img_height=512, tile_width=128, tile_height=128)
            first = convert_file(sample, tiff_output, ConvertOptions(tile_size=16))

            result = convert_file(
                tiff_output,
                svs_output,
                ConvertOptions(tile_size=16, output_format="svs", svs_finalize_with_libtiff=False, svs_validate_with_tiffinfo=False),
            )

            self.assertTrue(first.success)
            self.assertTrue(result.success)
            self.assertEqual(result.input_format, "generic_tiff")
            self.assertEqual(result.output_format, "svs")
            self.assertFalse(bool(result.svs_is_bigtiff))
            with tifffile.TiffFile(svs_output) as tif:
                self.assertGreaterEqual(len(tif.pages), 5)
                self.assertIn("Aperio Image Library", tif.pages[0].description or "")
                self.assertTrue(tif.pages[0].is_tiled)
                self.assertFalse(tif.pages[1].is_tiled)
                self.assertTrue(tif.pages[2].is_tiled)
                self.assertIn("label", (tif.pages[-2].description or "").lower())
                self.assertIn("macro", (tif.pages[-1].description or "").lower())

    def test_convert_file_svs_progress_uses_standard_phases_and_finishes_in_write_stage(self) -> None:
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

            progress_events: list[tuple[str, int, int, int, int]] = []
            result = convert_file(
                sample,
                output,
                ConvertOptions(tile_size=16, output_format="svs"),
                progress_callback=lambda current, level, done, total, overall_done, overall_total: progress_events.append(
                    (level, done, total, overall_done, overall_total)
                ),
            )

            self.assertTrue(result.success)
            self.assertTrue(progress_events)
            self.assertFalse(any(" L" in event[0] for event in progress_events))
            self.assertIn("生成附属图像", {event[0] for event in progress_events})
            last = progress_events[-1]
            self.assertEqual(last[0], "写出文件")
            self.assertEqual(last[1], last[2])
            self.assertEqual(last[3], last[4])

    def test_streaming_svs_streaming_direct_does_not_call_tiffpage_asarray(self) -> None:
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

            with mock.patch("tifffile.TiffPage.asarray", side_effect=AssertionError("page.asarray should not be used")):
                result = convert_file(
                    sample,
                    output,
                    ConvertOptions(tile_size=16, output_format="svs", svs_validate_with_tiffinfo=False),
                )

            self.assertTrue(result.success)
            self.assertEqual(result.backend, "svs-streaming-direct")

    def test_streaming_svs_progress_matches_gui_phase_contract(self) -> None:
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

            progress_events: list[tuple[str, int, int, int, int]] = []
            result = convert_file(
                sample,
                output,
                ConvertOptions(tile_size=16, output_format="svs", svs_validate_with_tiffinfo=False),
                progress_callback=lambda current, level, done, total, overall_done, overall_total: progress_events.append(
                    (level, done, total, overall_done, overall_total)
                ),
            )

            self.assertTrue(result.success)
            self.assertEqual(result.backend, "svs-streaming-direct")
            self.assertFalse(any(" L" in event[0] for event in progress_events))
            phases = {event[0] for event in progress_events}
            self.assertIn("解析 IBL", phases)
            self.assertEqual(progress_events[-1][0], "写出文件")
            self.assertEqual(progress_events[-1][3], progress_events[-1][4])

    def test_svs_streaming_direct_writes_correct_layout(self) -> None:
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

            self.assertTrue(result.success)
            self.assertEqual(result.backend, "svs-streaming-direct")
            self.assertEqual(result.pyramid_levels, 4)
            with tifffile.TiffFile(output) as tif:
                self.assertEqual(len(tif.pages), 7)
                # Main page
                self.assertTrue(tif.pages[0].is_tiled)
                self.assertEqual(int(tif.pages[0].subfiletype), 0)
                self.assertEqual(int(tif.pages[0].compression), 7)  # JPEG
                # Thumbnail
                self.assertFalse(tif.pages[1].is_tiled)
                self.assertEqual(int(tif.pages[1].compression), 7)  # JPEG
                # 4x pyramid
                self.assertTrue(tif.pages[2].is_tiled)
                self.assertEqual(int(tif.pages[2].subfiletype), 0)
                # 8x pyramid
                self.assertTrue(tif.pages[3].is_tiled)
                self.assertEqual(int(tif.pages[3].subfiletype), 0)
                # 16x pyramid
                self.assertTrue(tif.pages[4].is_tiled)
                self.assertEqual(int(tif.pages[4].subfiletype), 0)
                # Label
                self.assertFalse(tif.pages[5].is_tiled)
                self.assertEqual(int(tif.pages[5].subfiletype), 1)
                self.assertEqual(int(tif.pages[5].compression), 5)  # LZW
                # Macro
                self.assertFalse(tif.pages[6].is_tiled)
                self.assertEqual(int(tif.pages[6].subfiletype), 9)
                self.assertEqual(int(tif.pages[6].compression), 7)  # JPEG

    def test_svs_streaming_direct_progress_includes_parse_ibl(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            sample = root / "sample.ibl"
            output = root / "sample.svs"
            create_sample_ibl(sample, img_width=512, img_height=512, tile_width=128, tile_height=128)

            progress_events: list[tuple[str, int, int, int, int]] = []
            result = convert_file(
                sample,
                output,
                ConvertOptions(tile_size=16, output_format="svs", svs_validate_with_tiffinfo=False),
                progress_callback=lambda current, level, done, total, overall_done, overall_total: progress_events.append(
                    (level, done, total, overall_done, overall_total)
                ),
            )

            self.assertTrue(result.success)
            phases = [event[0] for event in progress_events]
            self.assertEqual(phases[0], "解析 IBL")
            self.assertIn("构建主图", phases)
            self.assertIn("生成金字塔", phases)

    def test_convert_file_rejects_invalid_tile_size(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            sample = root / "sample.ibl"
            output = root / "sample.tif"
            create_sample_ibl(sample)

            result = convert_file(sample, output, ConvertOptions(tile_size=15))

            self.assertFalse(result.success)
            self.assertEqual(result.status, "failed")
            self.assertEqual(result.error_code, "CONVERT_FAILED")
            self.assertIn("tile_size", result.error or "")
            self.assertFalse(output.exists())

    def test_parallel_mode_reports_encode_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            sample = root / "sample.ibl"
            output = root / "sample.tif"
            create_sample_ibl(sample, img_width=1024, img_height=1024, tile_width=256, tile_height=256)

            result = convert_file(
                sample,
                output,
                ConvertOptions(tile_size=16, encoder_workers=2, raw_queue_size=4, encoded_queue_size=4),
            )

            self.assertTrue(result.success)
            self.assertGreaterEqual(result.encode_sec, 0.0)
            self.assertGreaterEqual(result.writer_wait_sec, 0.0)
            self.assertGreater(result.main_write_sec + result.pyramid_sec, 0.0)

    @unittest.skipUnless(OPENSLIDE_AVAILABLE, "openslide is required for OpenSlide compatibility test")
    def test_output_is_readable_as_generic_tiff_by_openslide(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            sample = root / "sample.ibl"
            output = root / "sample.tif"
            create_sample_ibl(sample, img_width=512, img_height=512, tile_width=128, tile_height=128)

            result = convert_file(sample, output, ConvertOptions(tile_size=16))

            self.assertTrue(result.success)
            slide = openslide.OpenSlide(str(output))
            self.assertEqual(slide.properties.get(openslide.PROPERTY_NAME_VENDOR), "generic-tiff")
            self.assertEqual(slide.level_count, 2)
            self.assertEqual(slide.level_dimensions[0], (512, 512))
            self.assertEqual(slide.level_dimensions[1], (256, 256))
            slide.close()


if __name__ == "__main__":
    unittest.main()
