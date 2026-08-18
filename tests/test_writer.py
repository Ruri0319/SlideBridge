from __future__ import annotations

import importlib.util
from io import BytesIO
from unittest import mock
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from ibl2svs.converter import convert_file
from ibl2svs.kfb_source import KfbSlideSource
from ibl2svs.models import ConvertOptions
from ibl2svs.punuoxi_source import PunuoxiImageSource
from ibl2svs.tiff_source import TiffSlideSource
from ibl2svs.writer import (
    _build_macro_array,
    _build_label_array,
    _build_thumbnail_array,
    _compute_svs_pyramid_shapes,
    finalize_svs_with_libtiff,
    _svs_use_bigtiff,
    write_aperio_associated_images,
    write_image,
    WriteImageError,
)
from tests.support import create_sample_ibl, create_sample_image, create_sample_kfb, create_sample_kfba


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
                ConvertOptions(
                    tile_size=16,
                    output_format="svs",
                    svs_finalize_with_libtiff=False,
                    svs_validate_with_tiffinfo=False,
                    svs_synthesize_associated_images=True,
                ),
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

        self.assertEqual(shapes, [(39539, 15403), (9884, 3850), (4942, 1925)])

    def test_svs_helpers_preserve_source_level_shapes(self) -> None:
        class SlideStub:
            width = 26858
            height = 39428
            source_container = "svs"
            svs_level_dimensions = [(26858, 39428), (6714, 9857), (1678, 2464)]

        self.assertEqual(
            _compute_svs_pyramid_shapes(SlideStub(), ConvertOptions(output_format="svs")),
            [(6714, 9857), (1678, 2464)],
        )

    def test_svs_auto_detects_all_j2k_compression_names(self) -> None:
        options = ConvertOptions(output_format="svs")
        for source_codec in (
            "APERIO_JP2000_YCBC",
            "APERIO_JP2000_RGB",
            "JPEG_2000_LOSSY",
            "JPEG2000",
        ):
            self.assertEqual(options.resolved_svs_codec(source_codec), "aperio_j2k")

    def test_svs_bigtiff_estimate_uses_codec_quality_and_source_levels(self) -> None:
        class SlideStub:
            width = 50_000
            height = 50_000
            source_container = "svs"
            svs_level_dimensions = [
                (50_000, 50_000),
                (12_500, 12_500),
                (3_125, 3_125),
                (1_562, 1_562),
            ]

        jpeg_slide = SlideStub()
        jpeg_slide.source_codec = "JPEG"
        self.assertFalse(
            _svs_use_bigtiff(
                jpeg_slide,
                ConvertOptions(output_format="svs", main_quality=90, pyramid_quality=60),
            )
        )

        j2k_slide = SlideStub()
        j2k_slide.source_codec = "APERIO_JP2000_RGB"
        self.assertTrue(
            _svs_use_bigtiff(
                j2k_slide,
                ConvertOptions(output_format="svs", main_quality=100, pyramid_quality=100),
            )
        )

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

    def test_convert_file_writes_pyramidal_ome_tiff_and_reportable_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            sample = root / "sample.ibl"
            output = root / "sample.ome.tif"
            create_sample_ibl(sample, img_width=512, img_height=512, tile_width=128, tile_height=128)

            result = convert_file(sample, output, ConvertOptions(tile_size=16))

            self.assertTrue(result.success)
            self.assertTrue(output.exists())
            self.assertGreaterEqual(result.pyramid_levels or 0, 2)
            self.assertEqual(result.output_format, "ome_tiff")
            self.assertEqual(result.backend, "tifffile-ome")
            self.assertEqual(result.level_dimensions, [(512, 512), (256, 256)])
            self.assertGreaterEqual(result.read_decode_sec, 0.0)
            self.assertGreaterEqual(result.main_write_sec, 0.0)
            self.assertGreaterEqual(result.pyramid_sec, 0.0)
            self.assertGreaterEqual(result.peak_memory_mb, 0.0)
            self.assertGreaterEqual(result.avg_cpu_percent, 0.0)
            with tifffile.TiffFile(output) as tif:
                self.assertTrue(tif.is_ome)
                self.assertEqual(tif.pages[0].shape, (512, 512, 3))
                self.assertTrue(tif.pages[0].is_tiled)
                self.assertIn("AppMag = 40", tif.pages[0].description or "")
                self.assertIn("ObjectivePower = 40", tif.pages[0].description or "")
                self.assertEqual(tif.pages[0].tags["ResolutionUnit"].value, 3)
                mpp_x, mpp_y = _page_mpp(tif.pages[0])
                self.assertAlmostEqual(mpp_x, 0.25, places=4)
                self.assertAlmostEqual(mpp_y, 0.25, places=4)
                level_page = tif.series[0].levels[1].pages[0]
                self.assertEqual(level_page.shape, (256, 256, 3))
                self.assertTrue(level_page.is_tiled)
                self.assertEqual(level_page.subfiletype, 1)
                level_mpp_x, level_mpp_y = _page_mpp(level_page)
                self.assertAlmostEqual(level_mpp_x, 0.5, places=4)
                self.assertAlmostEqual(level_mpp_y, 0.5, places=4)
            with TiffSlideSource(output) as source:
                self.assertEqual(source.base_info.max_zoom_rate, 40)

    def test_image_conversion_preserves_restored_tile_orientation(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            sample = root / "directional.image"
            output = root / "directional.ome.tif"
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
            output = root / "sparse.ome.tif"
            create_sample_image(sample, empty_tiles={(0, 0, 1)})

            result = convert_file(sample, output, ConvertOptions(tile_size=16))

            self.assertTrue(result.success, result.error)
            with TiffSlideSource(output) as source:
                blank = source.read_region(0, 256, 64, 64)
            self.assertEqual(int(blank.min()), 250)
            self.assertEqual(int(blank.max()), 250)

    def test_native_image_ome_tiff_preserves_levels_and_associated_images(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            sample = root / "native.image"
            output = root / "native.ome.tif"
            create_sample_image(sample, include_native_resources=True)

            progress_events: list[tuple[str, int, int, int, int]] = []
            result = convert_file(
                sample,
                output,
                ConvertOptions(tile_size=16),
                progress_callback=lambda current, level, done, total, overall_done, overall_total: progress_events.append(
                    (level, done, total, overall_done, overall_total)
                ),
            )

            self.assertTrue(result.success, result.error)
            self.assertTrue(result.native_path)
            self.assertEqual(result.native_tile_mode, "lossless_transpose")
            self.assertEqual(result.native_level_dimensions, [(512, 512), (256, 256)])
            native_progress = [event for event in progress_events if event[0] == "写出原生层"]
            self.assertTrue(native_progress)
            self.assertEqual(native_progress[0][1:3], (1, 4))
            self.assertGreater(native_progress[-1][3], native_progress[0][3])
            with tifffile.TiffFile(output) as tif:
                self.assertEqual(len(tif.pages), 4)
                self.assertEqual(len(tif.series[0].levels), 2)
                self.assertEqual(tif.pages[0].shape, (512, 512, 3))
                self.assertEqual(tif.series[0].levels[1].shape, (256, 256, 3))
                self.assertEqual(tif.pages[1].shape, (5, 300, 3))
                self.assertEqual(tif.pages[2].shape, (625, 1152, 3))
                self.assertEqual(tif.pages[3].shape, (294, 300, 3))
                self.assertEqual(int(tif.pages[1].compression), 5)
                self.assertEqual([series.name for series in tif.series[1:]], ["thumbnail", "macro", "label"])
                self.assertTrue(np.all(tif.pages[1].asarray() == [10, 20, 30]))
                self.assertTrue(np.all(tif.pages[2].asarray() == [40, 50, 60]))
                self.assertTrue(np.all(tif.pages[3].asarray() == [70, 80, 90]))

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
            self.assertEqual(result.native_tile_mode, "svs_reencoded")
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

    def test_native_kfb_ome_tiff_preserves_levels_jpeg_tables_and_resources(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            sample = root / "native.kfbf"
            output = root / "native.ome.tif"
            create_sample_kfb(sample, variant="kfbf")

            with KfbSlideSource(sample) as source:
                expected = source.read_level_region(0, 0, 0, source.width, source.height)
                expected_thumbnail = np.asarray(source.get_thumbnail_image())
                expected_macro = np.asarray(source.get_macro_image())
                expected_label = np.asarray(source.get_label_image())
                first = source.levels[0].records[0]
                source_jpeg = source._read_at(first.offset, first.length)
                source_quantization = Image.open(BytesIO(source_jpeg)).quantization

            result = convert_file(sample, output, ConvertOptions(tile_size=16))

            self.assertTrue(result.success, result.error)
            self.assertTrue(result.native_path)
            self.assertEqual(result.native_tile_mode, "jpeg_passthrough")
            self.assertEqual(result.compatibility_level, "sample_verified")
            self.assertEqual(result.native_level_dimensions, [(24, 18), (12, 9)])
            with tifffile.TiffFile(output) as tif:
                self.assertEqual(len(tif.pages), 4)
                self.assertEqual(len(tif.series[0].levels), 2)
                self.assertEqual((tif.pages[0].tilewidth, tif.pages[0].tilelength), (16, 16))
                np.testing.assert_allclose(tif.pages[0].asarray(), expected, atol=5)
                for level in tif.series[0].levels:
                    page = level.pages[0]
                    for offset, size in zip(page.dataoffsets, page.databytecounts):
                        tif.filehandle.seek(offset)
                        with Image.open(BytesIO(tif.filehandle.read(size))) as tile:
                            self.assertEqual(tile.size, (16, 16))
                offset = tif.pages[0].dataoffsets[0]
                size = tif.pages[0].databytecounts[0]
                tif.filehandle.seek(offset)
                output_jpeg = tif.filehandle.read(size)
                self.assertEqual(Image.open(BytesIO(output_jpeg)).quantization, source_quantization)
                self.assertEqual(tif.pages[1].shape, (9, 12, 3))
                self.assertEqual(tif.pages[2].shape, (24, 64, 3))
                self.assertEqual(tif.pages[3].shape, (40, 32, 3))
                np.testing.assert_array_equal(tif.pages[1].asarray(), expected_thumbnail)
                np.testing.assert_array_equal(tif.pages[2].asarray(), expected_macro)
                np.testing.assert_array_equal(tif.pages[3].asarray(), expected_label)

    def test_native_kfb_svs_uses_240_tiles_and_preserves_associated_pixels(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            sample = root / "native.kfbf"
            output = root / "native.svs"
            create_sample_kfb(sample, variant="kfbf")
            with KfbSlideSource(sample) as source:
                expected_main = source.read_level_region(0, 0, 0, source.width, source.height)
                expected_thumbnail = np.asarray(source.get_thumbnail_image())
                expected_macro = np.asarray(source.get_macro_image())
                expected_label = np.asarray(source.get_label_image())

            result = convert_file(
                sample,
                output,
                ConvertOptions(
                    output_format="svs",
                    svs_finalize_with_libtiff=False,
                    svs_validate_with_tiffinfo=False,
                ),
            )

            self.assertTrue(result.success, result.error)
            self.assertEqual(result.native_tile_mode, "svs_reencoded")
            with tifffile.TiffFile(output) as tif:
                self.assertEqual((tif.pages[0].tilewidth, tif.pages[0].tilelength), (240, 240))
                np.testing.assert_allclose(tif.pages[0].asarray(), expected_main, atol=8)
                np.testing.assert_array_equal(tif.pages[1].asarray(), expected_thumbnail)
                np.testing.assert_array_equal(tif.pages[-2].asarray(), expected_label)
                np.testing.assert_array_equal(tif.pages[-1].asarray(), expected_macro)

    def test_kfba_ome_tiff_writes_raw_field_pyramids_only(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            sample = root / "multi.kfba"
            output = root / "multi.ome.tif"
            create_sample_kfba(sample)

            result = convert_file(sample, output, ConvertOptions(tile_size=16))

            self.assertTrue(result.success, result.error)
            self.assertEqual(result.native_tile_mode, "jpeg_passthrough")
            self.assertEqual(result.source_axes, "TZCYX")
            with tifffile.TiffFile(output) as tif:
                self.assertTrue(tif.is_ome)
                self.assertEqual(tif.series[0].shape, (2, 18, 24))
                self.assertEqual(len(tif.series[0].levels), 2)
                field_series = [series for series in tif.series if "field" in series.name]
                self.assertEqual(len(field_series), 2)
                field0 = field_series[0].asarray()
                field1 = field_series[1].asarray()
                self.assertEqual(field0.shape, (2, 18, 24))
                self.assertEqual(field1.shape, (2, 18, 24))
                np.testing.assert_allclose(field0[0], 30, atol=2)
                np.testing.assert_allclose(field0[1], 120, atol=2)
                np.testing.assert_allclose(field1[0], 70, atol=2)
                np.testing.assert_allclose(field1[1], 160, atol=2)
                self.assertIn('Name="Red"', tif.ome_metadata)
                self.assertIn('Name="Green"', tif.ome_metadata)
                self.assertIn('PhysicalSizeX="0.25"', tif.ome_metadata)

    def test_kfba_brightfield_svs_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            sample = root / "multi.kfba"
            output = root / "multi.svs"
            create_sample_kfba(sample)

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

            self.assertFalse(result.success)
            self.assertEqual(result.diagnostic_code, "incompatible_output")
            self.assertIn("荧光输入不能写为明场", result.error)
            self.assertFalse(output.exists())

    def test_image_string_scan_time_is_rendered_on_svs_label(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            sample = Path(tempdir) / "sample.image"
            create_sample_image(sample)

            with PunuoxiImageSource(sample) as source:
                with mock.patch("ibl2svs.writer.ImageDraw.Draw") as draw_factory:
                    _build_label_array(source)

            rendered_text = [call.args[1] for call in draw_factory.return_value.text.call_args_list]
            self.assertIn("Scanned: 2026-01-01 12:34:56", rendered_text)

    def test_native_associated_images_keep_source_dimensions(self) -> None:
        class NativeSlide:
            def __init__(self) -> None:
                self.images = {
                    "thumbnail": Image.new("RGB", (3, 1), (10, 20, 30)),
                    "macro": Image.new("RGB", (6, 2), (40, 50, 60)),
                    "label": Image.new("RGB", (2, 5), (70, 80, 90)),
                }

            def get_native_associated_image(self, name: str) -> Image.Image:
                return self.images[name]

        slide = NativeSlide()
        self.assertEqual(_build_thumbnail_array(slide).shape, (1, 3, 3))
        self.assertEqual(_build_macro_array(slide).shape, (2, 6, 3))
        self.assertEqual(_build_label_array(slide).shape, (5, 2, 3))

    def test_svs_preserves_generic_associated_images_without_synthesis(self) -> None:
        class GenericSlide:
            path = Path("generic.svs")

            def get_label_image(self) -> Image.Image:
                return Image.new("RGB", (7, 4), (11, 22, 33))

            def get_macro_image(self) -> Image.Image:
                return Image.new("RGB", (5, 3), (44, 55, 66))

        with tempfile.TemporaryDirectory() as tempdir:
            output = Path(tempdir) / "associated.svs"
            perf = {"thumbnail_sec": 0.0}
            with tifffile.TiffWriter(str(output)) as tif:
                metadata = write_aperio_associated_images(
                    tif,
                    GenericSlide(),
                    ConvertOptions(output_format="svs", svs_synthesize_associated_images=False),
                    perf,
                    overall_done=0,
                    overall_total=2,
                )

            self.assertEqual(metadata["svs_label_dimensions"], (7, 4))
            self.assertEqual(metadata["svs_macro_dimensions"], (5, 3))
            with tifffile.TiffFile(output) as tif:
                self.assertEqual(len(tif.pages), 2)
                self.assertIn("label", (tif.pages[0].description or "").lower())
                self.assertIn("macro", (tif.pages[1].description or "").lower())

    def test_ome_tiff_pyramid_progress_keeps_accumulated_overall_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            sample = root / "sample.ibl"
            output = root / "sample.ome.tif"
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

            result = convert_file(
                sample,
                output,
                ConvertOptions(
                    tile_size=16,
                    output_format="svs",
                    main_quality=78,
                    svs_synthesize_associated_images=True,
                ),
            )

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
                self.assertEqual(len(tif.pages), 6)
                self.assertIn("Aperio Image Library", tif.pages[0].description or "")
                self.assertIn("JPEG/RGB Q=78", tif.pages[0].description or "")
                self.assertIn("Aperio Image Library", tif.pages[1].description or "")
                self.assertTrue(tif.pages[0].is_tiled)
                self.assertFalse(tif.pages[1].is_tiled)
                self.assertTrue(tif.pages[2].is_tiled)
                self.assertIn("JPEG/RGB Q=60", tif.pages[2].description or "")
                self.assertTrue(tif.pages[3].is_tiled)
                self.assertIn("JPEG/RGB Q=60", tif.pages[3].description or "")
                self.assertFalse(tif.pages[4].is_tiled)
                self.assertFalse(tif.pages[5].is_tiled)
                self.assertIn("label", (tif.pages[4].description or "").lower())
                self.assertIn("macro", (tif.pages[5].description or "").lower())
                self.assertEqual(int(tif.pages[4].compression), 5)
                self.assertEqual(int(tif.pages[5].compression), 7)

    def test_convert_svs_to_ome_tiff_rebuilds_pyramid(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            sample = root / "sample.ibl"
            svs_output = root / "sample.svs"
            tiff_output = root / "rebuilt.ome.tif"
            create_sample_ibl(sample, img_width=512, img_height=512, tile_width=128, tile_height=128)
            first = convert_file(
                sample,
                svs_output,
                ConvertOptions(
                    tile_size=16,
                    output_format="svs",
                    svs_finalize_with_libtiff=False,
                    svs_validate_with_tiffinfo=False,
                    svs_synthesize_associated_images=True,
                ),
            )

            result = convert_file(svs_output, tiff_output, ConvertOptions(tile_size=16))

            self.assertTrue(first.success)
            self.assertTrue(result.success)
            self.assertEqual(result.input_format, "svs")
            self.assertEqual(result.output_format, "ome_tiff")
            with tifffile.TiffFile(tiff_output) as tif:
                self.assertTrue(tif.is_ome)
                self.assertGreaterEqual(len(tif.series[0].levels), 2)
                self.assertEqual(tif.pages[0].shape, (512, 512, 3))
                self.assertTrue(tif.pages[0].is_tiled)
                self.assertTrue(tif.series[0].levels[1].pages[0].is_tiled)
                self.assertEqual(int(tif.pages[0].compression), 7)
                self.assertIn("AppMag = 40", tif.pages[0].description or "")
                mpp_x, mpp_y = _page_mpp(tif.pages[0])
                self.assertAlmostEqual(mpp_x, 0.25, places=4)
                self.assertAlmostEqual(mpp_y, 0.25, places=4)
            with TiffSlideSource(tiff_output) as source:
                self.assertEqual(source.base_info.max_zoom_rate, 40)

    def test_convert_ome_tiff_to_svs_rebuilds_aperio_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            sample = root / "sample.ibl"
            tiff_output = root / "sample.ome.tif"
            svs_output = root / "rebuilt.svs"
            create_sample_ibl(sample, img_width=512, img_height=512, tile_width=128, tile_height=128)
            first = convert_file(sample, tiff_output, ConvertOptions(tile_size=16))

            result = convert_file(
                tiff_output,
                svs_output,
                ConvertOptions(
                    tile_size=16,
                    output_format="svs",
                    svs_finalize_with_libtiff=False,
                    svs_validate_with_tiffinfo=False,
                    svs_synthesize_associated_images=True,
                ),
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
                ConvertOptions(
                    tile_size=16,
                    output_format="svs",
                    svs_synthesize_associated_images=True,
                ),
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

            result = convert_file(
                sample,
                output,
                ConvertOptions(
                    tile_size=16,
                    output_format="svs",
                    svs_synthesize_associated_images=True,
                ),
            )

            self.assertTrue(result.success)
            self.assertEqual(result.backend, "svs-streaming-direct")
            self.assertEqual(result.pyramid_levels, 3)
            with tifffile.TiffFile(output) as tif:
                self.assertEqual(len(tif.pages), 6)
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
                # 16x pyramid
                self.assertTrue(tif.pages[3].is_tiled)
                self.assertEqual(int(tif.pages[3].subfiletype), 0)
                # Label
                self.assertFalse(tif.pages[4].is_tiled)
                self.assertEqual(int(tif.pages[4].subfiletype), 1)
                self.assertEqual(int(tif.pages[4].compression), 5)  # LZW
                # Macro
                self.assertFalse(tif.pages[5].is_tiled)
                self.assertEqual(int(tif.pages[5].subfiletype), 9)
                self.assertEqual(int(tif.pages[5].compression), 7)  # JPEG

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
            output = root / "sample.ome.tif"
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
            output = root / "sample.ome.tif"
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
    def test_output_is_readable_as_pyramidal_ome_tiff_by_openslide(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            sample = root / "sample.ibl"
            output = root / "sample.ome.tif"
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
