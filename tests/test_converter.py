from __future__ import annotations

import tempfile
import threading
import time
import unittest
import struct
from pathlib import Path
from unittest import mock

from ibl2svs.converter import (
    build_output_path,
    convert_file,
    convert_folder,
    detect_input_format,
    find_convertible_files,
    find_ibl_files,
    write_report,
)
from ibl2svs.models import ConvertOptions, ConvertResult
from ibl2svs.writer import WriteImageError
from tests.support import create_sample_ibl, create_sample_image, create_sample_kfb


class ConverterTests(unittest.TestCase):
    def test_find_ibl_files_honors_recursion_and_case(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            (root / "a.ibl").write_text("x")
            (root / "b.txt").write_text("x")
            nested = root / "nested"
            nested.mkdir()
            (nested / "c.IBL").write_text("x")

            flat = find_ibl_files(root, recursive=False)
            deep = find_ibl_files(root, recursive=True)

            self.assertEqual([path.name for path in flat], ["a.ibl"])
            self.assertEqual([path.name for path in deep], ["a.ibl", "c.IBL"])

    def test_detect_input_format_recognizes_wsi_extensions(self) -> None:
        self.assertEqual(detect_input_format("a.ibl"), "ibl")
        self.assertEqual(detect_input_format("a.svs"), "svs")
        self.assertEqual(detect_input_format("a.tif"), "generic_tiff")
        self.assertEqual(detect_input_format("a.tiff"), "generic_tiff")
        self.assertEqual(detect_input_format("a.kfb"), "kfb")
        self.assertEqual(detect_input_format("a.kfbf"), "kfb")
        self.assertEqual(detect_input_format("a.kfbl"), "kfb")
        self.assertEqual(detect_input_format("a.kfba"), "kfb")
        self.assertEqual(detect_input_format("a.kfbx"), "kfb")
        self.assertEqual(detect_input_format("a.image"), "image")
        self.assertEqual(detect_input_format("a.txt"), "unsupported")

    def test_find_convertible_files_depends_on_output_format(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            for name in ["a.ibl", "b.svs", "c.tif", "d.tiff", "e.kfb", "f.kfbf", "g.image", "h.kfbl", "i.kfba", "j.kfbx", "f.txt"]:
                (root / name).write_text("x")

            to_svs = find_convertible_files(root, recursive=False, output_format="svs")
            to_tiff = find_convertible_files(root, recursive=False, output_format="generic_tiff")

            self.assertEqual([path.name for path in to_svs], ["a.ibl", "c.tif", "d.tiff", "e.kfb", "f.kfbf", "g.image", "h.kfbl", "i.kfba", "j.kfbx"])
            self.assertEqual([path.name for path in to_tiff], ["a.ibl", "b.svs", "e.kfb", "f.kfbf", "g.image", "h.kfbl", "i.kfba", "j.kfbx"])

    def test_build_output_path_avoids_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            out = root / "out"
            out.mkdir()
            source = root / "in" / "sample.ibl"
            source.parent.mkdir()
            source.write_text("x")

            first = build_output_path(source, root / "in", out, "generic_tiff")
            first.touch()
            second = build_output_path(source, root / "in", out, "generic_tiff")

            self.assertEqual(first.name, "sample.tif")
            self.assertEqual(second.name, "sample_1.tif")

    def test_build_output_path_supports_svs_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            out = root / "out"
            out.mkdir()
            source = root / "in" / "sample.ibl"
            source.parent.mkdir()
            source.write_text("x")

            first = build_output_path(source, root / "in", out, "svs")

            self.assertEqual(first.name, "sample.svs")

    def test_write_report_includes_input_format_column(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            report = Path(tempdir) / "report.csv"
            write_report(
                [
                    ConvertResult(
                        input_path=Path("sample.svs"),
                        output_path=Path("sample.tif"),
                        success=True,
                        input_format="svs",
                    )
                ],
                report,
            )

            text = report.read_text(encoding="utf-8-sig")

            self.assertIn("input_format", text.splitlines()[0])
            self.assertIn("svs_omitted_native_data", text.splitlines()[0])
            self.assertIn("svs", text)

    def test_convert_file_marks_cancelled_and_removes_partial_output(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            source = root / "sample.ibl"
            output = root / "sample.tif"
            create_sample_ibl(source)
            cancel_event = threading.Event()
            cancel_event.set()

            result = convert_file(source, output, ConvertOptions(tile_size=16), cancel_event=cancel_event)

            self.assertFalse(result.success)
            self.assertEqual(result.status, "cancelled")
            self.assertEqual(result.error_code, "CANCELLED")
            self.assertIsNone(result.output_path)
            self.assertFalse(output.exists())

    def test_convert_file_preserves_svs_failure_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            source = root / "sample.ibl"
            output = root / "sample.svs"
            create_sample_ibl(source)

            perf = {
                "backend": "svs-streaming-direct",
                "failure_stage": "构建主图",
                "openslide_vendor": None,
                "svs_finalize_backend": "unavailable",
            }

            with mock.patch("ibl2svs.converter.write_image", side_effect=WriteImageError("mock failure", perf)):
                result = convert_file(source, output, ConvertOptions(tile_size=16, output_format="svs"))

            self.assertFalse(result.success)
            self.assertEqual(result.failure_stage, "构建主图")
            self.assertEqual(result.backend, "svs-streaming-direct")
            self.assertEqual(result.svs_finalize_backend, "unavailable")

    def test_convert_file_accepts_kfb_source(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            source = root / "sample.kfb"
            output = root / "sample.tif"
            create_sample_kfb(source)

            def fake_write_image(slide, output_path, options, progress_callback=None, cancel_event=None):
                Path(output_path).write_bytes(b"mock tiff")
                return 1, {
                    "backend": "mock",
                    "level_dimensions": [(slide.width, slide.height)],
                }

            with mock.patch("ibl2svs.converter.write_image", side_effect=fake_write_image):
                result = convert_file(source, output, ConvertOptions(tile_size=16))

            self.assertTrue(result.success)
            self.assertEqual(result.input_format, "kfb")
            self.assertEqual(result.width, 24)
            self.assertEqual(result.height, 18)
            self.assertEqual(result.mpp, 0.25)
            self.assertTrue(output.exists())

    def test_unknown_kfb_layout_returns_diagnostic_without_partial_output(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            source = root / "unknown.kfb"
            output = root / "unknown.tif"
            create_sample_kfb(source)
            with source.open("r+b") as fh:
                fh.seek(0x0C)
                fh.write(struct.pack("<f", 3.0))

            result = convert_file(source, output, ConvertOptions(tile_size=16))

            self.assertFalse(result.success)
            self.assertEqual(result.diagnostic_code, "unsupported_version")
            self.assertEqual(result.diagnostic_stage, "header")
            self.assertFalse(output.exists())
            self.assertFalse(output.with_suffix(".tif.part").exists())

    def test_convert_file_accepts_image_source(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            source = root / "sample.image"
            output = root / "sample.tif"
            create_sample_image(source)

            def fake_write_image(slide, output_path, options, progress_callback=None, cancel_event=None):
                self.assertEqual(slide.width, 512)
                self.assertEqual(slide.height, 512)
                Path(output_path).write_bytes(b"mock tiff")
                return 1, {
                    "backend": "mock",
                    "level_dimensions": [(slide.width, slide.height)],
                }

            with mock.patch("ibl2svs.converter.write_image", side_effect=fake_write_image):
                result = convert_file(source, output, ConvertOptions(tile_size=16))

            self.assertTrue(result.success)
            self.assertEqual(result.input_format, "image")
            self.assertEqual(result.width, 512)
            self.assertEqual(result.height, 512)
            self.assertAlmostEqual(result.mpp, 0.3466053, places=5)
            self.assertTrue(output.exists())

    def test_convert_folder_parallel_mode_preserves_unique_output_names(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            (input_dir / "sample.ibl").write_text("x")
            (input_dir / "sample.svs").write_text("x")

            def fake_convert(input_path, output_path, options, logger=None, progress_callback=None, cancel_event=None):
                return ConvertResult(
                    input_path=Path(input_path),
                    output_path=Path(output_path),
                    success=True,
                    output_format=options.output_format,
                )

            with mock.patch("ibl2svs.converter.convert_file", side_effect=fake_convert):
                batch = convert_folder(
                    input_dir,
                    output_dir,
                    ConvertOptions(output_format="generic_tiff", parallel_wsi=2),
                )

            self.assertEqual(batch.success_count, 2)
            self.assertEqual([result.output_path.name for result in batch.results if result.output_path], ["sample.tif", "sample_1.tif"])

    def test_convert_folder_parallel_mode_runs_multiple_files_concurrently(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            for name in ["a.ibl", "b.ibl", "c.ibl"]:
                (input_dir / name).write_text("x")

            lock = threading.Lock()
            active = 0
            max_active = 0

            def fake_convert(input_path, output_path, options, logger=None, progress_callback=None, cancel_event=None):
                nonlocal active, max_active
                with lock:
                    active += 1
                    max_active = max(max_active, active)
                time.sleep(0.05)
                with lock:
                    active -= 1
                return ConvertResult(
                    input_path=Path(input_path),
                    output_path=Path(output_path),
                    success=True,
                    output_format=options.output_format,
                )

            with mock.patch("ibl2svs.converter.convert_file", side_effect=fake_convert):
                batch = convert_folder(
                    input_dir,
                    output_dir,
                    ConvertOptions(output_format="generic_tiff", parallel_wsi=2),
                )

            self.assertEqual(batch.success_count, 3)
            self.assertEqual(max_active, 2)

    def test_convert_folder_parallel_mode_splits_total_memory_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            for name in ["a.ibl", "b.ibl", "c.ibl"]:
                (input_dir / name).write_text("x")

            task_budgets: list[int] = []

            def fake_convert(input_path, output_path, options, logger=None, progress_callback=None, cancel_event=None):
                task_budgets.append(options.memory_budget_mb)
                return ConvertResult(
                    input_path=Path(input_path),
                    output_path=Path(output_path),
                    success=True,
                    output_format=options.output_format,
                )

            with mock.patch("ibl2svs.converter.convert_file", side_effect=fake_convert):
                convert_folder(
                    input_dir,
                    output_dir,
                    ConvertOptions(
                        output_format="generic_tiff",
                        parallel_wsi=3,
                        memory_budget_mb=6144,
                    ),
                )

            self.assertEqual(task_budgets, [2048, 2048, 2048])


if __name__ == "__main__":
    unittest.main()
