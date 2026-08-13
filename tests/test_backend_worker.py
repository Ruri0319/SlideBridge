from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ibl2svs.backend_worker import BackendWorker, options_from_request, serialize_batch, serialize_result
from ibl2svs.models import BatchResult, ConvertOptions, ConvertResult


class BackendWorkerTests(unittest.TestCase):
    def test_options_from_request_uses_frontend_defaults(self) -> None:
        options = options_from_request({"output_format": "svs"})

        self.assertIsInstance(options, ConvertOptions)
        self.assertEqual(options.output_format, "svs")
        self.assertTrue(options.recursive)
        self.assertEqual(options.memory_budget_mb, 6144)
        self.assertEqual(options.tile_size, 256)
        self.assertEqual(options.jpeg_quality, 90)
        self.assertEqual(options.parallel_wsi, 1)

    def test_options_from_request_clamps_adjustable_backend_settings(self) -> None:
        options = options_from_request(
            {
                "output_format": "generic_tiff",
                "memory_budget_mb": 512,
                "tile_size": 8,
                "jpeg_quality": 120,
                "parallel_wsi": 9,
            }
        )

        self.assertEqual(options.memory_budget_mb, 1024)
        self.assertEqual(options.tile_size, 16)
        self.assertEqual(options.jpeg_quality, 100)
        self.assertEqual(options.parallel_wsi, 8)

    def test_options_from_request_accepts_parallel_wsi_upper_limit(self) -> None:
        options = options_from_request({"parallel_wsi": 8})

        self.assertEqual(options.parallel_wsi, 8)
        self.assertEqual(options.resolved_parallel_wsi(), 8)

    def test_serialize_result_converts_paths_to_strings(self) -> None:
        result = ConvertResult(
            input_path=Path("sample.svs"),
            output_path=Path("sample.tif"),
            success=True,
            input_format="svs",
            backend="tifffile-streaming",
            width=100,
            height=50,
        )

        payload = serialize_result(result)

        self.assertEqual(payload["input_path"], "sample.svs")
        self.assertEqual(payload["output_path"], "sample.tif")
        self.assertEqual(payload["input_format"], "svs")
        self.assertEqual(payload["backend"], "tifffile-streaming")

    def test_serialize_batch_includes_report_path_and_results(self) -> None:
        batch = BatchResult(
            total_files=1,
            success_count=1,
            failed_count=0,
            report_path=Path("conversion_report.csv"),
            results=[
                ConvertResult(
                    input_path=Path("sample.ibl"),
                    output_path=Path("sample.tif"),
                    success=True,
                )
            ],
        )

        payload = serialize_batch(batch)

        self.assertEqual(payload["report_path"], "conversion_report.csv")
        self.assertEqual(payload["results"][0]["input_path"], "sample.ibl")

    def test_worker_emits_json_lines(self) -> None:
        output = io.StringIO()
        worker = BackendWorker(stdin=io.StringIO(), stdout=output)

        worker.emit("log", message="hello")

        event = json.loads(output.getvalue())
        self.assertEqual(event["type"], "log")
        self.assertEqual(event["message"], "hello")

    def test_worker_start_job_invokes_convert_folder_and_emits_done(self) -> None:
        output = io.StringIO()
        worker = BackendWorker(stdin=io.StringIO(), stdout=output)
        batch = BatchResult(total_files=0, success_count=0, failed_count=0, report_path=Path("report.csv"))

        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()

            with mock.patch("ibl2svs.backend_worker.convert_folder", return_value=batch):
                worker.start_job(
                    {
                        "type": "start",
                        "job_id": "job-1",
                        "payload": {
                            "input_dir": str(input_dir),
                            "output_dir": str(output_dir),
                            "output_format": "generic_tiff",
                        },
                    }
                )
                assert worker._thread is not None
                worker._thread.join(timeout=2)

        events = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(events[0]["type"], "started")
        performance = next(event for event in events if event["type"] == "performance")
        self.assertGreaterEqual(performance["memory_mb"], 0)
        self.assertGreaterEqual(performance["cpu_percent"], 0)
        self.assertLessEqual(performance["cpu_percent"], 100)
        self.assertEqual(events[-1]["type"], "done")
        self.assertEqual(events[-1]["job_id"], "job-1")

    def test_worker_cancel_sets_cancel_event(self) -> None:
        output = io.StringIO()
        worker = BackendWorker(stdin=io.StringIO(), stdout=output)
        worker._current_job_id = "job-1"

        worker.cancel_job("job-1")

        self.assertTrue(worker._cancel_event.is_set())
        event = json.loads(output.getvalue())
        self.assertEqual(event["type"], "log")


if __name__ == "__main__":
    unittest.main()
