from __future__ import annotations

import io
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from ibl2svs.backend_worker import (
    BackendWorker,
    main,
    options_from_request,
    serialize_batch,
    serialize_inspection,
    serialize_result,
)
from ibl2svs.models import BatchInspection, BatchResult, ConvertOptions, ConvertResult, InputInspection


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
                "output_format": "ome_tiff",
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

    def test_options_from_request_keeps_preflight_signatures(self) -> None:
        options = options_from_request(
            {"input_signatures": {"sample.kfbf": {"size": 123, "mtime_ns": "1786937375479896854"}}}
        )

        self.assertEqual(options.input_signatures["sample.kfbf"]["size"], 123)
        self.assertEqual(
            options.input_signatures["sample.kfbf"]["mtime_ns"],
            "1786937375479896854",
        )

    def test_serialize_inspection_preserves_nanosecond_timestamp_as_string(self) -> None:
        inspection = BatchInspection(
            Path("/input"),
            True,
            (
                InputInspection(
                    input_path=Path("/input/sample.kfbf"),
                    file_size=123,
                    file_mtime_ns=1_786_937_375_479_896_854,
                    input_format="kfb",
                    source_modality="fluorescence",
                    source_container="kfbf",
                    source_version="2.1",
                    source_codec="JPEG",
                    source_bit_depth=8,
                    field_count=1,
                    channel_count=1,
                    z_count=1,
                    t_count=1,
                    channel_definitions=(),
                    allowed_output_formats=("ome_tiff",),
                    incompatible_reasons={},
                ),
            ),
        )

        payload = serialize_inspection(inspection)

        self.assertEqual(payload["files"][0]["file_mtime_ns"], "1786937375479896854")

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
        self.assertEqual(payload["skipped_count"], 0)
        self.assertEqual(payload["results"][0]["input_path"], "sample.ibl")

    def test_worker_emits_json_lines(self) -> None:
        output = io.StringIO()
        worker = BackendWorker(stdin=io.StringIO(), stdout=output)

        worker.emit("log", message="hello")

        event = json.loads(output.getvalue())
        self.assertEqual(event["type"], "log")
        self.assertEqual(event["message"], "hello")

    def test_worker_protocol_error_keeps_inspection_job_id(self) -> None:
        request = {
            "type": "inspect",
            "job_id": "inspect-missing",
            "payload": {"input_dir": "/path/that/does/not/exist", "recursive": True},
        }
        output = io.StringIO()
        worker = BackendWorker(
            stdin=io.StringIO(json.dumps(request) + "\n"),
            stdout=output,
        )

        worker.start()

        events = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(events[-1]["type"], "inspection_error")
        self.assertEqual(events[-1]["job_id"], "inspect-missing")

    def test_main_reads_utf8_windows_paths_from_a_gbk_stream(self) -> None:
        input_dir = r"F:\C4.STCH\冰冻测试"
        request = {
            "type": "start",
            "payload": {"input_dir": input_dir, "output_dir": r"F:\output"},
        }
        stdin_buffer = io.BytesIO((json.dumps(request, ensure_ascii=False) + "\n").encode("utf-8"))
        stdout_buffer = io.BytesIO()
        stdin = io.TextIOWrapper(stdin_buffer, encoding="gbk")
        stdout = io.TextIOWrapper(stdout_buffer, encoding="gbk")

        with (
            mock.patch.object(sys, "stdin", stdin),
            mock.patch.object(sys, "stdout", stdout),
            mock.patch.object(BackendWorker, "handle_message", autospec=True) as handle_message,
        ):
            self.assertEqual(main(), 0)

        self.assertEqual(handle_message.call_args.args[1]["payload"]["input_dir"], input_dir)
        self.assertEqual(json.loads(stdout_buffer.getvalue().decode("utf-8"))["type"], "ready")

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
                            "output_format": "ome_tiff",
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

    def test_conversion_state_is_idle_before_terminal_event(self) -> None:
        worker = BackendWorker(stdin=io.StringIO(), stdout=io.StringIO())
        batch = BatchResult(total_files=0, success_count=0, failed_count=0)
        terminal_state: list[tuple[object, object, object]] = []
        original_emit = worker.emit

        def observe(event_type, **payload):
            if event_type == "done":
                terminal_state.append((worker._thread, worker._current_job_id, worker._current_job_kind))
            original_emit(event_type, **payload)

        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            worker._thread = threading.current_thread()
            worker._current_job_id = "job-1"
            worker._current_job_kind = "conversion"
            with (
                mock.patch.object(worker, "emit", side_effect=observe),
                mock.patch("ibl2svs.backend_worker.convert_folder", return_value=batch),
            ):
                worker._run_job("job-1", root, root, ConvertOptions())

        self.assertEqual(terminal_state, [(None, None, None)])

    def test_worker_cancel_sets_cancel_event(self) -> None:
        output = io.StringIO()
        worker = BackendWorker(stdin=io.StringIO(), stdout=output)
        worker._current_job_id = "job-1"

        worker.cancel_job("job-1")

        self.assertTrue(worker._cancel_event.is_set())
        event = json.loads(output.getvalue())
        self.assertEqual(event["type"], "log")

    def test_worker_inspection_emits_progress_and_done(self) -> None:
        output = io.StringIO()
        worker = BackendWorker(stdin=io.StringIO(), stdout=output)
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            inspection = BatchInspection(root, True, ())

            def fake_inspect(input_dir, recursive, progress_callback, **kwargs):
                kwargs["discovered_callback"](1, {"kfbf": 1})
                progress_callback(0, 1, "sample.kfbf")
                progress_callback(1, 1, "")
                return inspection

            with mock.patch("ibl2svs.backend_worker.inspect_inputs", side_effect=fake_inspect):
                worker.start_inspection(
                    {
                        "type": "inspect",
                        "job_id": "inspect-1",
                        "payload": {"input_dir": str(root), "recursive": True},
                    }
                )
                thread = worker._thread
                assert thread is not None
                thread.join(timeout=2)

        events = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(events[0]["type"], "inspection_started")
        self.assertEqual([event["type"] for event in events].count("inspection_progress"), 2)
        self.assertEqual([event["type"] for event in events].count("inspection_discovered"), 1)
        self.assertEqual(events[-1]["type"], "inspection_done")

    def test_inspection_state_is_idle_before_terminal_event(self) -> None:
        worker = BackendWorker(stdin=io.StringIO(), stdout=io.StringIO())
        terminal_state: list[tuple[object, object, object]] = []
        original_emit = worker.emit

        def observe(event_type, **payload):
            if event_type == "inspection_done":
                terminal_state.append((worker._thread, worker._current_job_id, worker._current_job_kind))
            original_emit(event_type, **payload)

        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            worker._thread = threading.current_thread()
            worker._current_job_id = "inspect-1"
            worker._current_job_kind = "inspection"
            with (
                mock.patch.object(worker, "emit", side_effect=observe),
                mock.patch(
                    "ibl2svs.backend_worker.inspect_inputs",
                    return_value=BatchInspection(root, True, ()),
                ),
            ):
                worker._run_inspection("inspect-1", root, True)

        self.assertEqual(terminal_state, [(None, None, None)])

    def test_worker_rejects_inspection_while_busy(self) -> None:
        output = io.StringIO()
        worker = BackendWorker(stdin=io.StringIO(), stdout=output)
        stop = threading.Event()
        worker._thread = threading.Thread(target=stop.wait)
        worker._thread.start()
        worker._current_job_id = "job-1"
        try:
            with tempfile.TemporaryDirectory() as tempdir:
                worker.start_inspection(
                    {
                        "type": "inspect",
                        "job_id": "inspect-1",
                        "payload": {"input_dir": tempdir},
                    }
                )
        finally:
            stop.set()
            worker._thread.join(timeout=2)

        event = json.loads(output.getvalue())
        self.assertEqual(event["type"], "inspection_error")
        self.assertEqual(event["message"], "worker is busy")

    def test_new_inspection_cancels_and_replaces_running_inspection(self) -> None:
        output = io.StringIO()
        worker = BackendWorker(stdin=io.StringIO(), stdout=output)
        first_started = threading.Event()
        second_done = threading.Event()

        def fake_inspect(input_dir, recursive, cancel_event, **kwargs):
            if Path(input_dir).name == "first":
                first_started.set()
                self.assertTrue(cancel_event.wait(timeout=2))
                from ibl2svs.inspection import InspectionCancelled

                raise InspectionCancelled("replaced")
            second_done.set()
            return BatchInspection(Path(input_dir), recursive, ())

        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            with mock.patch("ibl2svs.backend_worker.inspect_inputs", side_effect=fake_inspect):
                worker.start_inspection({
                    "type": "inspect",
                    "job_id": "inspect-first",
                    "payload": {"input_dir": str(first), "recursive": True},
                })
                self.assertTrue(first_started.wait(timeout=2))
                worker.start_inspection({
                    "type": "inspect",
                    "job_id": "inspect-second",
                    "payload": {"input_dir": str(second), "recursive": True},
                })
                self.assertTrue(second_done.wait(timeout=2))
                for _ in range(100):
                    with worker._lock:
                        active = worker._thread is not None
                    if not active:
                        break
                    threading.Event().wait(0.01)

        events = [json.loads(line) for line in output.getvalue().splitlines()]
        started_jobs = [event["job_id"] for event in events if event["type"] == "inspection_started"]
        done_jobs = [event["job_id"] for event in events if event["type"] == "inspection_done"]
        self.assertEqual(started_jobs, ["inspect-first", "inspect-second"])
        self.assertEqual(done_jobs, ["inspect-second"])


if __name__ == "__main__":
    unittest.main()
