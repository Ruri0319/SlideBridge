from __future__ import annotations

import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from ibl2svs.inspection import InspectionCancelled, inspect_inputs
from ibl2svs.models import InputInspection


def _inspection(path: Path) -> InputInspection:
    return InputInspection(
        input_path=path,
        file_size=1,
        file_mtime_ns=1,
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
    )


class InspectionPerformanceTests(unittest.TestCase):
    def test_parallel_inspection_is_bounded_and_preserves_order(self) -> None:
        paths = [Path(f"/input/{index:02d}.kfbf") for index in range(8)]
        lock = threading.Lock()
        active = 0
        max_active = 0
        completed: list[str] = []
        discovered: list[tuple[int, dict[str, int]]] = []

        def fake_inspect(path: Path) -> InputInspection:
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.02 if path.name != "00.kfbf" else 0.05)
            with lock:
                active -= 1
            return _inspection(path)

        with mock.patch("ibl2svs.inspection.find_inspectable_files", return_value=paths), mock.patch(
            "ibl2svs.inspection.inspect_file", side_effect=fake_inspect
        ):
            result = inspect_inputs(
                "/input",
                discovered_callback=lambda total, counts: discovered.append((total, counts)),
                file_callback=lambda done, total, item: completed.append(item.input_path.name),
            )

        self.assertGreater(max_active, 1)
        self.assertLessEqual(max_active, 4)
        self.assertEqual([item.input_path for item in result.files], paths)
        self.assertEqual(len(completed), len(paths))
        self.assertEqual(discovered, [(8, {"kfbf": 8})])

    def test_parallel_inspection_is_at_least_twice_as_fast_as_serial(self) -> None:
        paths = [Path(f"/input/{index:02d}.kfbf") for index in range(8)]

        def fake_inspect(path: Path) -> InputInspection:
            time.sleep(0.04)
            return _inspection(path)

        with mock.patch("ibl2svs.inspection.find_inspectable_files", return_value=paths), mock.patch(
            "ibl2svs.inspection.inspect_file", side_effect=fake_inspect
        ):
            serial_started = time.perf_counter()
            inspect_inputs("/input", max_workers=1)
            serial_seconds = time.perf_counter() - serial_started

            parallel_started = time.perf_counter()
            inspect_inputs("/input", max_workers=4)
            parallel_seconds = time.perf_counter() - parallel_started

        self.assertLess(parallel_seconds, serial_seconds * 0.5)

    def test_parallel_inspection_honors_cancellation(self) -> None:
        paths = [Path(f"/input/{index:02d}.kfbf") for index in range(8)]
        cancel_event = threading.Event()

        def fake_inspect(path: Path) -> InputInspection:
            time.sleep(0.01)
            return _inspection(path)

        with mock.patch("ibl2svs.inspection.find_inspectable_files", return_value=paths), mock.patch(
            "ibl2svs.inspection.inspect_file", side_effect=fake_inspect
        ):
            with self.assertRaises(InspectionCancelled):
                inspect_inputs(
                    "/input",
                    cancel_event=cancel_event,
                    file_callback=lambda done, total, item: cancel_event.set(),
                )

    def test_replacement_inspection_keeps_global_parallel_limit(self) -> None:
        cancel_event = threading.Event()
        old_release = threading.Event()
        new_release = threading.Event()
        first_started = threading.Event()
        first_returned = threading.Event()
        lock = threading.Lock()
        active = 0
        max_active = 0

        def fake_files(root: str | Path, recursive: bool = True) -> list[Path]:
            del recursive
            root = Path(root)
            return [root / f"{index}.kfbf" for index in range(8)]

        def fake_inspect(path: Path) -> InputInspection:
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            try:
                if path.parent.name == "first":
                    first_started.set()
                    if path.name == "0.kfbf":
                        self.assertTrue(cancel_event.wait(timeout=2))
                    else:
                        self.assertTrue(old_release.wait(timeout=2))
                else:
                    self.assertTrue(new_release.wait(timeout=2))
                return _inspection(path)
            finally:
                with lock:
                    active -= 1

        def run_first() -> None:
            try:
                inspect_inputs("/input/first", cancel_event=cancel_event)
            except InspectionCancelled:
                pass
            finally:
                first_returned.set()

        with mock.patch("ibl2svs.inspection.find_inspectable_files", side_effect=fake_files), mock.patch(
            "ibl2svs.inspection.inspect_file", side_effect=fake_inspect
        ):
            first = threading.Thread(target=run_first)
            first.start()
            self.assertTrue(first_started.wait(timeout=2))
            cancel_event.set()
            self.assertTrue(first_returned.wait(timeout=2))

            second = threading.Thread(target=lambda: inspect_inputs("/input/second"))
            second.start()
            time.sleep(0.05)
            old_release.set()
            new_release.set()
            first.join(timeout=2)
            second.join(timeout=2)

        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertLessEqual(max_active, 4)


if __name__ == "__main__":
    unittest.main()
