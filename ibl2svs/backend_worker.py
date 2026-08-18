from __future__ import annotations

import json
import sys
import threading
import time
import traceback
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

from .app_meta import runtime_banner
from .converter import convert_folder
from .inspection import inspect_inputs
from .inspection import InspectionCancelled
from .models import (
    BatchInspection,
    BatchResult,
    ChannelDefinition,
    ConvertOptions,
    ConvertResult,
    InputInspection,
)
from .system_metrics import ProcessMetricsSampler


class WorkerProtocolError(ValueError):
    pass


def _configure_protocol_stdio() -> None:
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def serialize_result(result: ConvertResult) -> dict[str, Any]:
    return {
        "input_path": str(result.input_path),
        "output_path": str(result.output_path) if result.output_path else None,
        "output_files": [str(path) for path in (result.output_files or [])],
        "success": result.success,
        "input_format": result.input_format,
        "status": result.status,
        "output_format": result.output_format,
        "backend": result.backend,
        "width": result.width,
        "height": result.height,
        "pyramid_levels": result.pyramid_levels,
        "mpp": result.mpp,
        "duration_sec": result.duration_sec,
        "peak_memory_mb": result.peak_memory_mb,
        "native_path": result.native_path,
        "native_level_dimensions": result.native_level_dimensions,
        "native_resource_dimensions": result.native_resource_dimensions,
        "native_tile_mode": result.native_tile_mode,
        "native_fallback_reason": result.native_fallback_reason,
        "source_container": result.source_container,
        "source_version": result.source_version,
        "source_codec": result.source_codec,
        "source_bit_depth": result.source_bit_depth,
        "source_channel_count": result.source_channel_count,
        "source_axes": result.source_axes,
        "compatibility_level": result.compatibility_level,
        "diagnostic_code": result.diagnostic_code,
        "diagnostic_stage": result.diagnostic_stage,
        "svs_omitted_native_data": result.svs_omitted_native_data,
        "source_modality": result.source_modality,
        "channel_definitions": result.channel_definitions,
        "channel_identity_source": result.channel_identity_source,
        "channel_override_applied": result.channel_override_applied,
        "skipped_reason": result.skipped_reason,
        "failure_stage": result.failure_stage,
        "error_code": result.error_code,
        "error": result.error,
    }


def serialize_batch(batch: BatchResult) -> dict[str, Any]:
    return {
        "total_files": batch.total_files,
        "success_count": batch.success_count,
        "failed_count": batch.failed_count,
        "skipped_count": batch.skipped_count,
        "cancelled_count": batch.cancelled_count,
        "cancelled": batch.cancelled,
        "report_path": str(batch.report_path) if batch.report_path else None,
        "results": [serialize_result(result) for result in batch.results],
    }


def serialize_inspection(inspection: BatchInspection) -> dict[str, Any]:
    payload = _json_safe(inspection)
    for file_payload in payload["files"]:
        # Nanosecond timestamps exceed JavaScript's safe integer range.
        file_payload["file_mtime_ns"] = str(file_payload["file_mtime_ns"])
    return payload


def serialize_input_inspection(inspection: InputInspection) -> dict[str, Any]:
    payload = _json_safe(inspection)
    payload["file_mtime_ns"] = str(payload["file_mtime_ns"])
    return payload


def input_inspection_from_request(payload: dict[str, Any]) -> InputInspection:
    return InputInspection(
        input_path=Path(str(payload["input_path"])),
        file_size=int(payload.get("file_size", 0)),
        file_mtime_ns=int(payload.get("file_mtime_ns", 0)),
        input_format=str(payload.get("input_format", "unsupported")),
        source_modality=str(payload.get("source_modality", "unknown")),  # type: ignore[arg-type]
        source_container=payload.get("source_container"),
        source_version=payload.get("source_version"),
        source_codec=payload.get("source_codec"),
        source_bit_depth=int(payload.get("source_bit_depth", 0)),
        field_count=int(payload.get("field_count", 0)),
        channel_count=int(payload.get("channel_count", 0)),
        z_count=int(payload.get("z_count", 0)),
        t_count=int(payload.get("t_count", 0)),
        channel_definitions=tuple(
            ChannelDefinition(
                index=int(item.get("index", index)),
                name=str(item.get("name", "")),
                fluor=item.get("fluor"),
                color=tuple(int(value) for value in item.get("color", (255, 255, 255))),  # type: ignore[arg-type]
                excitation_nm=item.get("excitation_nm"),
                emission_nm=item.get("emission_nm"),
                exposure=item.get("exposure"),
                identity_source=str(item.get("identity_source", "unknown")),  # type: ignore[arg-type]
            )
            for index, item in enumerate(payload.get("channel_definitions", []))
        ),
        allowed_output_formats=tuple(payload.get("allowed_output_formats", [])),  # type: ignore[arg-type]
        incompatible_reasons=dict(payload.get("incompatible_reasons", {}) or {}),
        error=payload.get("error"),
    )


def _bounded_int(payload: dict[str, Any], key: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(payload.get(key, default))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def options_from_request(payload: dict[str, Any]) -> ConvertOptions:
    output_format = str(payload.get("output_format", "ome_tiff"))
    if output_format not in {"ome_tiff", "svs", "fluorescence_svs", "afi"}:
        raise WorkerProtocolError(f"unsupported output_format: {output_format}")
    return ConvertOptions(
        recursive=bool(payload.get("recursive", True)),
        output_format=output_format,  # type: ignore[arg-type]
        memory_budget_mb=_bounded_int(payload, "memory_budget_mb", 6144, 1024, 65536),
        tile_size=_bounded_int(payload, "tile_size", 256, 16, 4096),
        jpeg_quality=_bounded_int(payload, "jpeg_quality", 90, 1, 100),
        parallel_wsi=_bounded_int(payload, "parallel_wsi", 1, 1, 8),
        selected_input_paths=(
            tuple(str(path) for path in payload.get("selected_input_paths", []))
            if payload.get("selected_input_paths") is not None
            else None
        ),
        convert_compatible_only=bool(payload.get("convert_compatible_only", False)),
        channel_overrides=dict(payload.get("channel_overrides", {}) or {}),
        input_signatures=dict(payload.get("input_signatures", {}) or {}),
        preflight_files=(
            tuple(input_inspection_from_request(item) for item in payload.get("preflight_files", []))
            if payload.get("preflight_files") is not None
            else None
        ),
    )


class BackendWorker:
    def __init__(self, stdin: TextIO | None = None, stdout: TextIO | None = None):
        self.stdin = stdin or sys.stdin
        self.stdout = stdout or sys.stdout
        self._lock = threading.Lock()
        self._emit_lock = threading.Lock()
        self._cancel_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._current_job_id: str | None = None
        self._current_job_kind: str | None = None
        self._pending_inspection: dict[str, Any] | None = None

    def emit(self, event_type: str, **payload: Any) -> None:
        event = {"type": event_type, **payload}
        with self._emit_lock:
            print(json.dumps(_json_safe(event), ensure_ascii=False), file=self.stdout, flush=True)

    def start(self) -> None:
        self.emit("ready", banner=runtime_banner())
        for raw_line in self.stdin:
            line = raw_line.strip()
            if not line:
                continue
            message: dict[str, Any] = {}
            try:
                parsed = json.loads(line)
                if not isinstance(parsed, dict):
                    raise WorkerProtocolError("worker message must be a JSON object")
                message = parsed
                self.handle_message(message)
            except Exception as exc:
                event_type = "inspection_error" if message.get("type") == "inspect" else "error"
                self.emit(
                    event_type,
                    job_id=message.get("job_id"),
                    message=str(exc),
                    traceback=traceback.format_exc(),
                )

    def handle_message(self, message: dict[str, Any]) -> None:
        message_type = message.get("type")
        if message_type == "start":
            self.start_job(message)
        elif message_type == "inspect":
            self.start_inspection(message)
        elif message_type == "cancel":
            self.cancel_job(str(message.get("job_id") or ""))
        elif message_type == "ping":
            self.emit("ready", banner=runtime_banner())
        else:
            raise WorkerProtocolError(f"unknown message type: {message_type}")

    def start_job(self, message: dict[str, Any]) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                self.emit("error", job_id=self._current_job_id, message="conversion already running")
                return
            job_id = str(message.get("job_id") or int(time.time() * 1000))
            payload = message.get("payload", message)
            input_dir = Path(str(payload.get("input_dir", ""))).expanduser()
            output_dir = Path(str(payload.get("output_dir", ""))).expanduser()
            if not input_dir.is_dir():
                raise WorkerProtocolError(f"invalid input_dir: {input_dir}")
            output_dir.mkdir(parents=True, exist_ok=True)
            if not output_dir.is_dir():
                raise WorkerProtocolError(f"invalid output_dir: {output_dir}")
            options = options_from_request(payload)
            self._cancel_event = threading.Event()
            self._current_job_id = job_id
            self._current_job_kind = "conversion"
            self._thread = threading.Thread(
                target=self._run_job,
                args=(job_id, input_dir, output_dir, options),
                daemon=True,
            )
            self.emit("started", job_id=job_id)
            self._thread.start()

    def start_inspection(self, message: dict[str, Any]) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                if self._current_job_kind == "inspection":
                    self._pending_inspection = message
                    self._cancel_event.set()
                    return
                self.emit("inspection_error", job_id=message.get("job_id"), message="worker is busy")
                return
            job_id = str(message.get("job_id") or int(time.time() * 1000))
            payload = message.get("payload", message)
            input_dir = Path(str(payload.get("input_dir", ""))).expanduser()
            if not input_dir.is_dir():
                raise WorkerProtocolError(f"invalid input_dir: {input_dir}")
            recursive = bool(payload.get("recursive", True))
            self._cancel_event = threading.Event()
            self._current_job_id = job_id
            self._current_job_kind = "inspection"
            self._thread = threading.Thread(
                target=self._run_inspection,
                args=(job_id, input_dir, recursive),
                daemon=True,
            )
            self.emit("inspection_started", job_id=job_id)
            self._thread.start()

    def cancel_job(self, job_id: str) -> None:
        with self._lock:
            if self._current_job_id is None:
                self.emit("log", job_id=job_id or None, message="没有正在运行的转换任务")
                return
            if job_id and job_id != self._current_job_id:
                self.emit("error", job_id=job_id, message="job_id does not match running task")
                return
            self._cancel_event.set()
            self.emit("log", job_id=self._current_job_id, message="收到取消请求，等待当前步骤安全结束")

    def _start_performance_monitor(self, job_id: str) -> tuple[threading.Event, threading.Thread]:
        stop_event = threading.Event()
        sampler = ProcessMetricsSampler()

        def emit_sample() -> None:
            memory_mb, cpu_percent = sampler.sample()
            self.emit(
                "performance",
                job_id=job_id,
                memory_mb=round(memory_mb, 1),
                cpu_percent=round(cpu_percent, 1),
            )

        emit_sample()

        def monitor() -> None:
            while not stop_event.wait(1.0):
                emit_sample()

        thread = threading.Thread(target=monitor, daemon=True, name="slidebridge-performance")
        thread.start()
        return stop_event, thread

    def _run_job(self, job_id: str, input_dir: Path, output_dir: Path, options: ConvertOptions) -> None:
        log_path = output_dir / f"slidebridge_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_lock = threading.Lock()
        metrics_stop, metrics_thread = self._start_performance_monitor(job_id)

        def log(message: str) -> None:
            formatted = f"[{datetime.now().strftime('%H:%M:%S')}] {message}"
            with log_lock:
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(formatted + "\n")
            self.emit("log", job_id=job_id, message=formatted)

        terminal_type = "done"
        terminal_payload: dict[str, Any]
        try:
            self.emit("report_path", job_id=job_id, path=str(log_path))
            batch = convert_folder(
                input_dir,
                output_dir,
                options,
                logger=log,
                overall_callback=lambda done, total, current: self.emit(
                    "overall",
                    job_id=job_id,
                    done=done,
                    total=total,
                    current=current,
                ),
                progress_callback=lambda current, level, done, total, overall_done, overall_total: self.emit(
                    "file_progress",
                    job_id=job_id,
                    current=current,
                    level=level,
                    done=done,
                    total=total,
                    overall_done=overall_done,
                    overall_total=overall_total,
                ),
                cancel_event=self._cancel_event,
            )
            terminal_payload = {"batch": serialize_batch(batch)}
        except Exception as exc:
            terminal_type = "error"
            terminal_payload = {
                "message": str(exc),
                "traceback": traceback.format_exc(),
                "diagnostic_code": getattr(exc, "diagnostic_code", None),
                "diagnostic_stage": getattr(exc, "diagnostic_stage", None),
            }
        finally:
            metrics_stop.set()
            metrics_thread.join(timeout=2)
            with self._lock:
                self._current_job_id = None
                self._current_job_kind = None
                self._thread = None
        self.emit(terminal_type, job_id=job_id, **terminal_payload)

    def _run_inspection(self, job_id: str, input_dir: Path, recursive: bool) -> None:
        started_at = time.perf_counter()
        terminal_type: str | None = None
        terminal_payload: dict[str, Any] = {}
        try:
            inspection = inspect_inputs(
                input_dir,
                recursive=recursive,
                progress_callback=lambda done, total, current: self.emit(
                    "inspection_progress",
                    job_id=job_id,
                    done=done,
                    total=total,
                    current=current,
                ),
                discovered_callback=lambda total, format_counts: self.emit(
                    "inspection_discovered",
                    job_id=job_id,
                    total=total,
                    format_counts=format_counts,
                ),
                file_callback=lambda done, total, result: self.emit(
                    "inspection_file_done",
                    job_id=job_id,
                    done=done,
                    total=total,
                    current=str(result.input_path),
                    file=serialize_input_inspection(result),
                ),
                cancel_event=self._cancel_event,
            )
            terminal_type = "inspection_done"
            terminal_payload = {
                "inspection": serialize_inspection(inspection),
                "duration_ms": round((time.perf_counter() - started_at) * 1000, 1),
            }
        except InspectionCancelled:
            pass
        except Exception as exc:
            terminal_type = "inspection_error"
            terminal_payload = {"message": str(exc), "traceback": traceback.format_exc()}
        finally:
            pending = None
            with self._lock:
                self._current_job_id = None
                self._current_job_kind = None
                self._thread = None
                pending = self._pending_inspection
                self._pending_inspection = None
            if terminal_type is not None:
                self.emit(terminal_type, job_id=job_id, **terminal_payload)
            if pending is not None:
                self.start_inspection(pending)


def main() -> int:
    _configure_protocol_stdio()
    BackendWorker().start()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
