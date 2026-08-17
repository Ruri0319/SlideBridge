from __future__ import annotations

import csv
from dataclasses import asdict
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import replace
from pathlib import Path

from .app_meta import runtime_banner
from .inspection import (
    SUPPORTED_INPUT_SUFFIXES,
    apply_channel_overrides,
    channel_definitions,
    detect_input_format,
    find_inspectable_files,
    inspect_file,
    open_slide,
    output_eligibility,
)
from .models import BatchResult, ConvertOptions, ConvertResult, OutputFormat
from .writer import WriteImageError, write_image


class OutputCompatibilityError(RuntimeError):
    diagnostic_code = "incompatible_output"
    diagnostic_stage = "preflight"


def _safe_unlink(path: Path, retries: int = 8, delay: float = 0.15) -> None:
    for attempt in range(retries):
        try:
            if path.exists():
                path.unlink()
            return
        except (PermissionError, OSError):
            if attempt == retries - 1:
                return
            time.sleep(delay)


def find_ibl_files(input_dir: str | Path, recursive: bool = True) -> list[Path]:
    input_dir = Path(input_dir)
    pattern = "**/*" if recursive else "*"
    files = [
        path
        for path in input_dir.glob(pattern)
        if path.is_file() and path.suffix.lower() == ".ibl"
    ]
    return sorted(files)


def find_convertible_files(
    input_dir: str | Path,
    recursive: bool = True,
    output_format: str = "ome_tiff",
) -> list[Path]:
    del output_format
    return find_inspectable_files(input_dir, recursive=recursive)


def _output_suffix(output_format: str) -> str:
    if output_format == "ome_tiff":
        return ".ome.tif"
    if output_format in {"svs", "fluorescence_svs"}:
        return ".svs"
    if output_format == "afi":
        return ".afi"
    raise ValueError(f"不支持的输出格式: {output_format}")


def build_output_path(
    input_path: Path,
    input_root: Path,
    output_root: Path,
    output_format: str = "ome_tiff",
) -> Path:
    relative = input_path.relative_to(input_root)
    candidate = output_root / relative
    suffix = _output_suffix(output_format)
    candidate = candidate.with_suffix(suffix)
    candidate.parent.mkdir(parents=True, exist_ok=True)
    if not candidate.exists():
        return candidate

    base_name = candidate.name[: -len(suffix)]
    index = 1
    while True:
        alt = candidate.with_name(f"{base_name}_{index}{suffix}")
        if not alt.exists():
            return alt
        index += 1


def _build_output_plan(
    files: list[Path],
    input_root: Path,
    output_root: Path,
    output_format: str,
) -> list[tuple[int, Path, Path]]:
    reserved: set[Path] = set()
    planned: list[tuple[int, Path, Path]] = []
    suffix = _output_suffix(output_format)
    for index, input_path in enumerate(files, start=1):
        relative = input_path.relative_to(input_root)
        candidate = (output_root / relative).with_suffix(suffix)
        candidate.parent.mkdir(parents=True, exist_ok=True)
        if candidate.exists() or candidate in reserved:
            stem = candidate.name[: -len(suffix)]
            alt_index = 1
            while True:
                alt = candidate.with_name(f"{stem}_{alt_index}{suffix}")
                if not alt.exists() and alt not in reserved:
                    candidate = alt
                    break
                alt_index += 1
        reserved.add(candidate)
        planned.append((index, input_path, candidate))
    return planned


def write_report(results: list[ConvertResult], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "input_path",
                "input_format",
                "output_path",
                "output_files",
                "success",
                "status",
                "output_format",
                "backend",
                "width",
                "height",
                "level_dimensions",
                "pyramid_levels",
                "mpp",
                "duration_sec",
                "read_decode_sec",
                "main_write_sec",
                "pyramid_sec",
                "thumbnail_sec",
                "encode_sec",
                "writer_wait_sec",
                "peak_memory_mb",
                "avg_cpu_percent",
                "svs_is_bigtiff",
                "svs_label_dimensions",
                "svs_macro_dimensions",
                "openslide_vendor",
                "svs_photometric_pages",
                "svs_finalize_backend",
                "max_level_reached",
                "native_path",
                "native_level_dimensions",
                "native_resource_dimensions",
                "native_tile_mode",
                "native_fallback_reason",
                "source_container",
                "source_version",
                "source_codec",
                "source_bit_depth",
                "source_channel_count",
                "source_axes",
                "compatibility_level",
                "diagnostic_code",
                "diagnostic_stage",
                "svs_omitted_native_data",
                "source_modality",
                "channel_definitions",
                "channel_identity_source",
                "channel_override_applied",
                "skipped_reason",
                "failure_stage",
                "error_code",
                "error",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    str(result.input_path),
                    result.input_format,
                    str(result.output_path) if result.output_path else "",
                    "|".join(str(path) for path in (result.output_files or [])),
                    result.success,
                    result.status,
                    result.output_format,
                    result.backend,
                    result.width or "",
                    result.height or "",
                    "|".join(f"{w}x{h}" for w, h in (result.level_dimensions or [])),
                    result.pyramid_levels or "",
                    result.mpp or "",
                    f"{result.duration_sec:.3f}",
                    f"{result.read_decode_sec:.3f}",
                    f"{result.main_write_sec:.3f}",
                    f"{result.pyramid_sec:.3f}",
                    f"{result.thumbnail_sec:.3f}",
                    f"{result.encode_sec:.3f}",
                    f"{result.writer_wait_sec:.3f}",
                    f"{result.peak_memory_mb:.3f}",
                    f"{result.avg_cpu_percent:.3f}",
                    result.svs_is_bigtiff if result.svs_is_bigtiff is not None else "",
                    f"{result.svs_label_dimensions[0]}x{result.svs_label_dimensions[1]}" if result.svs_label_dimensions else "",
                    f"{result.svs_macro_dimensions[0]}x{result.svs_macro_dimensions[1]}" if result.svs_macro_dimensions else "",
                    result.openslide_vendor or "",
                    "|".join(result.svs_photometric_pages or []),
                    result.svs_finalize_backend or "",
                    result.max_level_reached or "",
                    result.native_path,
                    "|".join(f"{w}x{h}" for w, h in (result.native_level_dimensions or [])),
                    str(result.native_resource_dimensions or ""),
                    result.native_tile_mode or "",
                    result.native_fallback_reason or "",
                    result.source_container or "",
                    result.source_version or "",
                    result.source_codec or "",
                    result.source_bit_depth if result.source_bit_depth is not None else "",
                    result.source_channel_count if result.source_channel_count is not None else "",
                    result.source_axes or "",
                    result.compatibility_level or "",
                    result.diagnostic_code or "",
                    result.diagnostic_stage or "",
                    result.svs_omitted_native_data or "",
                    result.source_modality or "",
                    str(result.channel_definitions or ""),
                    "|".join(result.channel_identity_source or []),
                    result.channel_override_applied,
                    result.skipped_reason or "",
                    result.failure_stage or "",
                    result.error_code or "",
                    result.error or "",
                ]
            )


def _afi_perf(slide, options: ConvertOptions) -> dict:
    return {
        "backend": options.performance_backend,
        "current_stage": "初始化 AFI",
        "level_dimensions": list(getattr(slide, "level_dimensions", [])),
        "read_decode_sec": 0.0,
        "main_write_sec": 0.0,
        "pyramid_sec": 0.0,
        "thumbnail_sec": 0.0,
        "encode_sec": 0.0,
        "writer_wait_sec": 0.0,
        "peak_memory_mb": 0.0,
        "avg_cpu_percent": 0.0,
        "native_path": True,
        "native_level_dimensions": list(getattr(slide, "level_dimensions", [])),
        "native_resource_dimensions": dict(getattr(slide, "native_resource_dimensions", {})),
        "native_tile_mode": None,
        "native_fallback_reason": None,
        "source_container": getattr(slide, "source_container", None),
        "source_version": getattr(slide, "source_version", None),
        "source_codec": getattr(slide, "source_codec", None),
        "source_bit_depth": getattr(slide, "source_bit_depth", None),
        "source_channel_count": getattr(slide, "native_channel_count", None),
        "source_axes": getattr(slide, "native_axes", None),
        "compatibility_level": getattr(slide, "compatibility_level", None),
        "diagnostic_code": None,
        "diagnostic_stage": None,
        "failure_stage": None,
    }


def convert_file(
    input_path: str | Path,
    output_path: str | Path,
    options: ConvertOptions,
    logger=None,
    progress_callback=None,
    cancel_event=None,
) -> ConvertResult:
    input_path = Path(input_path)
    output_path = Path(output_path)
    input_format = detect_input_format(input_path)
    target_suffix = _output_suffix(options.output_format)
    if not output_path.name.lower().endswith(target_suffix):
        output_path = output_path.with_suffix(target_suffix)
    temp_output_path = output_path.with_suffix(f"{output_path.suffix}.part")
    start = time.perf_counter()
    error_code = None
    source_summary: dict = {}
    definitions = []
    perf: dict = {}
    output_files: list[Path] = []

    def existing_output_path() -> Path | None:
        return output_path if output_path.exists() else None

    try:
        if input_format == "unsupported":
            raise RuntimeError(f"不支持的输入格式: {input_path.suffix or input_path.name}")

        if temp_output_path.exists():
            _safe_unlink(temp_output_path)

        slide_context = open_slide(input_path, cache_size=options.cache_blocks_per_row)

        with slide_context as slide:
            override = options.channel_overrides.get(str(input_path))
            if override is None:
                override = options.channel_overrides.get(str(input_path.resolve()))
            definitions = apply_channel_overrides(slide, override)
            source_summary = {
                "source_modality": getattr(slide, "modality", "unknown"),
                "source_container": getattr(slide, "source_container", None),
                "source_version": getattr(slide, "source_version", None),
                "source_codec": getattr(slide, "source_codec", None),
                "source_bit_depth": getattr(slide, "source_bit_depth", None),
                "source_channel_count": getattr(slide, "native_channel_count", None),
                "source_axes": getattr(slide, "native_axes", None),
                "compatibility_level": getattr(slide, "compatibility_level", None),
            }
            allowed_formats, incompatible_reasons = output_eligibility(slide)
            if options.output_format not in allowed_formats:
                reason = incompatible_reasons.get(options.output_format, "输入与所选输出格式不兼容")
                raise OutputCompatibilityError(reason)
            if logger:
                logger(f"开始转换: {input_path}")

            def writer_progress(
                level_name: str,
                done: int,
                total: int,
                overall_done: int,
                overall_total: int,
            ) -> None:
                if progress_callback:
                    progress_callback(str(input_path), level_name, done, total, overall_done, overall_total)

            if options.output_format == "afi":
                from .afi_writer import write_afi

                perf = _afi_perf(slide, options)
                pyramid_levels, output_files = write_afi(
                    slide,
                    output_path,
                    options,
                    perf,
                    progress_callback=writer_progress,
                    cancel_event=cancel_event,
                )
                output_path = Path(perf["primary_output_path"])
            else:
                pyramid_levels, perf = write_image(
                    slide,
                    temp_output_path,
                    options,
                    progress_callback=writer_progress,
                    cancel_event=cancel_event,
                )
                temp_output_path.replace(output_path)
                output_files = [output_path]
            duration = time.perf_counter() - start
            result = ConvertResult(
                input_path=input_path,
                output_path=output_path,
                success=True,
                input_format=input_format,
                status="success",
                output_format=options.output_format,
                backend=perf.get("backend", options.performance_backend),
                width=slide.width,
                height=slide.height,
                level_dimensions=perf.get("level_dimensions"),
                pyramid_levels=pyramid_levels,
                mpp=slide.base_info.mpp,
                duration_sec=duration,
                read_decode_sec=perf.get("read_decode_sec", 0.0),
                main_write_sec=perf.get("main_write_sec", 0.0),
                pyramid_sec=perf.get("pyramid_sec", 0.0),
                thumbnail_sec=perf.get("thumbnail_sec", 0.0),
                encode_sec=perf.get("encode_sec", 0.0),
                writer_wait_sec=perf.get("writer_wait_sec", 0.0),
                peak_memory_mb=perf.get("peak_memory_mb", 0.0),
                avg_cpu_percent=perf.get("avg_cpu_percent", 0.0),
                svs_is_bigtiff=perf.get("svs_is_bigtiff"),
                svs_label_dimensions=perf.get("svs_label_dimensions"),
                svs_macro_dimensions=perf.get("svs_macro_dimensions"),
                openslide_vendor=perf.get("openslide_vendor"),
                svs_photometric_pages=perf.get("svs_photometric_pages"),
                svs_finalize_backend=perf.get("svs_finalize_backend"),
                max_level_reached=perf.get("max_level_reached"),
                failure_stage=perf.get("failure_stage"),
                native_path=bool(perf.get("native_path", False)),
                native_level_dimensions=perf.get("native_level_dimensions"),
                native_resource_dimensions=perf.get("native_resource_dimensions"),
                native_tile_mode=perf.get("native_tile_mode"),
                native_fallback_reason=perf.get("native_fallback_reason"),
                source_container=perf.get("source_container"),
                source_version=perf.get("source_version"),
                source_codec=perf.get("source_codec"),
                source_bit_depth=perf.get("source_bit_depth"),
                source_channel_count=perf.get("source_channel_count"),
                source_axes=perf.get("source_axes"),
                compatibility_level=perf.get("compatibility_level"),
                diagnostic_code=perf.get("diagnostic_code"),
                diagnostic_stage=perf.get("diagnostic_stage"),
                svs_omitted_native_data=perf.get("svs_omitted_native_data"),
                output_files=output_files,
                source_modality=source_summary["source_modality"],
                channel_definitions=[asdict(definition) for definition in definitions],
                channel_identity_source=[definition.identity_source for definition in definitions],
                channel_override_applied=bool(override),
            )
            if logger:
                logger(
                    "并行设置: "
                    f"workers={perf.get('encoder_workers', 0)}, "
                    f"raw_queue={perf.get('raw_queue_size', 0)}, "
                    f"encoded_queue={perf.get('encoded_queue_size', 0)}, "
                    f"chunk={perf.get('chunk_size', 0)}, "
                    f"backend={result.backend}"
                )
                logger(f"完成: {input_path.name} -> {output_path.name}")
                if result.native_path:
                    logger(
                        "原生路径: "
                        f"tile_mode={result.native_tile_mode or 'unknown'}, "
                        f"levels={len(result.native_level_dimensions or [])}"
                    )
                elif result.native_fallback_reason:
                    logger(f"原生资源回退: {result.native_fallback_reason}")
                if result.compatibility_level:
                    logger(
                        "兼容性: "
                        f"{result.compatibility_level}, "
                        f"container={result.source_container or 'unknown'}, "
                        f"version={result.source_version or 'unknown'}, "
                        f"codec={result.source_codec or 'unknown'}"
                    )
                if result.svs_omitted_native_data:
                    logger(f"SVS 未保存原始数据: {result.svs_omitted_native_data}")
                logger(
                    "性能: "
                    f"read/decode={result.read_decode_sec:.2f}s, "
                    f"main={result.main_write_sec:.2f}s, "
                    f"pyramid={result.pyramid_sec:.2f}s, "
                    f"thumbnail={result.thumbnail_sec:.2f}s, "
                    f"encode≈{result.encode_sec:.2f}s, "
                    f"writer_wait={result.writer_wait_sec:.2f}s, "
                    f"peak_mem={result.peak_memory_mb:.2f}MB, "
                    f"avg_cpu={result.avg_cpu_percent:.2f}%, "
                    f"total={result.duration_sec:.2f}s"
                )
            return result
    except WriteImageError as exc:
        perf = exc.perf
        if temp_output_path.exists():
            _safe_unlink(temp_output_path)
        if output_path.exists() and cancel_event is not None and cancel_event.is_set():
            _safe_unlink(output_path)
        duration = time.perf_counter() - start
        status = "cancelled" if cancel_event is not None and cancel_event.is_set() else "failed"
        error_code = "CANCELLED" if status == "cancelled" else "CONVERT_FAILED"
        if status != "cancelled" and perf.get("diagnostic_code"):
            error_code = perf["diagnostic_code"]
        if logger:
            logger(f"失败: {input_path.name}: {exc}")
        return ConvertResult(
            input_path=input_path,
            output_path=existing_output_path(),
            success=False,
            input_format=input_format,
            status=status,
            output_format=options.output_format,
            backend=perf.get("backend", options.performance_backend),
            level_dimensions=perf.get("level_dimensions"),
            duration_sec=duration,
            read_decode_sec=perf.get("read_decode_sec", 0.0),
            main_write_sec=perf.get("main_write_sec", 0.0),
            pyramid_sec=perf.get("pyramid_sec", 0.0),
            thumbnail_sec=perf.get("thumbnail_sec", 0.0),
            encode_sec=perf.get("encode_sec", 0.0),
            writer_wait_sec=perf.get("writer_wait_sec", 0.0),
            peak_memory_mb=perf.get("peak_memory_mb", 0.0),
            avg_cpu_percent=perf.get("avg_cpu_percent", 0.0),
            svs_is_bigtiff=perf.get("svs_is_bigtiff"),
            svs_label_dimensions=perf.get("svs_label_dimensions"),
            svs_macro_dimensions=perf.get("svs_macro_dimensions"),
            openslide_vendor=perf.get("openslide_vendor"),
            svs_photometric_pages=perf.get("svs_photometric_pages"),
            svs_finalize_backend=perf.get("svs_finalize_backend"),
            max_level_reached=perf.get("max_level_reached"),
            failure_stage=perf.get("failure_stage"),
            native_path=bool(perf.get("native_path", False)),
            native_level_dimensions=perf.get("native_level_dimensions"),
            native_resource_dimensions=perf.get("native_resource_dimensions"),
            native_tile_mode=perf.get("native_tile_mode"),
            native_fallback_reason=perf.get("native_fallback_reason"),
            source_container=perf.get("source_container"),
            source_version=perf.get("source_version"),
            source_codec=perf.get("source_codec"),
            source_bit_depth=perf.get("source_bit_depth"),
            source_channel_count=perf.get("source_channel_count"),
            source_axes=perf.get("source_axes"),
            compatibility_level=perf.get("compatibility_level"),
            diagnostic_code=perf.get("diagnostic_code"),
            diagnostic_stage=perf.get("diagnostic_stage"),
            svs_omitted_native_data=perf.get("svs_omitted_native_data"),
            output_files=[Path(path) for path in perf.get("output_files", [])] or None,
            source_modality=source_summary.get("source_modality"),
            channel_definitions=[asdict(definition) for definition in definitions] or None,
            channel_identity_source=[definition.identity_source for definition in definitions] or None,
            channel_override_applied=bool(getattr(slide_context, "channel_override_applied", False)),
            error_code=error_code,
            error=str(exc),
        )
    except Exception as exc:
        if temp_output_path.exists():
            _safe_unlink(temp_output_path)
        if output_path.exists() and cancel_event is not None and cancel_event.is_set():
            _safe_unlink(output_path)
        duration = time.perf_counter() - start
        if cancel_event is not None and cancel_event.is_set():
            status = "cancelled"
            error_code = "CANCELLED"
        else:
            status = "failed"
            error_code = "CONVERT_FAILED"
        if logger:
            logger(f"失败: {input_path.name}: {exc}")
        diagnostic_code = getattr(exc, "diagnostic_code", None)
        diagnostic_stage = getattr(exc, "diagnostic_stage", None)
        if diagnostic_code:
            error_code = diagnostic_code
        return ConvertResult(
            input_path=input_path,
            output_path=existing_output_path(),
            success=False,
            input_format=input_format,
            status=status,
            output_format=options.output_format,
            backend=options.performance_backend,
            duration_sec=duration,
            source_container=getattr(exc, "source_container", None) or source_summary.get("source_container"),
            source_version=getattr(exc, "source_version", None) or source_summary.get("source_version"),
            compatibility_level=(
                "static_unverified"
                if getattr(exc, "source_container", None) in {"kfba", "kfbx"}
                else source_summary.get("compatibility_level")
            ),
            diagnostic_code=diagnostic_code,
            diagnostic_stage=diagnostic_stage,
            source_modality=source_summary.get("source_modality"),
            source_codec=source_summary.get("source_codec"),
            source_bit_depth=source_summary.get("source_bit_depth"),
            source_channel_count=source_summary.get("source_channel_count"),
            source_axes=source_summary.get("source_axes"),
            channel_definitions=[asdict(definition) for definition in definitions] or None,
            channel_identity_source=[definition.identity_source for definition in definitions] or None,
            channel_override_applied=bool(definitions and any(definition.identity_source == "user_supplied" for definition in definitions)),
            error_code=error_code,
            error=str(exc),
        )


def convert_folder(
    input_dir: str | Path,
    output_dir: str | Path,
    options: ConvertOptions,
    logger=None,
    overall_callback=None,
    progress_callback=None,
    cancel_event=None,
) -> BatchResult:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    if options.selected_input_paths is None:
        files = find_convertible_files(input_dir, recursive=options.recursive, output_format=options.output_format)
    else:
        selected = {Path(path).resolve() for path in options.selected_input_paths}
        files = [
            path
            for path in find_inspectable_files(input_dir, recursive=options.recursive)
            if path.resolve() in selected
        ]

    inspections = [(path, inspect_file(path)) for path in files]
    compatible_files: list[Path] = []
    incompatible: list[tuple[int, Path, object, str]] = []
    for index, (path, inspection) in enumerate(inspections, start=1):
        expected_signature = options.input_signatures.get(str(path))
        if expected_signature is None:
            expected_signature = options.input_signatures.get(str(path.resolve()))
        signature_changed = expected_signature is not None and (
            int(expected_signature.get("size", -1)) != inspection.file_size
            or int(expected_signature.get("mtime_ns", -1)) != inspection.file_mtime_ns
        )
        if signature_changed:
            incompatible.append((index, path, inspection, "文件在预检后发生变化，请重新预检"))
        elif inspection.error:
            incompatible.append((index, path, inspection, inspection.error))
        elif options.output_format not in inspection.allowed_output_formats:
            incompatible.append(
                (
                    index,
                    path,
                    inspection,
                    inspection.incompatible_reasons.get(
                        options.output_format,
                        "输入与所选输出格式不兼容",
                    ),
                )
            )
        else:
            compatible_files.append(path)
    if incompatible and not options.convert_compatible_only:
        details = "；".join(f"{path.name}: {reason}" for _, path, _, reason in incompatible[:5])
        raise OutputCompatibilityError(
            f"有 {len(incompatible)} 个文件与 {options.output_format} 不兼容；"
            f"请选择“只转换兼容文件”或修改输出格式。{details}"
        )

    path_indexes = {path: index for index, path in enumerate(files, start=1)}
    output_plan = [
        (path_indexes[input_path], input_path, output_path)
        for _, input_path, output_path in _build_output_plan(
            compatible_files,
            input_dir,
            output_dir,
            options.output_format,
        )
    ]
    result_entries: list[tuple[int, ConvertResult]] = []
    for index, path, inspection, reason in incompatible:
        result_entries.append(
            (
                index,
                ConvertResult(
                    input_path=path,
                    output_path=None,
                    success=False,
                    input_format=inspection.input_format,
                    status="skipped_incompatible",
                    output_format=options.output_format,
                    source_modality=inspection.source_modality,
                    source_container=inspection.source_container,
                    source_version=inspection.source_version,
                    source_codec=inspection.source_codec,
                    source_bit_depth=inspection.source_bit_depth,
                    source_channel_count=inspection.channel_count,
                    channel_definitions=[asdict(item) for item in inspection.channel_definitions],
                    channel_identity_source=[item.identity_source for item in inspection.channel_definitions],
                    skipped_reason=reason,
                ),
            )
        )
    cancelled = False
    parallel_wsi = options.resolved_parallel_wsi()

    if logger:
        logger(runtime_banner())
        logger(f"发现 {len(files)} 个输入文件，其中 {len(compatible_files)} 个兼容当前输出格式")
        logger(f"WSI 并行任务数: {parallel_wsi}")

    completed_before_conversion = len(incompatible)
    if overall_callback and files:
        overall_callback(completed_before_conversion, len(files), "")

    if parallel_wsi <= 1 or len(output_plan) <= 1:
        completed = completed_before_conversion
        for index, input_path, output_path in output_plan:
            if cancel_event is not None and cancel_event.is_set():
                if logger:
                    logger("用户取消了批处理")
                cancelled = True
                break

            if overall_callback:
                overall_callback(completed, len(files), str(input_path))
            result = convert_file(
                input_path,
                output_path,
                options,
                logger=logger,
                progress_callback=progress_callback,
                cancel_event=cancel_event,
            )
            result_entries.append((index, result))
            completed += 1

            if overall_callback:
                overall_callback(completed, len(files), str(input_path))
            if not result.success and not options.continue_on_error:
                break
    else:
        pending_index = 0
        completed = completed_before_conversion
        stop_submitting = False
        cancel_logged = False
        active_parallel_wsi = min(parallel_wsi, len(output_plan))
        task_options = replace(
            options,
            memory_budget_mb=max(1, options.memory_budget_mb // active_parallel_wsi),
        )
        running: dict[Future[ConvertResult], tuple[int, Path]] = {}

        if logger:
            logger(
                f"内存预算: 总计 {options.memory_budget_mb} MB, "
                f"每个并发 WSI {task_options.memory_budget_mb} MB"
            )

        def submit_next(executor: ThreadPoolExecutor) -> None:
            nonlocal pending_index
            if pending_index >= len(output_plan):
                return
            index, input_path, output_path = output_plan[pending_index]
            pending_index += 1
            future = executor.submit(
                convert_file,
                input_path,
                output_path,
                task_options,
                logger,
                progress_callback,
                cancel_event,
            )
            running[future] = (index, input_path)

        with ThreadPoolExecutor(max_workers=parallel_wsi, thread_name_prefix="slidebridge-wsi") as executor:
            while len(running) < parallel_wsi and pending_index < len(output_plan):
                submit_next(executor)

            while running:
                if cancel_event is not None and cancel_event.is_set():
                    stop_submitting = True
                    cancelled = True
                    if logger and not cancel_logged:
                        logger("用户取消了批处理")
                    cancel_logged = True

                done, _ = wait(running.keys(), return_when=FIRST_COMPLETED)
                for future in done:
                    index, input_path = running.pop(future)
                    result = future.result()
                    result_entries.append((index, result))
                    completed += 1
                    if overall_callback:
                        overall_callback(completed, len(files), str(input_path))
                    if not result.success and not options.continue_on_error:
                        stop_submitting = True

                while (
                    not stop_submitting
                    and len(running) < parallel_wsi
                    and pending_index < len(output_plan)
                    and not (cancel_event is not None and cancel_event.is_set())
                ):
                    submit_next(executor)

        if cancelled and logger:
            logger("批处理取消等待已运行任务结束")

    results = [result for _, result in sorted(result_entries, key=lambda item: item[0])]
    batch = BatchResult(
        total_files=len(files),
        success_count=sum(1 for result in results if result.success),
        failed_count=sum(1 for result in results if result.status == "failed"),
        skipped_count=sum(1 for result in results if result.status == "skipped_incompatible"),
        cancelled_count=sum(1 for result in results if result.status == "cancelled"),
        cancelled=cancelled or any(result.status == "cancelled" for result in results),
        results=results,
    )
    batch.report_path = output_dir / "conversion_report.csv"
    write_report(results, batch.report_path)
    if logger:
        message = (
            f"批处理结束: 成功 {batch.success_count} 个, 失败 {batch.failed_count} 个, "
            f"跳过 {batch.skipped_count} 个, 取消 {batch.cancelled_count} 个, 总计 {batch.total_files} 个"
        )
        logger(message)
        logger(f"报告已写入: {batch.report_path}")
    return batch
