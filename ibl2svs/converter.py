from __future__ import annotations

import csv
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import replace
from pathlib import Path

from .app_meta import runtime_banner
from .kfb_source import KfbSlideSource
from .models import BatchResult, ConvertOptions, ConvertResult
from .punuoxi_source import PunuoxiImageSource
from .reader import IBLSlide
from .tiff_source import TiffSlideSource
from .writer import WriteImageError, write_image


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


def detect_input_format(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".ibl":
        return "ibl"
    if suffix == ".svs":
        return "svs"
    if suffix in {".tif", ".tiff"}:
        return "generic_tiff"
    if suffix == ".kfb":
        return "kfb"
    if suffix == ".image":
        return "image"
    return "unsupported"


def find_convertible_files(
    input_dir: str | Path,
    recursive: bool = True,
    output_format: str = "generic_tiff",
) -> list[Path]:
    input_dir = Path(input_dir)
    pattern = "**/*" if recursive else "*"
    if output_format == "svs":
        suffixes = {".ibl", ".tif", ".tiff", ".kfb", ".image"}
    else:
        suffixes = {".ibl", ".svs", ".kfb", ".image"}
    files = [
        path
        for path in input_dir.glob(pattern)
        if path.is_file() and path.suffix.lower() in suffixes
    ]
    return sorted(files)


def build_output_path(
    input_path: Path,
    input_root: Path,
    output_root: Path,
    output_format: str = "generic_tiff",
) -> Path:
    relative = input_path.relative_to(input_root)
    candidate = output_root / relative
    suffix = ".svs" if output_format == "svs" else ".tif"
    candidate = candidate.with_suffix(suffix)
    candidate.parent.mkdir(parents=True, exist_ok=True)
    if not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix
    index = 1
    while True:
        alt = candidate.with_name(f"{stem}_{index}{suffix}")
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
    suffix = ".svs" if output_format == "svs" else ".tif"
    for index, input_path in enumerate(files, start=1):
        relative = input_path.relative_to(input_root)
        candidate = (output_root / relative).with_suffix(suffix)
        candidate.parent.mkdir(parents=True, exist_ok=True)
        if candidate.exists() or candidate in reserved:
            stem = candidate.stem
            suffix_value = candidate.suffix
            alt_index = 1
            while True:
                alt = candidate.with_name(f"{stem}_{alt_index}{suffix_value}")
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
                    result.failure_stage or "",
                    result.error_code or "",
                    result.error or "",
                ]
            )


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
    target_suffix = ".tif" if options.output_format == "generic_tiff" else ".svs"
    output_path = output_path.with_suffix(target_suffix)
    temp_output_path = output_path.with_suffix(f"{output_path.suffix}.part")
    start = time.perf_counter()
    error_code = None

    def existing_output_path() -> Path | None:
        return output_path if output_path.exists() else None

    try:
        if input_format == "unsupported":
            raise RuntimeError(f"不支持的输入格式: {input_path.suffix or input_path.name}")

        if temp_output_path.exists():
            _safe_unlink(temp_output_path)

        if input_format == "ibl":
            slide_context = IBLSlide(input_path, cache_size=options.cache_blocks_per_row)
        elif input_format == "kfb":
            slide_context = KfbSlideSource(input_path, cache_size=options.cache_blocks_per_row or 256)
        elif input_format == "image":
            slide_context = PunuoxiImageSource(input_path, cache_size=options.cache_blocks_per_row or 256)
        else:
            slide_context = TiffSlideSource(input_path)

        with slide_context as slide:
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

            pyramid_levels, perf = write_image(
                slide,
                temp_output_path,
                options,
                progress_callback=writer_progress,
                cancel_event=cancel_event,
            )
            temp_output_path.replace(output_path)
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
        return ConvertResult(
            input_path=input_path,
            output_path=existing_output_path(),
            success=False,
            input_format=input_format,
            status=status,
            output_format=options.output_format,
            backend=options.performance_backend,
            duration_sec=duration,
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
    files = find_convertible_files(input_dir, recursive=options.recursive, output_format=options.output_format)
    output_plan = _build_output_plan(files, input_dir, output_dir, options.output_format)
    result_entries: list[tuple[int, ConvertResult]] = []
    cancelled = False
    parallel_wsi = options.resolved_parallel_wsi()

    if logger:
        logger(runtime_banner())
        logger(f"发现 {len(files)} 个可转换文件")
        logger(f"WSI 并行任务数: {parallel_wsi}")

    if overall_callback and files:
        overall_callback(0, len(files), str(files[0]))

    if parallel_wsi <= 1 or len(output_plan) <= 1:
        for index, input_path, output_path in output_plan:
            if cancel_event is not None and cancel_event.is_set():
                if logger:
                    logger("用户取消了批处理")
                cancelled = True
                break

            if overall_callback:
                overall_callback(index - 1, len(files), str(input_path))
            result = convert_file(
                input_path,
                output_path,
                options,
                logger=logger,
                progress_callback=progress_callback,
                cancel_event=cancel_event,
            )
            result_entries.append((index, result))

            if overall_callback:
                overall_callback(index, len(files), str(input_path))
            if not result.success and not options.continue_on_error:
                break
    else:
        pending_index = 0
        completed = 0
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
        cancelled_count=sum(1 for result in results if result.status == "cancelled"),
        cancelled=cancelled or any(result.status == "cancelled" for result in results),
        results=results,
    )
    batch.report_path = output_dir / "conversion_report.csv"
    write_report(results, batch.report_path)
    if logger:
        message = (
            f"批处理结束: 成功 {batch.success_count} 个, 失败 {batch.failed_count} 个, "
            f"取消 {batch.cancelled_count} 个, 总计 {batch.total_files} 个"
        )
        logger(message)
        logger(f"报告已写入: {batch.report_path}")
    return batch
