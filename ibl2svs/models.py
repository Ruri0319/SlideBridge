from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class BaseInfo:
    magic_no: str
    version: str
    focus_num: int
    image_format: int
    layer_size: int
    img_color: int
    check_sum: int
    ratio_step: int
    max_layer_size: int
    slide_type: int
    background_color: int
    pixel_size_mm: float
    total_img_num: int
    max_zoom_rate: int
    img_col: int
    img_row: int
    img_width: int
    img_height: int
    tile_width: int
    tile_height: int
    shrink_tile_num: int
    total_img_width: int
    total_img_height: int

    @property
    def mpp(self) -> float:
        return self.pixel_size_mm * 1000.0


@dataclass(frozen=True)
class BlockRecord:
    block_id: int
    grid_col: int
    grid_row: int
    x: int
    y: int
    width: int
    height: int
    x1: int
    y1: int


@dataclass
class ConvertOptions:
    recursive: bool = True
    output_format: Literal["generic_tiff", "svs"] = "generic_tiff"
    performance_backend: Literal["pyvips"] = "pyvips"
    svs_use_bigtiff: bool | Literal["auto"] = "auto"
    svs_generate_label: bool = True
    svs_generate_macro: bool = True
    svs_finalize_with_libtiff: bool = True
    svs_validate_with_tiffinfo: bool = True
    chunk_size: int | None = None
    tile_size: int = 256
    jpeg_quality: int = 90
    memory_budget_mb: int = 6144
    gui_log_limit: int = 2000
    generate_dense_pyramid: bool = True
    continue_on_error: bool = True
    cache_blocks_per_row: int | None = None
    encoder_workers: int | None = None
    raw_queue_size: int | None = None
    encoded_queue_size: int | None = None
    parallel_wsi: int = 1

    def resolved_tile_size(self) -> int:
        if self.output_format == "svs" and self.tile_size == 256:
            return 240
        return self.tile_size

    def resolved_jpeg_quality(self) -> int:
        return self.jpeg_quality

    def resolved_encoder_workers(self) -> int:
        if self.encoder_workers is not None:
            return max(1, self.encoder_workers)
        workers = max(1, min(4, os.cpu_count() or 4))
        if self.memory_budget_mb <= 4096:
            workers = 1
        return workers

    def resolved_chunk_size(self) -> int:
        tile_size = self.resolved_tile_size()
        if self.chunk_size is not None:
            return max(tile_size, self.chunk_size)
        workers = self.resolved_encoder_workers()
        if self.output_format == "svs":
            if self.memory_budget_mb <= 4096:
                return max(tile_size, 512)
            if self.memory_budget_mb <= 6144:
                return max(tile_size, 1024)
            return max(tile_size, 1536)
        if workers <= 2:
            return 2048
        return 1024

    def resolved_raw_queue_size(self) -> int:
        if self.raw_queue_size is not None:
            return max(1, self.raw_queue_size)
        if self.output_format == "svs":
            return max(2, self.resolved_encoder_workers())
        return max(2, self.resolved_encoder_workers())

    def resolved_encoded_queue_size(self) -> int:
        if self.encoded_queue_size is not None:
            return max(1, self.encoded_queue_size)
        if self.output_format == "svs":
            return max(2, self.resolved_encoder_workers())
        return max(2, self.resolved_encoder_workers())

    def resolved_svs_use_bigtiff(self) -> bool:
        if self.output_format != "svs":
            return False
        if self.svs_use_bigtiff == "auto":
            return False
        return bool(self.svs_use_bigtiff)

    def resolved_parallel_wsi(self) -> int:
        return max(1, min(4, int(self.parallel_wsi)))


@dataclass
class ConvertResult:
    input_path: Path
    output_path: Path | None
    success: bool
    input_format: str = "ibl"
    status: str = "success"
    output_format: str = "generic_tiff"
    backend: str = "pyvips"
    width: int | None = None
    height: int | None = None
    level_dimensions: list[tuple[int, int]] | None = None
    pyramid_levels: int | None = None
    mpp: float | None = None
    duration_sec: float = 0.0
    read_decode_sec: float = 0.0
    main_write_sec: float = 0.0
    pyramid_sec: float = 0.0
    thumbnail_sec: float = 0.0
    encode_sec: float = 0.0
    writer_wait_sec: float = 0.0
    peak_memory_mb: float = 0.0
    avg_cpu_percent: float = 0.0
    svs_is_bigtiff: bool | None = None
    svs_label_dimensions: tuple[int, int] | None = None
    svs_macro_dimensions: tuple[int, int] | None = None
    openslide_vendor: str | None = None
    svs_photometric_pages: list[str] | None = None
    svs_finalize_backend: str | None = None
    max_level_reached: int | None = None
    failure_stage: str | None = None
    error_code: str | None = None
    error: str | None = None


@dataclass
class BatchResult:
    total_files: int
    success_count: int
    failed_count: int
    cancelled_count: int = 0
    cancelled: bool = False
    report_path: Path | None = None
    results: list[ConvertResult] = field(default_factory=list)
