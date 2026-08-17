from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Literal


OutputFormat = Literal["ome_tiff", "svs", "fluorescence_svs", "afi"]
SourceModality = Literal["brightfield", "fluorescence", "unknown"]
ChannelIdentitySource = Literal[
    "source_metadata",
    "documented_vendor_id",
    "user_supplied",
    "unknown",
]


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


@dataclass(frozen=True)
class ChannelDefinition:
    index: int
    name: str
    fluor: str | None = None
    color: tuple[int, int, int] = (255, 255, 255)
    excitation_nm: float | None = None
    emission_nm: float | None = None
    exposure: float | None = None
    identity_source: ChannelIdentitySource = "unknown"


@dataclass(frozen=True)
class InputInspection:
    input_path: Path
    file_size: int
    file_mtime_ns: int
    input_format: str
    source_modality: SourceModality
    source_container: str | None
    source_version: str | None
    source_codec: str | None
    source_bit_depth: int
    field_count: int
    channel_count: int
    z_count: int
    t_count: int
    channel_definitions: tuple[ChannelDefinition, ...]
    allowed_output_formats: tuple[OutputFormat, ...]
    incompatible_reasons: dict[str, str]
    error: str | None = None


@dataclass(frozen=True)
class BatchInspection:
    input_dir: Path
    recursive: bool
    files: tuple[InputInspection, ...]

    @property
    def total_files(self) -> int:
        return len(self.files)


@dataclass
class ConvertOptions:
    recursive: bool = True
    output_format: OutputFormat = "ome_tiff"
    performance_backend: Literal["tifffile"] = "tifffile"
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
    selected_input_paths: tuple[str, ...] | None = None
    convert_compatible_only: bool = False
    channel_overrides: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    input_signatures: dict[str, dict[str, int | str]] = field(default_factory=dict)

    def resolved_tile_size(self) -> int:
        if self.output_format == "svs" and self.tile_size == 256:
            return 240
        if self.output_format in {"fluorescence_svs", "afi"}:
            return 256
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
        return max(1, min(8, int(self.parallel_wsi)))


@dataclass
class ConvertResult:
    input_path: Path
    output_path: Path | None
    success: bool
    input_format: str = "ibl"
    status: str = "success"
    output_format: str = "ome_tiff"
    backend: str = "tifffile"
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
    native_path: bool = False
    native_level_dimensions: list[tuple[int, int]] | None = None
    native_resource_dimensions: dict[str, tuple[int, int] | None] | None = None
    native_tile_mode: str | None = None
    native_fallback_reason: str | None = None
    source_container: str | None = None
    source_version: str | None = None
    source_codec: str | None = None
    source_bit_depth: int | None = None
    source_channel_count: int | None = None
    source_axes: str | None = None
    compatibility_level: str | None = None
    diagnostic_code: str | None = None
    diagnostic_stage: str | None = None
    svs_omitted_native_data: str | None = None
    output_files: list[Path] | None = None
    source_modality: str | None = None
    channel_definitions: list[dict[str, Any]] | None = None
    channel_identity_source: list[str] | None = None
    channel_override_applied: bool = False
    skipped_reason: str | None = None


@dataclass
class BatchResult:
    total_files: int
    success_count: int
    failed_count: int
    skipped_count: int = 0
    cancelled_count: int = 0
    cancelled: bool = False
    report_path: Path | None = None
    results: list[ConvertResult] = field(default_factory=list)
