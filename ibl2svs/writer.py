from __future__ import annotations

import gc
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np
from PIL import Image, ImageCms, ImageDraw

from .app_meta import APP_NAME, APP_VERSION
from .assembler import (
    DensePyramidDrive,
    PILImageSource,
    ProgressState,
    StripDownsampleDrive,
    compute_pyramid_shapes,
    iter_source_tiles,
    tile_count,
)
from .models import ConvertOptions
from .reader import IBLSlide
from .system_metrics import PerfTracker


class WriteImageError(RuntimeError):
    def __init__(self, message: str, perf: dict[str, Any]):
        super().__init__(message)
        self.perf = perf


def _load_tiff_runtime(*, require_imagecodecs: bool = True):
    try:
        import tifffile
    except ImportError as exc:
        raise RuntimeError(
            "缺少 tifffile 依赖，无法写出 TIFF/SVS。请先在 Windows 环境安装 requirements.txt。"
        ) from exc
    if require_imagecodecs:
        try:
            import imagecodecs  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "缺少 imagecodecs 依赖，无法写出 JPEG 压缩的 TIFF/SVS。请先在 Windows 环境安装 requirements.txt。"
            ) from exc
    return tifffile


def build_generic_description(slide: IBLSlide) -> str:
    app_mag = slide.base_info.max_zoom_rate
    return (
        f"{APP_NAME} generic pyramidal TIFF | "
        f"OriginalFile={slide.path.name} | "
        f"Width={slide.width} | Height={slide.height} | "
        f"MPP = {slide.base_info.mpp:.6f} | "
        f"AppMag = {app_mag} | "
        f"ObjectivePower = {app_mag} | "
        f"openslide.objective-power = {app_mag}"
    )


def build_aperio_description(slide: IBLSlide, options: ConvertOptions) -> str:
    tile_size = options.resolved_tile_size()
    return (
        "Aperio Image Library v12.0.0 \r\n"
        f"{slide.width}x{slide.height} [0,0 {slide.width}x{slide.height}] "
        f"({tile_size}x{tile_size}) JPEG/RGB Q={options.jpeg_quality}|"
        f"AppMag = {slide.base_info.max_zoom_rate}|"
        f"MPP = {slide.base_info.mpp:.6f}|"
        f"Filename = {slide.path.stem}|"
        f"OriginalWidth = {slide.width}|"
        f"OriginalHeight = {slide.height}"
    )


def build_aperio_pyramid_description(
    slide: IBLSlide,
    options: ConvertOptions,
    level_width: int,
    level_height: int,
) -> str:
    tile_size = options.resolved_tile_size()
    return (
        "Aperio Image Library v12.0.0 \r\n"
        f"{slide.width}x{slide.height} [0,0 {slide.width}x{slide.height}] "
        f"({tile_size}x{tile_size}) -> {level_width}x{level_height} JPEG/RGB Q={options.jpeg_quality}"
    )


def build_aperio_thumbnail_description(slide: IBLSlide, width: int, height: int) -> str:
    return (
        "Aperio Image Library v12.0.0 \n"
        f"{slide.width}x{slide.height} -> {width}x{height} - |"
        f"AppMag = {slide.base_info.max_zoom_rate}|"
        f"MPP = {slide.base_info.mpp:.6f}|"
        f"Filename = {slide.path.stem}|"
        f"OriginalWidth = {slide.width}|"
        f"OriginalHeight = {slide.height}"
    )


def build_aperio_label_description(width: int, height: int) -> str:
    return f"Aperio Image Library v12.0.0 \nlabel {width}x{height}"


def build_aperio_macro_description(width: int, height: int) -> str:
    return f"Aperio Image Library v12.0.0 \nmacro {width}x{height}"


def _resolution_kwargs_for_mpp(
    mpp: float,
    *,
    base_width: int,
    base_height: int,
    level_width: int,
    level_height: int,
) -> dict[str, Any]:
    if mpp <= 0:
        return {}
    x_downsample = base_width / max(1, level_width)
    y_downsample = base_height / max(1, level_height)
    x_pixels_per_cm = 10000.0 / (mpp * x_downsample)
    y_pixels_per_cm = 10000.0 / (mpp * y_downsample)
    return {
        "resolution": (x_pixels_per_cm, y_pixels_per_cm),
        "resolutionunit": "CENTIMETER",
    }


def _svs_use_bigtiff(slide: IBLSlide, options: ConvertOptions) -> bool:
    # Prefer classic TIFF for SVS whenever possible. Real Aperio samples are
    # commonly classic TIFF, and Bio-Formats/OpenSlide interoperability is
    # better when we stay closer to that layout.
    #
    # Switch to BigTIFF automatically when the estimated output size could
    # exceed Classic TIFF's 4 GB file-offset limit.  Estimate based on total
    # pixels × 0.5 bytes/pixel (conservative JPEG Q=90) across all pyramid
    # levels; use 3.2 GB as the safety threshold.
    total_pixels = slide.width * slide.height
    downsample = 2
    while True:
        downsample *= 2
        w = max(1, slide.width // downsample)
        h = max(1, slide.height // downsample)
        if min(w, h) < 512:
            break
        total_pixels += w * h
    estimated_bytes = total_pixels * 0.5
    return estimated_bytes > 3_200_000_000


def _get_srgb_icc_profile() -> bytes:
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def _find_sos_marker(data: bytes) -> int:
    """Return byte offset of the SOS (0xFF 0xDA) marker in JPEG *data*.

    Raises ``ValueError`` if no SOS marker is found.
    """
    i = 2  # skip SOI
    end = len(data) - 1
    while i < end:
        if data[i] == 0xFF:
            b = data[i + 1]
            if b == 0xDA:  # SOS – Start of Scan
                return i
            if b in (0xD0, 0xD1, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0x00, 0x01):
                i += 2
                continue
            if i + 3 < len(data):
                seg_len = (data[i + 2] << 8) | data[i + 3]
                i += 2 + seg_len
            else:
                i += 2
        else:
            i += 1
    raise ValueError("No SOS marker found in JPEG data")


def _extract_jpeg_tables(data: bytes) -> bytes:
    """Return ``SOI + DQT* + DHT* + EOI`` from a complete JPEG *data*.

    Only DQT (0xDB) and DHT (0xC4) segments are included; SOF0, SOS, APP
    and other markers are intentionally excluded so that the result forms
    a pure *abbreviated table-specification* stream suitable for the
    ``JPEGTables`` TIFF tag (347).
    """
    sos_pos = _find_sos_marker(data)
    segments: list[bytes] = []
    i = 2  # skip SOI
    while i < sos_pos:
        b = data[i + 1]
        if b in (0xD0, 0xD1, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0x00, 0x01):
            i += 2
            continue
        seg_len = (data[i + 2] << 8) | data[i + 3]
        if b in (0xDB, 0xC4):  # DQT or DHT
            segments.append(data[i : i + 2 + seg_len])
        i += 2 + seg_len
    return b'\xff\xd8' + b''.join(segments) + b'\xff\xd9'


def _find_sof0_marker(data: bytes) -> int:
    """Return byte offset of the SOF0 (0xFF 0xC0) marker in JPEG *data*.

    Raises ``ValueError`` if no SOF0 marker is found.
    """
    i = 2  # skip SOI
    end = len(data) - 1
    while i < end:
        if data[i] == 0xFF:
            b = data[i + 1]
            if b == 0xC0:  # SOF0 – Start of Frame (baseline DCT)
                return i
            if b in (0xD0, 0xD1, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0x00, 0x01):
                i += 2
                continue
            if i + 3 < len(data):
                seg_len = (data[i + 2] << 8) | data[i + 3]
                i += 2 + seg_len
            else:
                i += 2
        else:
            i += 1
    raise ValueError("No SOF0 marker found in JPEG data")


def _encode_jpeg(data: np.ndarray, quality: int) -> bytes:
    """Encode an ndarray as an **RGB** self-contained JPEG byte-string.

    Uses Adobe-style component IDs (R=82, G=71, B=66) so that decoders
    recognise the data as RGB natively — no YCbCr round-trip.  This
    matches the Aperio SVS ``JPEG/RGB`` convention found in ground-truth
    CPTAC files.
    """
    from imagecodecs import jpeg8_encode

    return jpeg8_encode(data, level=quality, colorspace="rgb", outcolorspace="rgb", optimize=False)


def _encode_jpeg_stripped(data: np.ndarray, quality: int) -> tuple[bytes, bytes]:
    """Encode as RGB JPEG and split into ``(tables, stripped_tile)``.

    *tables* is suitable for the JPEGTables TIFF tag (347):
    ``SOI + DQT* + DHT* + EOI`` — every table segment between SOI and SOS
    except SOF0.

    *stripped_tile* is ``SOI + SOF0 + SOS + data + EOI`` — the absolute
    minimum abbreviated JPEG.  It contains **no** DQT or DHT markers;
    every tile on the page shares the single JPEGTables copy.

    Together, OpenSlide can load *tables* once via libjpeg and share them
    across every tile on the page, dramatically reducing memory.
    """
    raw = _encode_jpeg(data, quality)
    sof0_pos = _find_sof0_marker(raw)
    sos_pos = _find_sos_marker(raw)

    # Extract the SOF0 segment (frame header) between DQT/DHT and SOS
    sof0_len = (raw[sof0_pos + 2] << 8) | raw[sof0_pos + 3]
    sof0_segment = raw[sof0_pos : sof0_pos + 2 + sof0_len]

    # Tile = SOI + SOF0 + SOS + entropy-coded data  (no DQT, no DHT)
    stripped = b'\xff\xd8' + sof0_segment + raw[sos_pos:]
    tables = _extract_jpeg_tables(raw)
    return tables, stripped


def _get_jpeg_tables_for_shape(h: int, w: int, quality: int) -> bytes:
    """Return the shared JPEG header tables for a tile of shape *(h, w)*."""
    sample = np.zeros((h, w, 3), dtype=np.uint8)
    raw = _encode_jpeg(sample, quality)
    return _extract_jpeg_tables(raw)


class _EncodedTileIter:
    """Wrap a tile-ndarray iterator so it yields **stripped** JPEG bytes.

    Each yielded tile is ``SOI + SOF0 + SOS + data + EOI`` — an
    abbreviated JPEG with **no** DQT or DHT markers.  The shared tables
    are written separately via the ``JPEGTables`` TIFF tag (347).

    OpenSlide / libjpeg loads the tables once and reuses them for every
    tile, keeping per-tile decoder allocations to a minimum.
    """

    def __init__(self, source, quality: int):
        self._source = source
        self._quality = quality

    def __iter__(self):
        for tile in self._source:
            _, stripped = _encode_jpeg_stripped(tile, self._quality)
            yield stripped


def build_aperio_svs_layout(slide: IBLSlide, options: ConvertOptions) -> dict[str, Any]:
    pyramid_levels = _compute_svs_pyramid_shapes(slide, options)
    return {
        "main": (slide.width, slide.height),
        "thumbnail_max": 1024,
        "pyramid": pyramid_levels,
        "include_label": options.svs_generate_label,
        "include_macro": options.svs_generate_macro,
        "bigtiff": options.resolved_svs_use_bigtiff() if options.svs_use_bigtiff != "auto" else _svs_use_bigtiff(slide, options),
    }


def _resolve_parallel_settings(options: ConvertOptions) -> dict[str, int]:
    tile_size = options.resolved_tile_size()
    workers = options.resolved_encoder_workers()
    raw_queue_size = options.resolved_raw_queue_size()
    encoded_queue_size = options.resolved_encoded_queue_size()
    chunk_size = options.resolved_chunk_size()
    return {
        "encoder_workers": workers,
        "raw_queue_size": raw_queue_size,
        "encoded_queue_size": encoded_queue_size,
        "chunk_size": chunk_size,
        "buffersize": encoded_queue_size * tile_size * tile_size * 3,
    }


def _emit_progress(progress_callback, level_name: str, done: int, total: int, overall_done: int, overall_total: int) -> None:
    if progress_callback is None:
        return
    progress_callback(level_name, done, total, overall_done, overall_total)


GENERIC_TIFF_WRITE_UNITS = 1


def _svs_write_tail_units(options: ConvertOptions) -> int:
    return 1 + int(bool(options.svs_finalize_with_libtiff)) + int(bool(options.svs_validate_with_tiffinfo))


def _svs_associated_units(options: ConvertOptions) -> int:
    return int(bool(options.svs_generate_label)) + int(bool(options.svs_generate_macro))


def _advance_write_tail_progress(
    perf: dict[str, float],
    progress_callback,
    *,
    step_done: int,
    step_total: int,
    overall_done: int,
    overall_total: int,
) -> None:
    perf["current_stage"] = "写出文件"
    _emit_progress(
        progress_callback,
        "写出文件",
        step_done,
        step_total,
        overall_done,
        overall_total,
    )


def _make_temp_path(directory: Path | None, suffix: str) -> Path:
    fd, raw_path = tempfile.mkstemp(prefix="ibl2svs_flat_", suffix=suffix, dir=directory)
    os.close(fd)
    return Path(raw_path)


def _safe_unlink(path: Path, retries: int = 5, delay: float = 0.1, *, strict: bool = True) -> bool:
    for attempt in range(retries):
        try:
            if path.exists():
                path.unlink()
            return True
        except PermissionError:
            if attempt == retries - 1:
                if strict:
                    raise
                return False
            time.sleep(delay)
        except FileNotFoundError:
            return True
    return not path.exists()


def _cleanup_vips_temp_file(path: Path) -> bool:
    # libvips can keep file handles open slightly longer on Windows; use a
    # best-effort cleanup path outside the user output directory.
    return _safe_unlink(path, retries=12, delay=0.25, strict=False)


def finalize_svs_with_libtiff(path: Path) -> str:
    cleaned = False
    tifffile = _load_tiff_runtime(require_imagecodecs=False)
    with tifffile.TiffFile(str(path)) as tif:
        page_count = len(tif.pages)
    for index in range(page_count):
        for tag in ("282", "283", "296", "305"):
            try:
                proc = subprocess.run(
                    ["tiffset", "-d", str(index), "-u", tag, str(path)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
            except FileNotFoundError:
                return "unavailable"
            cleaned = cleaned or proc.returncode == 0
    return "libtiff-tiffset" if cleaned else "none"


def inspect_svs_compatibility(path: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "openslide_vendor": None,
        "svs_photometric_pages": None,
    }
    try:
        tifffile = _load_tiff_runtime(require_imagecodecs=False)
        with tifffile.TiffFile(str(path)) as tif:
            report["svs_photometric_pages"] = [
                str(page.photometric.name if hasattr(page.photometric, "name") else page.photometric)
                for page in tif.pages
            ]
    except Exception:
        pass

    try:
        import openslide
    except ImportError:
        return report

    try:
        slide = openslide.OpenSlide(str(path))
        try:
            report["openslide_vendor"] = slide.properties.get(openslide.PROPERTY_NAME_VENDOR)
        finally:
            slide.close()
    except Exception:
        pass
    return report


def _compute_svs_pyramid_shapes(slide: IBLSlide, options: ConvertOptions) -> list[tuple[int, int]]:
    shapes: list[tuple[int, int]] = []
    downsample = 2
    while True:
        downsample *= 2  # 4, 8, 16, 32, 64, …
        width = max(1, slide.width // downsample)
        height = max(1, slide.height // downsample)
        if min(width, height) < 512:
            break
        shapes.append((width, height))
    if not shapes and min(slide.width, slide.height) >= 128:
        shapes.append((max(1, slide.width // 4), max(1, slide.height // 4)))
    return shapes


def _build_thumbnail_array(slide: IBLSlide, max_size: int = 1024) -> np.ndarray:
    image = slide.get_thumbnail_image() or slide.get_preview_image()
    if image is None:
        sample_w = min(max_size, slide.width)
        sample_h = min(max_size, slide.height)
        image = Image.fromarray(slide.read_region(0, 0, sample_w, sample_h, decode_workers=1))

    width, height = image.size
    scale = min(max_size / max(1, width), max_size / max(1, height), 1.0)
    thumb = image.resize(
        (max(1, int(width * scale)), max(1, int(height * scale))),
        resample=Image.Resampling.BILINEAR,
    )
    return np.array(thumb, dtype=np.uint8)


def _build_macro_array(slide: IBLSlide, max_size: int = 1600) -> np.ndarray:
    macro_getter = getattr(slide, "get_macro_image", None)
    image = macro_getter() if callable(macro_getter) else None
    # Prefer the native label/overview image stored in the IBL file.
    if image is None:
        image = slide.get_label_image()
    if image is None:
        image = slide.get_preview_image() or slide.get_thumbnail_image()
    if image is None:
        image = Image.fromarray(slide.assemble_preview_from_layer1(scale=4))
    width, height = image.size
    scale = min(max_size / max(1, width), max_size / max(1, height), 1.0)
    macro = image.resize(
        (max(1, int(width * scale)), max(1, int(height * scale))),
        resample=Image.Resampling.BILINEAR,
    )
    return np.asarray(macro.convert("RGB"), dtype=np.uint8)


def _build_label_array(slide: IBLSlide) -> np.ndarray:
    target_w = 666
    target_h = 716
    # Use the native label image as background when available.
    native = slide.get_label_image()
    if native is not None:
        native.thumbnail((target_w, target_h), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (target_w, target_h), (30, 30, 30))
        nx = (target_w - native.width) // 2
        ny = (target_h - native.height) // 2
        canvas.paste(native, (nx, ny))
    else:
        canvas = Image.new("RGB", (target_w, target_h), (255, 255, 255))

    draw = ImageDraw.Draw(canvas)
    # Overlay key metadata in a semi-transparent footer bar.
    footer_y = target_h - 52
    draw.rectangle([(0, footer_y), (target_w, target_h)], fill=(0, 0, 0, 180))
    meta = slide.get_scan_metadata()
    device = meta.get("deviceNo", "")
    scan_time = meta.get("scanTime", "")
    if isinstance(scan_time, (int, float)) and int(scan_time) > 0:
        from datetime import datetime, timezone
        scan_time = datetime.fromtimestamp(int(scan_time), tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    else:
        scan_time = ""
    lines = [slide.path.stem, f"MPP {slide.base_info.mpp:.4f}  AppMag {slide.base_info.max_zoom_rate}"]
    if device:
        lines.append(f"Device: {device}")
    if scan_time:
        lines.append(f"Scanned: {scan_time}")
    for idx, line in enumerate(lines):
        draw.text((10, footer_y + 4 + idx * 12), line, fill=(220, 220, 220))
    return np.asarray(canvas, dtype=np.uint8)


def write_aperio_associated_images(
    tif,
    slide: IBLSlide,
    options: ConvertOptions,
    perf: dict[str, float],
    *,
    overall_done: int,
    overall_total: int,
    progress_callback=None,
) -> dict[str, tuple[int, int] | None]:
    metadata: dict[str, tuple[int, int] | None] = {
        "svs_label_dimensions": None,
        "svs_macro_dimensions": None,
    }
    iccprofile = _get_srgb_icc_profile()
    associated_total = _svs_associated_units(options)
    associated_done = 0

    macro_array = _build_macro_array(slide) if options.svs_generate_macro else None

    if options.svs_generate_label:
        perf["current_stage"] = "生成附属图像"
        label_started = time.perf_counter()
        label_array = _build_label_array(slide)
        tif.write(
            label_array,
            photometric="rgb",
            compression="lzw",
            rowsperstrip=4,
            description=build_aperio_label_description(label_array.shape[1], label_array.shape[0]),
            metadata=None,
            software="",
            iccprofile=iccprofile,
            subfiletype=1,
        )
        perf["thumbnail_sec"] += time.perf_counter() - label_started
        metadata["svs_label_dimensions"] = (label_array.shape[1], label_array.shape[0])
        associated_done += 1
        _emit_progress(
            progress_callback,
            "生成附属图像",
            associated_done,
            max(1, associated_total),
            overall_done + associated_done,
            overall_total,
        )

    if macro_array is not None:
        perf["current_stage"] = "生成附属图像"
        macro_started = time.perf_counter()
        mh, mw = macro_array.shape[:2]
        tables, stripped = _encode_jpeg_stripped(macro_array, options.jpeg_quality)
        tif.write(
            data=iter([stripped]),
            shape=(mh, mw, 3),
            dtype=np.uint8,
            photometric="rgb",
            compression="jpeg",
            compressionargs={"outcolorspace": "rgb"},
            subsampling=(1, 1),
            jpegtables=tables,
            rowsperstrip=mh,
            description=build_aperio_macro_description(mw, mh),
            metadata=None,
            software="",
            iccprofile=iccprofile,
            subfiletype=9,
        )
        perf["thumbnail_sec"] += time.perf_counter() - macro_started
        metadata["svs_macro_dimensions"] = (mw, mh)
        associated_done += 1
        _emit_progress(
            progress_callback,
            "生成附属图像",
            associated_done,
            max(1, associated_total),
            overall_done + associated_done,
            overall_total,
        )

    return metadata


def _svs_common_kwargs(options: ConvertOptions) -> dict[str, Any]:
    tile_size = options.resolved_tile_size()
    return {
        "photometric": "rgb",
        "tile": (tile_size, tile_size),
        "compression": "jpeg",
        "compressionargs": {"outcolorspace": "rgb"},
        "subsampling": (1, 1),
        "metadata": None,
        "dtype": np.uint8,
    }


def write_aperio_thumbnail_page(
    tif,
    slide: IBLSlide,
    options: ConvertOptions,
    perf: dict[str, float],
    *,
    iccprofile: bytes,
    overall_done: int,
    overall_total: int,
    thumbnail_max: int,
    progress_callback=None,
) -> int:
    perf["current_stage"] = "生成缩略图"
    thumb_started = time.perf_counter()
    thumbnail = _build_thumbnail_array(slide, max_size=thumbnail_max)
    h, w = thumbnail.shape[:2]
    tables, stripped = _encode_jpeg_stripped(thumbnail, options.jpeg_quality)
    tif.write(
        data=iter([stripped]),
        shape=(h, w, 3),
        dtype=np.uint8,
        photometric="rgb",
        compression="jpeg",
        compressionargs={"outcolorspace": "rgb"},
        subsampling=(1, 1),
        jpegtables=tables,
        rowsperstrip=h,
        description=build_aperio_thumbnail_description(slide, w, h),
        software="",
        iccprofile=iccprofile,
        metadata=None,
    )
    perf["thumbnail_sec"] += time.perf_counter() - thumb_started
    _emit_progress(progress_callback, "生成缩略图", 1, 1, overall_done + 1, overall_total)
    return 1


def _write_generic_tiff_streaming(
    slide: IBLSlide,
    output_path: Path,
    options: ConvertOptions,
    perf: dict[str, Any],
    *,
    progress_callback=None,
    cancel_event=None,
) -> int:
    """Single-pass streaming Generic TIFF writer — no temp files, no pyvips.

    Reads IBL data exactly once via *DensePyramidDrive*.  The main page is
    streamed directly into the BigTIFF while a downsampled buffer (2× or 4×,
    chosen adaptively) is accumulated in memory.  After the main page every
    further pyramid level is generated by cascading (resize → write) from
    that buffer, giving per-tile progress for every level.
    """
    tifffile = _load_tiff_runtime(require_imagecodecs=True)
    tile_size = options.resolved_tile_size()
    parallel = _resolve_parallel_settings(options)
    quality = options.jpeg_quality

    # -- pyramid shape plan --
    pyramid_shapes = (
        compute_pyramid_shapes(slide.width, slide.height)
        if options.generate_dense_pyramid
        else []
    )
    main_w, main_h = slide.width, slide.height
    main_tiles = tile_count(main_w, main_h, tile_size)

    # -- overall progress budget --
    pyramid_tiles = sum(
        tile_count(w, h, tile_size) for w, h in pyramid_shapes
    )
    overall_total = (
        1                       # "解析 IBL"
        + main_tiles
        + pyramid_tiles
        + 1                     # "写出文件"
    )

    # -- "解析 IBL" phase marker --
    perf["current_stage"] = "解析 IBL"
    _emit_progress(progress_callback, "解析 IBL", 1, 1, 1, overall_total)
    overall_offset = 1

    strip_height = max(tile_size, min(parallel["chunk_size"], tile_size * 2))
    strip_height = max(tile_size, (strip_height // tile_size) * tile_size)
    common_kwargs = {
        "photometric": "rgb",
        "tile": (tile_size, tile_size),
        "compression": "jpeg",
        "compressionargs": {"outcolorspace": "rgb"},
        "subsampling": (1, 1),
        "metadata": None,
        "dtype": np.uint8,
    }

    with tifffile.TiffWriter(str(output_path), bigtiff=True) as tif:
        # ================================================================
        # Main page (streaming from IBL, accumulating downsampled buffer)
        # ================================================================
        perf["current_stage"] = "构建主图"
        perf["max_level_reached"] = 0
        main_progress = ProgressState()
        stats: dict[str, float] = {}

        drive = DensePyramidDrive(
            slide,
            tile_size,
            strip_height,
            memory_budget_mb=options.memory_budget_mb,
            progress=main_progress,
            cancel_event=cancel_event,
            stats=stats,
            notify=lambda state: _emit_progress(
                progress_callback,
                "构建主图",
                state.done,
                state.total,
                overall_offset + state.done,
                overall_total,
            ),
        )

        encoder = _EncodedTileIter(drive, quality)
        started = time.perf_counter()
        tif.write(
            data=iter(encoder),
            shape=(main_h, main_w, 3),
            description=build_generic_description(slide),
            software=f"{APP_NAME} {APP_VERSION}",
            subfiletype=0,
            jpegtables=_get_jpeg_tables_for_shape(tile_size, tile_size, quality),
            **_resolution_kwargs_for_mpp(
                slide.base_info.mpp,
                base_width=main_w,
                base_height=main_h,
                level_width=main_w,
                level_height=main_h,
            ),
            **common_kwargs,
        )
        perf["read_decode_sec"] += stats.get("read_decode_sec", 0.0)
        perf["main_write_sec"] += time.perf_counter() - started
        overall_offset += main_tiles

        # ================================================================
        # Pyramid levels — cascade from the accumulated buffer
        # ================================================================
        ds_factor = drive.downsample_factor  # 2 or 4
        buf = drive.accumulation_buffer
        buf_h, buf_w = buf.shape[:2]
        buf_image = Image.fromarray(buf)

        # Determine which pyramid levels to write.
        # If we accumulated at 2× we write ALL levels; at 4× we skip the
        # first (2×) entry from *pyramid_shapes*.
        level_shapes: list[tuple[int, int]] = []
        if ds_factor == 2:
            level_shapes = list(pyramid_shapes)
        else:
            # ds_factor == 4 — keep levels ≤ buffer (1 px tolerance for rounding)
            level_shapes = [
                (w, h) for w, h in pyramid_shapes
                if w <= buf_w + 1 and h <= buf_h + 1
            ]

        for level_index, (lw, lh) in enumerate(level_shapes, start=1):
            perf["current_stage"] = "生成金字塔"
            perf["max_level_reached"] = level_index
            level_progress = ProgressState()

            if (lw, lh) == (buf_w, buf_h):
                level_source = PILImageSource(buf_image)
            else:
                level_resized = buf_image.resize(
                    (lw, lh), resample=Image.Resampling.BILINEAR
                )
                level_source = PILImageSource(level_resized)
                buf_image = level_resized

            lvl_iter = iter_source_tiles(
                level_source,
                tile_size,
                chunk_size=parallel["chunk_size"],
                progress=level_progress,
                cancel_event=cancel_event,
                notify=lambda state: _emit_progress(
                    progress_callback,
                    "生成金字塔",
                    state.done,
                    state.total,
                    overall_offset + state.done,
                    overall_total,
                ),
            )

            lvl_encoder = _EncodedTileIter(lvl_iter, quality)
            started = time.perf_counter()
            tif.write(
                data=iter(lvl_encoder),
                shape=(lh, lw, 3),
                description="",
                software=f"{APP_NAME} {APP_VERSION}",
                subfiletype=1,
                jpegtables=_get_jpeg_tables_for_shape(tile_size, tile_size, quality),
                **_resolution_kwargs_for_mpp(
                    slide.base_info.mpp,
                    base_width=main_w,
                    base_height=main_h,
                    level_width=lw,
                    level_height=lh,
                ),
                **common_kwargs,
            )
            perf["pyramid_sec"] += time.perf_counter() - started
            overall_offset += tile_count(lw, lh, tile_size)

    _advance_write_tail_progress(
        perf,
        progress_callback,
        step_done=1,
        step_total=1,
        overall_done=overall_total,
        overall_total=overall_total,
    )

    perf["level_dimensions"] = [(main_w, main_h), *level_shapes]
    return len(level_shapes) + 1


def _write_svs_streaming_direct(
    slide: IBLSlide,
    output_path: Path,
    options: ConvertOptions,
    perf: dict[str, Any],
    *,
    progress_callback=None,
    cancel_event=None,
) -> tuple[str, int, int]:
    """Single-pass streaming SVS writer — no temp files, no pyvips.

    Reads IBL data exactly once.  Each strip is simultaneously fed as JPEG
    tiles to the main page AND downsampled into a 4× accumulation buffer.
    The 8× pyramid level is generated in-memory from the completed 4×
    buffer.  No pyvips/libvips or intermediate temporary files needed.
    """
    tifffile = _load_tiff_runtime(require_imagecodecs=True)
    layout = build_aperio_svs_layout(slide, options)
    desired_levels = layout["pyramid"]
    tile_size = options.resolved_tile_size()
    parallel = _resolve_parallel_settings(options)
    bigtiff = bool(layout["bigtiff"])
    write_tail_total = _svs_write_tail_units(options)
    quality = options.jpeg_quality

    perf["level_dimensions"] = [(slide.width, slide.height), *desired_levels]
    perf["svs_is_bigtiff"] = bigtiff

    main_w, main_h = slide.width, slide.height
    main_tiles = tile_count(main_w, main_h, tile_size)
    associated_units = _svs_associated_units(options)

    # Compute per-level tile counts
    level_tiles: list[int] = []
    for lw, lh in desired_levels:
        level_tiles.append(tile_count(lw, lh, tile_size))

    overall_total = (
        1  # "解析 IBL"
        + main_tiles
        + 1  # thumbnail
        + sum(level_tiles)
        + associated_units
        + write_tail_total
    )
    overall_offset = 1

    # -- "解析 IBL" phase marker --
    perf["current_stage"] = "解析 IBL"
    _emit_progress(progress_callback, "解析 IBL", 1, 1, 1, overall_total)

    # -- pre-allocate 4× accumulation buffer --
    first_level_w, first_level_h = desired_levels[0] if desired_levels else (1, 1)
    l4_buffer = np.zeros((first_level_h, first_level_w, 3), dtype=np.uint8)

    iccprofile = _get_srgb_icc_profile()
    common = _svs_common_kwargs(options)
    strip_height = max(tile_size, min(parallel["chunk_size"], tile_size * 2))
    strip_height = max(tile_size, (strip_height // tile_size) * tile_size)

    with tifffile.TiffWriter(str(output_path), bigtiff=bigtiff) as tif:
        # ================================================================
        # PAGE 0: Main image  (subfiletype=0, tiled JPEG)
        # ================================================================
        perf["current_stage"] = "构建主图"
        perf["max_level_reached"] = 0
        main_progress = ProgressState()
        stats: dict[str, float] = {}

        main_drive = StripDownsampleDrive(
            slide,
            tile_size,
            strip_height,
            l4_buffer,
            progress=main_progress,
            cancel_event=cancel_event,
            stats=stats,
            notify=lambda state: _emit_progress(
                progress_callback,
                "构建主图",
                state.done,
                state.total,
                overall_offset + state.done,
                overall_total,
            ),
        )

        main_encoder = _EncodedTileIter(main_drive, quality)
        started = time.perf_counter()
        tif.write(
            data=iter(main_encoder),
            shape=(main_h, main_w, 3),
            description=build_aperio_description(slide, options),
            software="",
            iccprofile=iccprofile,
            subfiletype=0,
            jpegtables=_get_jpeg_tables_for_shape(tile_size, tile_size, quality),
            **common,
        )
        perf["read_decode_sec"] += stats.get("read_decode_sec", 0.0)
        perf["main_write_sec"] += time.perf_counter() - started
        overall_offset += main_tiles

        # ================================================================
        # PAGE 1: Thumbnail  (not tiled, JPEG)
        # ================================================================
        overall_offset += write_aperio_thumbnail_page(
            tif,
            slide,
            options,
            perf,
            iccprofile=iccprofile,
            overall_done=overall_offset,
            overall_total=overall_total,
            thumbnail_max=layout["thumbnail_max"],
            progress_callback=progress_callback,
        )

        # ================================================================
        # Pyramid levels — cascade from the 4× accumulation buffer
        # ================================================================
        prev_image = Image.fromarray(l4_buffer)
        for level_index, ((lw, lh), lt) in enumerate(zip(desired_levels, level_tiles)):
            perf["current_stage"] = "生成金字塔"
            perf["max_level_reached"] = level_index + 1
            level_progress = ProgressState()

            if level_index == 0:
                # First pyramid level (4×) — use buffer directly
                level_source = PILImageSource(prev_image)
            else:
                # Cascaded level — down-sample from the previous
                level_resized = prev_image.resize((lw, lh), resample=Image.Resampling.BILINEAR)
                prev_image = level_resized
                level_source = PILImageSource(level_resized)

            level_iter = iter_source_tiles(
                level_source,
                tile_size,
                chunk_size=parallel["chunk_size"],
                progress=level_progress,
                cancel_event=cancel_event,
                notify=lambda state, _lt=lt: _emit_progress(
                    progress_callback,
                    "生成金字塔",
                    state.done,
                    state.total,
                    overall_offset + state.done,
                    overall_total,
                ),
            )

            level_encoder = _EncodedTileIter(level_iter, quality)
            started = time.perf_counter()
            tif.write(
                data=iter(level_encoder),
                shape=(lh, lw, 3),
                description=build_aperio_pyramid_description(slide, options, lw, lh),
                software="",
                iccprofile=iccprofile,
                subfiletype=0,
                jpegtables=_get_jpeg_tables_for_shape(tile_size, tile_size, quality),
                **common,
            )
            perf["pyramid_sec"] += time.perf_counter() - started
            overall_offset += lt

        # ================================================================
        # PAGES 4 & 5: Label + Macro  (subfiletype=1 / 9)
        # ================================================================
        associated = write_aperio_associated_images(
            tif,
            slide,
            options,
            perf,
            overall_done=overall_offset,
            overall_total=overall_total,
            progress_callback=progress_callback,
        )
        perf.update({k: v for k, v in associated.items() if v is not None})

    completed = overall_offset + associated_units + 1
    _advance_write_tail_progress(
        perf,
        progress_callback,
        step_done=1,
        step_total=write_tail_total,
        overall_done=completed,
        overall_total=overall_total,
    )

    return "svs-streaming-direct", completed, overall_total


def write_image(
    slide: IBLSlide,
    output_path: str | Path,
    options: ConvertOptions,
    progress_callback=None,
    cancel_event=None,
) -> tuple[int, dict[str, float]]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tile_size = options.resolved_tile_size()

    if tile_size <= 0 or tile_size % 16 != 0:
        raise RuntimeError("JPEG 压缩的 TIFF tile_size 必须是 16 的正整数倍")

    perf_tracker = PerfTracker()
    parallel = _resolve_parallel_settings(options)
    if options.output_format == "svs":
        level_dimensions = [(slide.width, slide.height), *_compute_svs_pyramid_shapes(slide, options)]
    else:
        shapes = compute_pyramid_shapes(slide.width, slide.height) if options.generate_dense_pyramid else []
        level_dimensions = [(slide.width, slide.height), *shapes]
    perf = {
        "backend": options.performance_backend,
        "current_stage": "初始化",
        "level_dimensions": level_dimensions,
        "read_decode_sec": 0.0,
        "main_write_sec": 0.0,
        "pyramid_sec": 0.0,
        "thumbnail_sec": 0.0,
        "encode_sec": 0.0,
        "writer_wait_sec": 0.0,
        "peak_memory_mb": 0.0,
        "avg_cpu_percent": 0.0,
        "svs_finalize_backend": None,
        "svs_photometric_pages": None,
        "openslide_vendor": None,
        "max_level_reached": None,
        "failure_stage": None,
        **parallel,
    }

    try:
        if options.output_format == "generic_tiff":
            pyramid_levels = _write_generic_tiff_streaming(
                slide,
                output_path,
                options,
                perf,
                progress_callback=progress_callback,
                cancel_event=cancel_event,
            )
            perf["backend"] = "tifffile-streaming"
        elif options.output_format == "svs":
            perf["backend"], overall_done, overall_total = _write_svs_streaming_direct(
                slide,
                output_path,
                options,
                perf,
                progress_callback=progress_callback,
                cancel_event=cancel_event,
            )
            write_tail_total = _svs_write_tail_units(options)
            write_tail_done = 1
            if options.svs_finalize_with_libtiff:
                write_tail_done += 1
                perf["svs_finalize_backend"] = finalize_svs_with_libtiff(output_path)
                _advance_write_tail_progress(
                    perf,
                    progress_callback,
                    step_done=write_tail_done,
                    step_total=write_tail_total,
                    overall_done=min(overall_total, overall_done + 1),
                    overall_total=overall_total,
                )
                overall_done = min(overall_total, overall_done + 1)
            if options.svs_validate_with_tiffinfo:
                write_tail_done += 1
                perf.update(inspect_svs_compatibility(output_path))
                _advance_write_tail_progress(
                    perf,
                    progress_callback,
                    step_done=write_tail_done,
                    step_total=write_tail_total,
                    overall_done=min(overall_total, overall_done + 1),
                    overall_total=overall_total,
                )
                overall_done = min(overall_total, overall_done + 1)
        else:
            raise RuntimeError(f"不支持的输出格式: {options.output_format}")
    except Exception as exc:
        perf_tracker.sample()
        perf["peak_memory_mb"] = perf_tracker.peak_memory_mb
        perf["avg_cpu_percent"] = perf_tracker.average_cpu_percent()
        perf["failure_stage"] = perf.get("failure_stage") or perf.get("current_stage")
        perf["encode_sec"] = max(
            0.0,
            perf["main_write_sec"] + perf["pyramid_sec"] - perf["writer_wait_sec"],
        )
        raise WriteImageError(str(exc), perf) from exc

    perf_tracker.sample()
    perf["peak_memory_mb"] = perf_tracker.peak_memory_mb
    perf["avg_cpu_percent"] = perf_tracker.average_cpu_percent()
    perf["encode_sec"] = max(
        0.0,
        perf["main_write_sec"] + perf["pyramid_sec"] - perf["writer_wait_sec"],
    )
    if options.output_format == "generic_tiff":
        return pyramid_levels, perf
    return len(level_dimensions), perf


write_svs = write_image
