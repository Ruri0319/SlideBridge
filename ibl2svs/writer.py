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
    NativeLevelSource,
    PILImageSource,
    ProgressState,
    ResizedSource,
    StripDownsampleDrive,
    compute_pyramid_shapes,
    iter_source_tiles,
    tile_count,
)
from .models import ConvertOptions
from .native_jpeg import iter_full_size_jpeg_tiles
from .reader import IBLSlide
from .system_metrics import PerfTracker


class WriteImageError(RuntimeError):
    def __init__(self, message: str, perf: dict[str, Any]):
        super().__init__(message)
        self.perf = perf


APERIO_LIBRARY_VERSION = "12.0.11"
APERIO_JP2000_YCBC = 33003
APERIO_IMAGE_DEPTH_TAG = 32997
APERIO_JPEG_SUBSAMPLING = (2, 2)
SVS_CLASSIC_TIFF_SAFE_LIMIT = 3_200_000_000
SVS_TIFF_ESTIMATE_OVERHEAD = 64 * 1024 * 1024


def _aperio_library_header(slide) -> str:
    version = str(getattr(slide, "aperio_library_version", "") or APERIO_LIBRARY_VERSION).strip()
    return f"Aperio Image Library v{version}"


def _svs_codec(slide, options: ConvertOptions) -> str:
    return options.resolved_svs_codec(getattr(slide, "source_codec", None))


def _svs_tile_size(slide, options: ConvertOptions) -> int:
    return options.resolved_svs_tile_size(getattr(slide, "source_codec", None))


def _svs_quality(slide, options: ConvertOptions) -> int:
    return max(1, min(100, int(options.main_quality)))


def _svs_pyramid_quality(slide, options: ConvertOptions, level_index: int) -> int:
    del level_index
    return max(1, min(100, int(options.pyramid_quality)))


def _svs_preview_quality(options: ConvertOptions) -> int:
    return max(1, min(100, int(options.preview_quality)))


def _svs_estimated_bytes_per_pixel(codec: str, quality: int) -> float:
    """Estimate encoded tile density for the selected codec and quality."""
    if codec == "aperio_j2k":
        return 0.50 + 0.035 * quality
    return 0.15 + 0.010 * quality


def _svs_description_codec(slide, options: ConvertOptions, quality: int | None = None) -> str:
    codec = _svs_codec(slide, options)
    value = _svs_quality(slide, options) if quality is None else quality
    return f"J2K/YUV16 Q={value}" if codec == "aperio_j2k" else f"JPEG/RGB Q={value}"


def _svs_icc_profile(slide) -> bytes:
    profile = getattr(slide, "icc_profile", None)
    return bytes(profile) if isinstance(profile, bytes) else _get_srgb_icc_profile()


def _svs_icc_name(slide) -> str:
    return str(getattr(slide, "icc_profile_name", "") or "sRGB")


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


def _is_native_source(slide) -> bool:
    return bool(
        getattr(slide, "supports_native_pyramid", False)
        and callable(getattr(slide, "iter_native_level_jpegs", None))
        and callable(getattr(slide, "read_level_region", None))
    )


def _native_output_ready(slide) -> bool:
    ready = getattr(slide, "native_output_ready", None)
    if ready is not None:
        return _is_native_source(slide) and bool(ready)
    return _is_native_source(slide) and getattr(slide, "native_resource_status", "") == "native"


def _mpp_xy(slide) -> tuple[float, float]:
    mpp = float(getattr(slide.base_info, "mpp", 0.0) or 0.0)
    mpp_x = float(getattr(slide, "mpp_x", 0.0) or 0.0)
    mpp_y = float(getattr(slide, "mpp_y", 0.0) or 0.0)
    getter = getattr(slide, "get_scan_metadata", None)
    if callable(getter):
        try:
            metadata = getter()
        except Exception:
            metadata = {}
        mpp_x = float(metadata.get("mppX", mpp_x) or mpp_x or mpp)
        mpp_y = float(metadata.get("mppY", mpp_y) or mpp_y or mpp)
    return mpp_x or mpp, mpp_y or mpp


def build_generic_description(slide: IBLSlide) -> str:
    app_mag = slide.base_info.max_zoom_rate
    mpp_x, mpp_y = _mpp_xy(slide)
    return (
        f"{APP_NAME} generic pyramidal TIFF | "
        f"OriginalFile={slide.path.name} | "
        f"Width={slide.width} | Height={slide.height} | "
        f"MPP = {slide.base_info.mpp:.6f} | "
        f"MPP_X = {mpp_x:.6f} | MPP_Y = {mpp_y:.6f} | "
        f"AppMag = {app_mag} | "
        f"ObjectivePower = {app_mag} | "
        f"openslide.objective-power = {app_mag}"
    )


def build_aperio_description(slide: IBLSlide, options: ConvertOptions) -> str:
    tile_size = _svs_tile_size(slide, options)
    return (
        f"{_aperio_library_header(slide)} \r\n"
        f"{slide.width}x{slide.height} [0,0 {slide.width}x{slide.height}] "
        f"({tile_size}x{tile_size}) {_svs_description_codec(slide, options)}|"
        f"AppMag = {slide.base_info.max_zoom_rate}|"
        f"Filename = {slide.path.stem}|"
        f"MPP = {slide.base_info.mpp:.6f}|"
        "DisplayColor = 0|"
        f"OriginalWidth = {slide.width}|"
        f"OriginalHeight = {slide.height}|"
        f"ICC Profile = {_svs_icc_name(slide)}"
    )


def build_aperio_pyramid_description(
    slide: IBLSlide,
    options: ConvertOptions,
    level_width: int,
    level_height: int,
    quality: int | None = None,
) -> str:
    tile_size = _svs_tile_size(slide, options)
    return (
        f"{_aperio_library_header(slide)} \r\n"
        f"{slide.width}x{slide.height} [0,0 {slide.width}x{slide.height}] "
        f"({tile_size}x{tile_size}) -> {level_width}x{level_height} "
        f"{_svs_description_codec(slide, options, quality=quality)}"
    )


def build_aperio_thumbnail_description(slide: IBLSlide, width: int, height: int) -> str:
    return (
        f"{_aperio_library_header(slide)} \n"
        f"{slide.width}x{slide.height} -> {width}x{height} - |"
        f"AppMag = {slide.base_info.max_zoom_rate}|"
        f"Filename = {slide.path.stem}|"
        f"MPP = {slide.base_info.mpp:.6f}|"
        "DisplayColor = 0|"
        f"OriginalWidth = {slide.width}|"
        f"OriginalHeight = {slide.height}|"
        f"ICC Profile = {_svs_icc_name(slide)}"
    )


def build_aperio_label_description(width: int, height: int) -> str:
    return f"Aperio Image Library v{APERIO_LIBRARY_VERSION} \nlabel {width}x{height}"


def build_aperio_macro_description(width: int, height: int) -> str:
    return f"Aperio Image Library v{APERIO_LIBRARY_VERSION} \nmacro {width}x{height}"


def _resolution_kwargs_for_mpp(
    mpp: float,
    *,
    mpp_x: float | None = None,
    mpp_y: float | None = None,
    base_width: int,
    base_height: int,
    level_width: int,
    level_height: int,
) -> dict[str, Any]:
    mpp_x = mpp if mpp_x is None else mpp_x
    mpp_y = mpp if mpp_y is None else mpp_y
    if mpp_x <= 0 or mpp_y <= 0:
        return {}
    x_downsample = base_width / max(1, level_width)
    y_downsample = base_height / max(1, level_height)
    x_pixels_per_cm = 10000.0 / (mpp_x * x_downsample)
    y_pixels_per_cm = 10000.0 / (mpp_y * y_downsample)
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
    # exceed Classic TIFF's 4 GB file-offset limit.  Use the actual output
    # levels and account for the codec and quality used on each level.
    tile_size = _svs_tile_size(slide, options)
    codec = _svs_codec(slide, options)
    level_shapes = [(slide.width, slide.height), *_compute_svs_pyramid_shapes(slide, options)]
    estimated_bytes = SVS_TIFF_ESTIMATE_OVERHEAD
    for level_index, (width, height) in enumerate(level_shapes):
        quality = (
            _svs_quality(slide, options)
            if level_index == 0
            else _svs_pyramid_quality(slide, options, level_index - 1)
        )
        padded_pixels = tile_count(width, height, tile_size) * tile_size * tile_size
        estimated_bytes += padded_pixels * _svs_estimated_bytes_per_pixel(codec, quality)

    return estimated_bytes > SVS_CLASSIC_TIFF_SAFE_LIMIT


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

    Uses unlabelled RGB component IDs (0, 1, 2), which is the abbreviated
    JPEG convention observed in the supplied Aperio ``JPEG/RGB`` tiles.
    ``unknown`` keeps the input channels in RGB order without adding a JFIF
    or Adobe APP marker.
    """
    from imagecodecs import jpeg8_encode

    return jpeg8_encode(data, level=quality, colorspace="unknown", outcolorspace="unknown", optimize=False)


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


def _encode_aperio_j2k(data: np.ndarray, quality: int) -> bytes:
    from imagecodecs import JPEG2K, jpeg2k_encode

    return bytes(
        jpeg2k_encode(
            np.ascontiguousarray(data),
            level=quality,
            codecformat=JPEG2K.CODEC.J2K,
            colorspace=JPEG2K.CLRSPC.SYCC,
            mct=True,
        )
    )


class _EncodedTileIter:
    """Wrap a tile-ndarray iterator so it yields **stripped** JPEG bytes.

    Each yielded tile is ``SOI + SOF0 + SOS + data + EOI`` — an
    abbreviated JPEG with **no** DQT or DHT markers.  The shared tables
    are written separately via the ``JPEGTables`` TIFF tag (347).

    OpenSlide / libjpeg loads the tables once and reuses them for every
    tile, keeping per-tile decoder allocations to a minimum.
    """

    def __init__(self, source, quality: int, codec: str = "jpeg_rgb"):
        self._source = source
        self._quality = quality
        self._codec = codec

    def __iter__(self):
        for tile in self._source:
            if self._codec == "aperio_j2k":
                yield _encode_aperio_j2k(tile, self._quality)
            else:
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
    fd, raw_path = tempfile.mkstemp(prefix="slidebridge_flat_", suffix=suffix, dir=directory)
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
        try:
            proc = subprocess.run(
                ["tiffset", "-d", str(index), "-s", str(APERIO_IMAGE_DEPTH_TAG), "1", str(path)],
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            return "unavailable"
        cleaned = cleaned or proc.returncode == 0
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
    source_dimensions = getattr(slide, "svs_level_dimensions", None)
    if getattr(slide, "source_container", "") == "svs" and source_dimensions:
        return [tuple(map(int, dimensions)) for dimensions in source_dimensions[1:]]

    shapes: list[tuple[int, int]] = []
    for downsample in (4, 16, 32):
        width = max(1, slide.width // downsample)
        height = max(1, slide.height // downsample)
        if min(width, height) < 512:
            break
        shapes.append((width, height))
    if not shapes and min(slide.width, slide.height) >= 128:
        shapes.append((max(1, slide.width // 4), max(1, slide.height // 4)))
    return shapes


def _native_level_for_shape(slide, width: int, height: int) -> int:
    dimensions = list(getattr(slide, "level_dimensions", []))
    if not dimensions:
        return 0
    return min(
        range(len(dimensions)),
        key=lambda index: abs(dimensions[index][0] - width) + abs(dimensions[index][1] - height),
    )


def _native_level_source_for_shape(slide, width: int, height: int):
    level_index = _native_level_for_shape(slide, width, height)
    native = NativeLevelSource(slide, level_index)
    if (native.width, native.height) == (width, height):
        return native, level_index
    return ResizedSource(native, width, height), level_index


def _native_resource_image(slide, name: str) -> Image.Image | None:
    if not _is_native_source(slide) or getattr(slide, "native_resource_status", "") != "native":
        return None
    getter = getattr(slide, f"get_{name}_image", None)
    if not callable(getter):
        return None
    return getter()


def _source_associated_image(slide, name: str) -> Image.Image | None:
    native_getter = getattr(slide, "get_native_associated_image", None)
    if callable(native_getter):
        image = native_getter(name)
        if image is not None:
            return image

    image = _native_resource_image(slide, name)
    if image is not None:
        return image

    getter = getattr(slide, f"get_{name}_image", None)
    if callable(getter):
        return getter()
    return None


def _write_native_page(
    tif,
    image: Image.Image,
    *,
    description: str | None,
    subfiletype: int,
    metadata: dict[str, Any] | None = None,
) -> tuple[int, int]:
    if image.mode in {"1", "L"}:
        array = np.asarray(image, dtype=np.uint8)
        photometric = "minisblack"
    elif image.mode.startswith("I;16"):
        array = np.asarray(image, dtype=np.uint16)
        photometric = "minisblack"
    else:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8)
        photometric = "rgb"
    height, width = array.shape[:2]
    tif.write(
        array,
        photometric=photometric,
        compression="lzw",
        rowsperstrip=max(1, min(height, 64)),
        description=description,
        metadata=metadata,
        software="",
        subfiletype=subfiletype,
    )
    return width, height


def _source_requires_raw_ome(slide) -> bool:
    return bool(getattr(slide, "requires_raw_ome", False))


def _svs_omitted_native_data(slide) -> str | None:
    if not _source_requires_raw_ome(slide):
        return None
    fields = tuple(getattr(slide, "native_fields", (0,)))
    return (
        f"fields={len(fields)}, C={int(getattr(slide, 'source_channel_count', 1))}, "
        f"Z={int(getattr(slide, 'native_z_count', 1))}, "
        f"T={int(getattr(slide, 'native_t_count', 1))}, "
        f"bit_depth={int(getattr(slide, 'source_bit_depth', 8))}; "
        f"仅保存 field={int(getattr(slide, 'default_field_index', 0))} 的 RGB 合成观感"
    )


def _write_native_ome_tiff(
    slide,
    output_path: Path,
    options: ConvertOptions,
    perf: dict[str, Any],
    *,
    progress_callback=None,
    cancel_event=None,
) -> int:
    """Write a vendor-native JPEG pyramid without resampling."""

    tifffile = _load_tiff_runtime(require_imagecodecs=True)
    tile_size = int(getattr(slide, "tile_size", 256))
    levels = list(slide.levels)
    mpp_x, mpp_y = _mpp_xy(slide)
    resource_status = getattr(slide, "native_resource_status", "unavailable")
    resource_dimensions = dict(getattr(slide, "native_resource_dimensions", {}))
    aux_names = tuple(
        name for name in ("thumbnail", "macro", "label")
        if resource_status == "native" and resource_dimensions.get(name) is not None
    )
    display_tiles = sum(level.columns * level.rows for level in levels)
    overall_total = 1 + display_tiles + len(aux_names) + 1
    overall_done = 1
    source_name = str(getattr(slide, "source_container", "IMAGE")).upper()
    perf["current_stage"] = f"解析 {source_name}"
    perf["native_path"] = True
    perf["native_level_dimensions"] = list(slide.level_dimensions)
    perf["native_resource_dimensions"] = dict(getattr(slide, "native_resource_dimensions", {}))
    perf["native_fallback_reason"] = getattr(slide, "native_resource_reason", "") or None
    _emit_progress(progress_callback, f"解析 {source_name}", 1, 1, overall_done, overall_total)

    parallel_native_iterator = getattr(slide, "iter_native_level_jpegs_parallel", None)
    native_transform_workers = (
        max(1, min(4, int(options.encoder_workers or 2)))
        if callable(parallel_native_iterator)
        else 1
    )
    if callable(parallel_native_iterator):
        perf["encoder_workers"] = native_transform_workers
        perf["raw_queue_size"] = native_transform_workers * 2
        perf["encoded_queue_size"] = native_transform_workers * 2

    source_channels = int(getattr(slide, "source_channel_count", 3))
    encoded_passthrough = not hasattr(slide, "source_codec") or getattr(slide, "native_tile_mode", "") == "jpeg_passthrough"
    if source_channels == 1:
        from imagecodecs import jpeg8_encode

        blank_tile = jpeg8_encode(
            np.full((tile_size, tile_size), slide.base_info.background_color, dtype=np.uint8),
            level=100,
        )
        page_shape_suffix: tuple[int, ...] = ()
        page_photometric = "minisblack"
    else:
        blank_tile = _encode_jpeg(
            np.full((tile_size, tile_size, 3), slide.base_info.background_color, dtype=np.uint8),
            100,
        )
        page_shape_suffix = (3,)
        page_photometric = "rgb"

    with tifffile.TiffWriter(str(output_path), bigtiff=True, ome=True) as tif:
        for level_index, level in enumerate(levels):
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("转换已取消")

            width, height = level.dimensions
            description = None
            page_metadata = None
            page_subifds = None
            if level_index == 0:
                page_subifds = len(levels) - 1
                page_metadata = {
                    "axes": "YX" if source_channels == 1 else "YXS",
                    "Name": f"{slide.path.stem} {'grayscale' if source_channels == 1 else 'RGB'}",
                    "Description": build_generic_description(slide),
                }
                if mpp_x > 0:
                    page_metadata.update({"PhysicalSizeX": mpp_x, "PhysicalSizeXUnit": "µm"})
                if mpp_y > 0:
                    page_metadata.update({"PhysicalSizeY": mpp_y, "PhysicalSizeYUnit": "µm"})
            resolution = _resolution_kwargs_for_mpp(
                slide.base_info.mpp,
                mpp_x=mpp_x,
                mpp_y=mpp_y,
                base_width=slide.width,
                base_height=slide.height,
                level_width=width,
                level_height=height,
            )
            level_tile_total = level.columns * level.rows
            level_base_done = overall_done
            last_progress_at = 0.0

            def report_tiles(payloads):
                nonlocal last_progress_at
                for tile_done, payload in enumerate(payloads, start=1):
                    now = time.monotonic()
                    if tile_done == 1 or tile_done == level_tile_total or now - last_progress_at >= 0.25:
                        _emit_progress(
                            progress_callback,
                            "写出原生层",
                            tile_done,
                            level_tile_total,
                            level_base_done + tile_done,
                            overall_total,
                        )
                        last_progress_at = now
                    yield payload

            perf["current_stage"] = "写出原生层"
            level_started = time.perf_counter()
            if encoded_passthrough:
                def encoded_tiles(level_index=level_index):
                    source_payloads = (
                        parallel_native_iterator(
                            level_index,
                            workers=native_transform_workers,
                            cancel_event=cancel_event,
                        )
                        if callable(parallel_native_iterator)
                        else slide.iter_native_level_jpegs(level_index)
                    )
                    yield from iter_full_size_jpeg_tiles(
                        source_payloads,
                        tile_size=tile_size,
                        blank_tile=blank_tile,
                        background=int(slide.base_info.background_color),
                        quality=100,
                        cancel_event=cancel_event,
                    )

                tif.write(
                    data=report_tiles(encoded_tiles()),
                    shape=(height, width, *page_shape_suffix),
                    dtype=np.uint8,
                    photometric=page_photometric,
                    tile=(tile_size, tile_size),
                    compression="jpeg",
                    subsampling=(1, 1),
                    metadata=page_metadata,
                    software=f"{APP_NAME} {APP_VERSION}",
                    description=description,
                    subfiletype=0 if level_index == 0 else 1,
                    subifds=page_subifds,
                    **resolution,
                )
            else:
                source = NativeLevelSource(slide, level_index)
                decoded_tiles = iter_source_tiles(
                    source,
                    tile_size,
                    chunk_size=max(tile_size, options.resolved_chunk_size()),
                    cancel_event=cancel_event,
                )
                tif.write(
                    data=report_tiles(decoded_tiles),
                    shape=(height, width, 3),
                    dtype=np.uint8,
                    photometric="rgb",
                    tile=(tile_size, tile_size),
                    compression="deflate",
                    predictor=True,
                    metadata=page_metadata,
                    software=f"{APP_NAME} {APP_VERSION}",
                    description=description,
                    subfiletype=0 if level_index == 0 else 1,
                    subifds=page_subifds,
                    **resolution,
                )
            level_elapsed = time.perf_counter() - level_started
            if level_index == 0:
                perf["main_write_sec"] += level_elapsed
            else:
                perf["pyramid_sec"] += level_elapsed
            overall_done += level_tile_total

        if resource_status == "native":
            for name in aux_names:
                image = _native_resource_image(slide, name)
                if image is None:
                    continue
                width, height = image.size
                _write_native_page(
                    tif,
                    image,
                    description=None,
                    subfiletype=9 if name == "macro" else 1,
                    metadata={
                        "axes": "YXS" if image.mode not in {"1", "L"} else "YX",
                        "Name": name,
                    },
                )
                overall_done += 1
                _emit_progress(
                    progress_callback,
                    "写出原生附属图",
                    1,
                    len(aux_names),
                    overall_done,
                    overall_total,
                )
    perf["native_tile_mode"] = getattr(slide, "native_tile_mode", "unknown")
    perf["level_dimensions"] = list(slide.level_dimensions)
    perf["max_level_reached"] = max(0, len(levels) - 1)
    _advance_write_tail_progress(
        perf,
        progress_callback,
        step_done=1,
        step_total=1,
        overall_done=overall_total,
        overall_total=overall_total,
    )
    return len(levels)


def _build_thumbnail_array(slide: IBLSlide, max_size: int = 1024) -> np.ndarray:
    native_getter = getattr(slide, "get_native_associated_image", None)
    native = native_getter("thumbnail") if callable(native_getter) else None
    if native is not None:
        return np.asarray(native.convert("RGB"), dtype=np.uint8)

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
    native_getter = getattr(slide, "get_native_associated_image", None)
    native = native_getter("macro") if callable(native_getter) else None
    if native is not None:
        return np.asarray(native.convert("RGB"), dtype=np.uint8)

    macro_getter = getattr(slide, "get_macro_image", None)
    image = macro_getter() if callable(macro_getter) else None
    overview_getter = getattr(slide, "get_overview_image", None)
    if image is None and callable(overview_getter):
        image = overview_getter()
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
    native_getter = getattr(slide, "get_native_associated_image", None)
    native = native_getter("label") if callable(native_getter) else None
    if native is not None:
        return np.asarray(native.convert("RGB"), dtype=np.uint8)

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
    elif isinstance(scan_time, str):
        scan_time = scan_time.strip()
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
    associated_total = _svs_associated_units(options)
    associated_done = 0

    source_images: dict[str, Image.Image] = {}
    for name, enabled in (
        ("label", options.svs_generate_label),
        ("macro", options.svs_generate_macro),
    ):
        if enabled:
            image = _source_associated_image(slide, name)
            if image is not None:
                source_images[name] = image

    for name in ("label", "macro"):
        image = source_images.get(name)
        if image is None:
            continue
        perf["current_stage"] = "生成附属图像"
        started = time.perf_counter()
        width, height = _write_native_page(
            tif,
            image,
            description=f"{_aperio_library_header(slide)} \n{name} {image.width}x{image.height}",
            subfiletype=1 if name == "label" else 9,
        )
        perf["thumbnail_sec"] += time.perf_counter() - started
        metadata[f"svs_{name}_dimensions"] = (width, height)
        associated_done += 1
        _emit_progress(
            progress_callback,
            "生成附属图像",
            associated_done,
            max(1, associated_total),
            overall_done + associated_done,
            overall_total,
        )

    macro_array = (
        _build_macro_array(slide)
        if (
            options.svs_generate_macro
            and "macro" not in source_images
            and options.svs_synthesize_associated_images
        )
        else None
    )

    if (
        options.svs_generate_label
        and "label" not in source_images
        and options.svs_synthesize_associated_images
    ):
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
        tables, stripped = _encode_jpeg_stripped(macro_array, _svs_preview_quality(options))
        tif.write(
            data=iter([stripped]),
            shape=(mh, mw, 3),
            dtype=np.uint8,
            photometric="rgb",
            compression="jpeg",
            compressionargs={"outcolorspace": "rgb"},
            subsampling=APERIO_JPEG_SUBSAMPLING,
            jpegtables=tables,
            rowsperstrip=mh,
            description=build_aperio_macro_description(mw, mh),
            metadata=None,
            software="",
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


def _svs_common_kwargs(slide, options: ConvertOptions) -> dict[str, Any]:
    tile_size = _svs_tile_size(slide, options)
    if _svs_codec(slide, options) == "aperio_j2k":
        return {
            "photometric": "rgb",
            "tile": (tile_size, tile_size),
            "compression": APERIO_JP2000_YCBC,
            "dtype": np.uint8,
        }
    return {
        "photometric": "rgb",
        "tile": (tile_size, tile_size),
        "compression": "jpeg",
        "compressionargs": {"outcolorspace": "rgb"},
        "subsampling": APERIO_JPEG_SUBSAMPLING,
        "dtype": np.uint8,
    }


def write_aperio_thumbnail_page(
    tif,
    slide: IBLSlide,
    options: ConvertOptions,
    perf: dict[str, float],
    *,
    iccprofile: bytes | None,
    overall_done: int,
    overall_total: int,
    thumbnail_max: int,
    progress_callback=None,
) -> int:
    perf["current_stage"] = "生成缩略图"
    thumb_started = time.perf_counter()
    native = _native_resource_image(slide, "thumbnail")
    if native is not None:
        _write_native_page(
            tif,
            native,
            description=f"{_aperio_library_header(slide)} \nthumbnail {native.width}x{native.height}",
            subfiletype=1,
        )
        perf["thumbnail_sec"] += time.perf_counter() - thumb_started
        _emit_progress(progress_callback, "生成缩略图", 1, 1, overall_done + 1, overall_total)
        return 1

    thumbnail = _build_thumbnail_array(slide, max_size=thumbnail_max)
    h, w = thumbnail.shape[:2]
    tables, stripped = _encode_jpeg_stripped(thumbnail, _svs_preview_quality(options))
    tif.write(
        data=iter([stripped]),
        shape=(h, w, 3),
        dtype=np.uint8,
        photometric="rgb",
        compression="jpeg",
        compressionargs={"outcolorspace": "rgb"},
        subsampling=APERIO_JPEG_SUBSAMPLING,
        jpegtables=tables,
        rowsperstrip=h,
        description=build_aperio_thumbnail_description(slide, w, h),
        software="",
        metadata=None,
        subfiletype=0,
    )
    perf["thumbnail_sec"] += time.perf_counter() - thumb_started
    _emit_progress(progress_callback, "生成缩略图", 1, 1, overall_done + 1, overall_total)
    return 1


def _write_svs_native_streaming(
    slide,
    output_path: Path,
    options: ConvertOptions,
    perf: dict[str, Any],
    *,
    progress_callback=None,
    cancel_event=None,
) -> tuple[str, int, int]:
    tifffile = _load_tiff_runtime(require_imagecodecs=True)
    layout = build_aperio_svs_layout(slide, options)
    desired_levels = layout["pyramid"]
    tile_size = _svs_tile_size(slide, options)
    parallel = _resolve_parallel_settings(options)
    bigtiff = bool(layout["bigtiff"])
    codec = _svs_codec(slide, options)
    quality = _svs_quality(slide, options)
    associated_units = _svs_associated_units(options)
    write_tail_total = _svs_write_tail_units(options)
    level_tiles = [tile_count(width, height, tile_size) for width, height in desired_levels]
    main_tiles = tile_count(slide.width, slide.height, tile_size)
    overall_total = 1 + main_tiles + 1 + sum(level_tiles) + associated_units + write_tail_total
    overall_done = 1

    perf["native_path"] = True
    perf["native_level_dimensions"] = list(slide.level_dimensions)
    perf["native_resource_dimensions"] = dict(getattr(slide, "native_resource_dimensions", {}))
    perf["native_fallback_reason"] = getattr(slide, "native_resource_reason", "") or None
    perf["level_dimensions"] = [(slide.width, slide.height), *desired_levels]
    perf["svs_is_bigtiff"] = bigtiff
    perf["current_stage"] = "解析 IMAGE"
    _emit_progress(progress_callback, "解析 IMAGE", 1, 1, overall_done, overall_total)

    common = _svs_common_kwargs(slide, options)
    iccprofile = _svs_icc_profile(slide)
    strip_height = max(tile_size, min(parallel["chunk_size"], tile_size * 2))
    strip_height = max(tile_size, (strip_height // tile_size) * tile_size)

    with tifffile.TiffWriter(str(output_path), bigtiff=bigtiff) as tif:
        main_source = NativeLevelSource(slide, 0)
        main_progress = ProgressState()
        main_iter = iter_source_tiles(
            main_source,
            tile_size,
            chunk_size=strip_height,
            progress=main_progress,
            cancel_event=cancel_event,
            notify=lambda state: _emit_progress(
                progress_callback,
                "构建主图",
                state.done,
                state.total,
                overall_done + state.done,
                overall_total,
            ),
        )
        started = time.perf_counter()
        tif.write(
            data=iter(_EncodedTileIter(main_iter, quality, codec)),
            shape=(slide.height, slide.width, 3),
            description=build_aperio_description(slide, options),
            software="",
            iccprofile=iccprofile,
            subfiletype=0,
            jpegtables=(
                _get_jpeg_tables_for_shape(tile_size, tile_size, quality)
                if codec == "jpeg_rgb"
                else None
            ),
            **common,
        )
        perf["main_write_sec"] += time.perf_counter() - started
        overall_done += main_tiles

        overall_done += write_aperio_thumbnail_page(
            tif,
            slide,
            options,
            perf,
            iccprofile=None,
            overall_done=overall_done,
            overall_total=overall_total,
            thumbnail_max=layout["thumbnail_max"],
            progress_callback=progress_callback,
        )

        for level_index, ((width, height), level_tile_count) in enumerate(zip(desired_levels, level_tiles), start=1):
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("转换已取消")
            level_source, _ = _native_level_source_for_shape(slide, width, height)
            level_quality = _svs_pyramid_quality(slide, options, level_index - 1)
            level_progress = ProgressState()
            level_iter = iter_source_tiles(
                level_source,
                tile_size,
                chunk_size=strip_height,
                progress=level_progress,
                cancel_event=cancel_event,
                notify=lambda state, offset=overall_done: _emit_progress(
                    progress_callback,
                    "生成金字塔",
                    state.done,
                    state.total,
                    offset + state.done,
                    overall_total,
                ),
            )
            started = time.perf_counter()
            tif.write(
                data=iter(_EncodedTileIter(level_iter, level_quality, codec)),
                shape=(height, width, 3),
                description=build_aperio_pyramid_description(
                    slide, options, width, height, quality=level_quality
                ),
                software="",
                subfiletype=0,
                jpegtables=(
                    _get_jpeg_tables_for_shape(tile_size, tile_size, level_quality)
                    if codec == "jpeg_rgb"
                    else None
                ),
                **common,
            )
            perf["pyramid_sec"] += time.perf_counter() - started
            overall_done += level_tile_count

        associated = write_aperio_associated_images(
            tif,
            slide,
            options,
            perf,
            overall_done=overall_done,
            overall_total=overall_total,
            progress_callback=progress_callback,
        )
        perf.update({key: value for key, value in associated.items() if value is not None})

    perf["native_tile_mode"] = "svs_reencoded"
    completed = overall_done + associated_units + 1
    _advance_write_tail_progress(
        perf,
        progress_callback,
        step_done=1,
        step_total=write_tail_total,
        overall_done=completed,
        overall_total=overall_total,
    )
    return "svs-native-streaming", completed, overall_total


def _write_brightfield_ome_tiff_streaming(
    slide: IBLSlide,
    output_path: Path,
    options: ConvertOptions,
    perf: dict[str, Any],
    *,
    progress_callback=None,
    cancel_event=None,
) -> int:
    """Single-pass streaming Pyramidal OME-TIFF writer for RGB sources.

    Reads IBL data exactly once via *DensePyramidDrive*.  The main page is
    streamed directly into the BigTIFF while a downsampled buffer (2× or 4×,
    chosen adaptively) is accumulated in memory.  After the main page every
    further pyramid level is generated by cascading (resize → write) from
    that buffer, giving per-tile progress for every level.
    """
    if _native_output_ready(slide):
        return _write_native_ome_tiff(
            slide,
            output_path,
            options,
            perf,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )

    tifffile = _load_tiff_runtime(require_imagecodecs=True)
    tile_size = options.resolved_tile_size()
    parallel = _resolve_parallel_settings(options)
    quality = options.main_quality

    # -- pyramid shape plan --
    pyramid_shapes = (
        compute_pyramid_shapes(slide.width, slide.height)
        if options.generate_dense_pyramid
        else []
    )
    main_w, main_h = slide.width, slide.height
    main_tiles = tile_count(main_w, main_h, tile_size)
    associated_images: list[tuple[str, Image.Image]] = []
    for name in ("thumbnail", "macro", "label"):
        getter = getattr(slide, f"get_{name}_image", None)
        image = getter() if callable(getter) else None
        if image is not None:
            associated_images.append((name, image))

    # -- overall progress budget --
    pyramid_tiles = sum(
        tile_count(w, h, tile_size) for w, h in pyramid_shapes
    )
    overall_total = (
        1                       # "解析 IBL"
        + main_tiles
        + pyramid_tiles
        + len(associated_images)
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
        "dtype": np.uint8,
    }

    with tifffile.TiffWriter(str(output_path), bigtiff=True, ome=True) as tif:
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

        if drive.downsample_factor == 2:
            level_shapes = list(pyramid_shapes)
        else:
            buffer_height, buffer_width = drive.accumulation_buffer.shape[:2]
            level_shapes = [
                (width, height)
                for width, height in pyramid_shapes
                if width <= buffer_width + 1 and height <= buffer_height + 1
            ]

        encoder = _EncodedTileIter(drive, quality)
        mpp_x, mpp_y = _mpp_xy(slide)
        ome_metadata: dict[str, Any] = {
            "axes": "YXS",
            "Name": f"{slide.path.stem} RGB",
            "Description": build_generic_description(slide),
        }
        if mpp_x > 0:
            ome_metadata.update({"PhysicalSizeX": mpp_x, "PhysicalSizeXUnit": "µm"})
        if mpp_y > 0:
            ome_metadata.update({"PhysicalSizeY": mpp_y, "PhysicalSizeYUnit": "µm"})
        started = time.perf_counter()
        tif.write(
            data=iter(encoder),
            shape=(main_h, main_w, 3),
            description=None,
            software=f"{APP_NAME} {APP_VERSION}",
            subfiletype=0,
            subifds=len(level_shapes),
            metadata=ome_metadata,
            jpegtables=_get_jpeg_tables_for_shape(tile_size, tile_size, quality),
            **_resolution_kwargs_for_mpp(
                slide.base_info.mpp,
                mpp_x=mpp_x,
                mpp_y=mpp_y,
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
        buf = drive.accumulation_buffer
        buf_h, buf_w = buf.shape[:2]
        buf_image = Image.fromarray(buf)

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
                description=None,
                software=f"{APP_NAME} {APP_VERSION}",
                subfiletype=1,
                metadata=None,
                jpegtables=_get_jpeg_tables_for_shape(tile_size, tile_size, quality),
                **_resolution_kwargs_for_mpp(
                    slide.base_info.mpp,
                    mpp_x=mpp_x,
                    mpp_y=mpp_y,
                    base_width=main_w,
                    base_height=main_h,
                    level_width=lw,
                    level_height=lh,
                ),
                **common_kwargs,
            )
            perf["pyramid_sec"] += time.perf_counter() - started
            overall_offset += tile_count(lw, lh, tile_size)

        for name, image in associated_images:
            _write_native_page(
                tif,
                image,
                description=None,
                subfiletype=9 if name == "macro" else 1,
                metadata={
                    "axes": "YX" if image.mode in {"1", "L"} else "YXS",
                    "Name": name,
                },
            )
            overall_offset += 1

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

    Reads IBL data exactly once.  Each strip is simultaneously fed to the
    main page AND downsampled into a 4× accumulation buffer.  The remaining
    Aperio levels are generated from that buffer.  No pyvips/libvips or
    intermediate temporary files are needed.
    """
    if _native_output_ready(slide):
        return _write_svs_native_streaming(
            slide,
            output_path,
            options,
            perf,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )

    tifffile = _load_tiff_runtime(require_imagecodecs=True)
    layout = build_aperio_svs_layout(slide, options)
    desired_levels = layout["pyramid"]
    tile_size = _svs_tile_size(slide, options)
    parallel = _resolve_parallel_settings(options)
    bigtiff = bool(layout["bigtiff"])
    write_tail_total = _svs_write_tail_units(options)
    codec = _svs_codec(slide, options)
    quality = _svs_quality(slide, options)

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

    iccprofile = _svs_icc_profile(slide)
    common = _svs_common_kwargs(slide, options)
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

        main_encoder = _EncodedTileIter(main_drive, quality, codec)
        started = time.perf_counter()
        tif.write(
            data=iter(main_encoder),
            shape=(main_h, main_w, 3),
            description=build_aperio_description(slide, options),
            software="",
            iccprofile=iccprofile,
            subfiletype=0,
            jpegtables=(
                _get_jpeg_tables_for_shape(tile_size, tile_size, quality)
                if codec == "jpeg_rgb"
                else None
            ),
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
            iccprofile=None,
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
                # Cascaded level — down-sample from the previous level
                level_resized = prev_image.resize((lw, lh), resample=Image.Resampling.BILINEAR)
                prev_image = level_resized
                level_source = PILImageSource(level_resized)

            level_quality = _svs_pyramid_quality(slide, options, level_index)

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

            level_encoder = _EncodedTileIter(level_iter, level_quality, codec)
            started = time.perf_counter()
            tif.write(
                data=iter(level_encoder),
                shape=(lh, lw, 3),
                description=build_aperio_pyramid_description(
                    slide, options, lw, lh, quality=level_quality
                ),
                software="",
                subfiletype=0,
                jpegtables=(
                    _get_jpeg_tables_for_shape(tile_size, tile_size, level_quality)
                    if codec == "jpeg_rgb"
                    else None
                ),
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
    tile_size = _svs_tile_size(slide, options) if options.output_format == "svs" else options.resolved_tile_size()

    if tile_size <= 0 or tile_size % 16 != 0:
        raise RuntimeError("JPEG 压缩的 TIFF tile_size 必须是 16 的正整数倍")

    perf_tracker = PerfTracker()
    parallel = _resolve_parallel_settings(options)
    if options.output_format == "svs":
        level_dimensions = [(slide.width, slide.height), *_compute_svs_pyramid_shapes(slide, options)]
    elif options.output_format in {"ome_tiff", "fluorescence_svs", "afi"} and getattr(
        slide, "supports_native_pyramid", False
    ):
        level_dimensions = list(slide.level_dimensions)
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
        "native_path": False,
        "native_level_dimensions": None,
        "native_resource_dimensions": None,
        "native_tile_mode": None,
        "native_fallback_reason": None,
        "source_container": getattr(slide, "source_container", None),
        "source_version": getattr(slide, "source_version", None),
        "source_codec": getattr(slide, "source_codec", None),
        "source_bit_depth": getattr(slide, "source_bit_depth", None),
        "source_channel_count": getattr(slide, "source_channel_count", None),
        "source_axes": getattr(slide, "native_axes", None),
        "compatibility_level": getattr(slide, "compatibility_level", None),
        "diagnostic_code": getattr(slide, "diagnostic_code", None),
        "diagnostic_stage": getattr(slide, "diagnostic_stage", None),
        "svs_omitted_native_data": (
            _svs_omitted_native_data(slide) if options.output_format == "svs" else None
        ),
        "failure_stage": None,
        **parallel,
    }
    if getattr(slide, "native_resource_status", "") == "legacy_fallback":
        perf["native_fallback_reason"] = getattr(slide, "native_resource_reason", "") or None

    try:
        if options.output_format == "ome_tiff":
            from .ome_tiff_writer import write_ome_tiff

            pyramid_levels = write_ome_tiff(
                slide,
                output_path,
                options,
                perf,
                progress_callback=progress_callback,
                cancel_event=cancel_event,
            )
            perf["backend"] = "tifffile-ome"
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
        elif options.output_format == "fluorescence_svs":
            from .fluorescence_svs_writer import write_fluorescence_svs

            pyramid_levels = write_fluorescence_svs(
                slide,
                output_path,
                options,
                perf,
                progress_callback=progress_callback,
                cancel_event=cancel_event,
            )
        else:
            raise RuntimeError(f"不支持的输出格式: {options.output_format}")
    except Exception as exc:
        perf_tracker.sample()
        perf["peak_memory_mb"] = perf_tracker.peak_memory_mb
        perf["avg_cpu_percent"] = perf_tracker.average_cpu_percent()
        perf["failure_stage"] = perf.get("failure_stage") or perf.get("current_stage")
        perf["diagnostic_code"] = getattr(exc, "diagnostic_code", None) or perf.get("diagnostic_code")
        perf["diagnostic_stage"] = getattr(exc, "diagnostic_stage", None) or perf.get("diagnostic_stage")
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
    if options.output_format in {"ome_tiff", "fluorescence_svs"}:
        return pyramid_levels, perf
    return len(level_dimensions), perf


write_svs = write_image
