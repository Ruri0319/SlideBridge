from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import io
import os
import re
from pathlib import Path
import struct
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

from .native_jpeg import jpeg_dimensions


JPEG_COMPRESSION_IDS = {6, 7, 33007, 34892}
PALETTE_PHOTOMETRIC = 3
TIFF_HEADER_SIZE = 8
IFD_SCAN_BYTES = 64 * 1024 * 1024


def _ome_length(value: str | None, unit: str | None, target_unit: str) -> float:
    if not value:
        return 0.0
    normalized = (unit or target_unit).strip().replace("μ", "µ").lower()
    meters = {
        "m": 1.0,
        "cm": 1e-2,
        "mm": 1e-3,
        "µm": 1e-6,
        "um": 1e-6,
        "nm": 1e-9,
        "pm": 1e-12,
    }
    target = target_unit.replace("μ", "µ").lower()
    if normalized not in meters or target not in meters:
        raise RuntimeError(f"不支持的 OME 长度单位: {unit}")
    return float(value) * meters[normalized] / meters[target]


def _ome_time(value: str | None, unit: str | None) -> float | None:
    if value is None or value == "":
        return None
    normalized = (unit or "s").strip().replace("μ", "µ").lower()
    seconds = {
        "s": 1.0,
        "ms": 1e-3,
        "µs": 1e-6,
        "us": 1e-6,
        "ns": 1e-9,
        "min": 60.0,
    }
    if normalized not in seconds:
        raise RuntimeError(f"不支持的 OME 时间单位: {unit}")
    return float(value) * seconds[normalized]


@dataclass(frozen=True)
class TiffBaseInfo:
    mpp: float = 0.0
    max_zoom_rate: int = 0
    background_color: int = 255


@dataclass(frozen=True)
class TiffNativeLevel:
    dimensions: tuple[int, int]
    records: tuple = ()


class TiffSlideSource:
    """Tile/strip based TIFF/SVS source compatible with the writer API."""

    def __init__(self, path: str | Path, cache_size: int = 64):
        import tifffile

        self.path = Path(path)
        self.cache_size = max(1, cache_size)
        self._shifted_reader = None
        self._tif = self._open_tiff(tifffile)
        try:
            self._page = self._select_main_page()
            self.width = int(self._page.imagewidth)
            self.height = int(self._page.imagelength)
            self.channels = 3
            self._segment_cache: OrderedDict[int, np.ndarray] = OrderedDict()
            self._plane_segment_cache: OrderedDict[tuple[int, int], np.ndarray] = OrderedDict()
            self._preview_image: Image.Image | None = None
            self._configure_source_semantics()
            self.base_info = TiffBaseInfo(
                mpp=self._extract_mpp(),
                max_zoom_rate=self._extract_app_mag(),
                background_color=0 if self.modality == "fluorescence" else 255,
            )
            self._validate_main_page()
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        tif = getattr(self, "_tif", None)
        if tif is not None:
            tif.close()
            self._tif = None
        reader = getattr(self, "_shifted_reader", None)
        if reader is not None:
            reader.close()
            self._shifted_reader = None

    def __enter__(self) -> "TiffSlideSource":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.height, self.width, self.channels)

    def _open_tiff(self, tifffile):
        try:
            return tifffile.TiffFile(str(self.path))
        except Exception as exc:
            recovery = _recover_shifted_ifd(self.path)
            if recovery is None:
                raise
            threshold, delta = recovery
            self._shifted_reader = _ShiftedIfdReader(self.path, threshold, delta)
            try:
                return tifffile.TiffFile(self._shifted_reader)
            except Exception:
                self._shifted_reader.close()
                self._shifted_reader = None
                raise exc

    def _select_main_page(self):
        if not self._tif.pages:
            raise RuntimeError("TIFF/SVS 文件没有可读取页面")
        return self._tif.pages[0]

    def _validate_main_page(self) -> None:
        if self._page.dtype not in (np.dtype(np.uint8), np.dtype(np.uint16)):
            raise RuntimeError("仅支持 uint8 TIFF/SVS 输入")
        if self._page.dtype == np.uint16 and self.modality != "fluorescence":
            raise RuntimeError("仅支持 uint8；只有明确荧光 OME-TIFF 支持 uint16 输入")
        if int(getattr(self._page, "planarconfig", 1) or 1) != 1:
            raise RuntimeError("暂不支持 planar-separated TIFF/SVS 输入")
        if int(getattr(self._page, "photometric", 0) or 0) == PALETTE_PHOTOMETRIC:
            raise RuntimeError("暂不支持 palette TIFF 输入")
        samples = int(getattr(self._page, "samplesperpixel", 1) or 1)
        if samples not in (1, 3, 4):
            raise RuntimeError("仅支持 grayscale、RGB 或 RGBA TIFF/SVS 输入")

    @staticmethod
    def _ome_color(value: str | None) -> tuple[int, int, int]:
        if not value:
            return (255, 255, 255)
        packed = int(value) & 0xFFFFFFFF
        return ((packed >> 24) & 0xFF, (packed >> 16) & 0xFF, (packed >> 8) & 0xFF)

    @staticmethod
    def _description_property(description: str, name: str) -> str | None:
        match = re.search(rf"(?:^|\|){re.escape(name)}\s*=\s*([^|\r\n]+)", description, re.IGNORECASE)
        return match.group(1).strip() if match else None

    def _configure_source_semantics(self) -> None:
        samples = int(getattr(self._page, "samplesperpixel", 1) or 1)
        bit_depth = int(np.dtype(self._page.dtype).itemsize * 8)
        compression = int(getattr(self._page, "compression", 1) or 1)
        self.modality = "brightfield"
        self.native_fields = (0,)
        self.native_channel_count = 0
        self.native_z_count = 1
        self.native_t_count = 1
        self.source_channel_count = samples
        self.source_bit_depth = bit_depth
        self.channel_metadata: list[dict[str, Any]] = []
        self.supports_native_planes = False
        self.supports_plane_jpeg_passthrough = False
        self.source_container = "svs" if self.path.suffix.lower() == ".svs" else "tiff"
        description = self._page.description or ""
        version_match = re.match(r"Aperio Image Library v([^\r\n]+)", description.strip())
        self.aperio_library_version = version_match.group(1).strip() if version_match else None
        self.source_version = self.aperio_library_version
        self.source_codec = "JPEG" if compression in JPEG_COMPRESSION_IDS else str(self._page.compression.name)
        profile_tag = self._page.tags.get(34675)
        profile_value = getattr(profile_tag, "value", None) if profile_tag is not None else None
        self.icc_profile = bytes(profile_value) if isinstance(profile_value, bytes) else None
        self.icc_profile_name = self._description_property(description, "ICC Profile") or (
            "embedded" if self.icc_profile is not None else "sRGB"
        )
        self.native_axes = "YXS" if samples > 1 else "YX"
        self.compatibility_level = "static_unverified"
        self.mpp_x = 0.0
        self.mpp_y = 0.0
        self.tile_size = int(getattr(self._page, "tilewidth", 0) or 256)
        self._field_series = [self._tif.series[0]]
        self.svs_level_dimensions: list[tuple[int, int]] = []
        if self.source_container == "svs":
            self.svs_level_dimensions = [
                (int(page.imagewidth), int(page.imagelength))
                for page in self._tif.pages
                if getattr(page, "is_tiled", False)
                and int(getattr(page, "subfiletype", 0) or 0) == 0
            ]
            if not self.svs_level_dimensions or self.svs_level_dimensions[0] != (self.width, self.height):
                self.svs_level_dimensions = []
        self._associated_series = {
            str(getattr(series, "name", "") or "").strip().lower(): series
            for series in self._tif.series
            if str(getattr(series, "name", "") or "").strip()
        }

        if self._tif.ome_metadata:
            self._configure_ome_semantics(self._tif.ome_metadata)
        else:
            self._configure_aperio_semantics(description)

        if (
            self.source_container == "ome_tiff" and self.modality != "brightfield"
        ) or self.modality == "fluorescence":
            first_series = self._field_series[0]
            self.level_dimensions = [
                (
                    int(level.shape[level.axes.index("X")]),
                    int(level.shape[level.axes.index("Y")]),
                )
                for level in first_series.levels
            ]
            self.levels = [TiffNativeLevel(dimensions) for dimensions in self.level_dimensions]
            self.supports_native_pyramid = len(self.levels) > 1
            self.supports_native_planes = True
            self.supports_plane_jpeg_passthrough = (
                self.source_bit_depth == 8
                and all(
                    int(page.compression) in JPEG_COMPRESSION_IDS
                    and int(getattr(page, "samplesperpixel", 1) or 1) == 1
                    and bool(getattr(page, "is_tiled", False))
                    and int(getattr(page, "tilewidth", 0) or 0) == self.tile_size
                    and int(getattr(page, "tilelength", 0) or 0) == self.tile_size
                    and getattr(page, "jpegtables", None) is None
                    for series in self._field_series
                    for level in series.levels
                    for page in level.pages
                )
            )

    def _configure_ome_semantics(self, xml: str) -> None:
        root = ET.fromstring(xml)
        images = [element for element in root if element.tag.rsplit("}", 1)[-1] == "Image"]
        associated_names = {"thumbnail", "macro", "label"}
        main_images = [
            image for image in images
            if str(image.attrib.get("Name", "")).strip().lower() not in associated_names
        ]
        main_series = [
            series for series in self._tif.series
            if str(getattr(series, "name", "") or "").strip().lower() not in associated_names
        ]
        if not main_images or not main_series:
            return
        self.source_container = "ome_tiff"
        image = main_images[0]
        pixels = next((child for child in image if child.tag.rsplit("}", 1)[-1] == "Pixels"), None)
        if pixels is None:
            return
        channels = [child for child in pixels if child.tag.rsplit("}", 1)[-1] == "Channel"]
        samples_per_pixel = max((int(channel.attrib.get("SamplesPerPixel", "1")) for channel in channels), default=1)
        description = next(
            (child.text or "" for child in image if child.tag.rsplit("}", 1)[-1] == "Description"),
            "",
        )
        self.mpp_x = _ome_length(
            pixels.attrib.get("PhysicalSizeX"),
            pixels.attrib.get("PhysicalSizeXUnit"),
            "µm",
        )
        self.mpp_y = _ome_length(
            pixels.attrib.get("PhysicalSizeY"),
            pixels.attrib.get("PhysicalSizeYUnit"),
            "µm",
        )
        explicit_fluorescence = "modality=fluorescence" in description.lower()
        has_fluor = any(str(channel.attrib.get("Fluor", "")).strip() for channel in channels)
        if samples_per_pixel >= 3 or int(getattr(self._page, "samplesperpixel", 1) or 1) >= 3:
            self.modality = "brightfield"
            self.native_channel_count = 0
            return
        self.modality = "fluorescence" if explicit_fluorescence or has_fluor else "unknown"
        self.native_channel_count = int(pixels.attrib.get("SizeC", "1"))
        self.native_z_count = int(pixels.attrib.get("SizeZ", "1"))
        self.native_t_count = int(pixels.attrib.get("SizeT", "1"))
        self.source_channel_count = self.native_channel_count
        self.source_bit_depth = int(pixels.attrib.get("SignificantBits", str(self.source_bit_depth)))
        self.native_axes = "TZCYX"
        self._field_series = main_series[: len(main_images)]
        self.native_fields = tuple(range(len(self._field_series)))
        exposures: dict[int, float] = {}
        for plane in (child for child in pixels if child.tag.rsplit("}", 1)[-1] == "Plane"):
            if "ExposureTime" in plane.attrib:
                exposures.setdefault(
                    int(plane.attrib.get("TheC", "0")),
                    _ome_time(plane.attrib["ExposureTime"], plane.attrib.get("ExposureTimeUnit")),
                )
        metadata: list[dict[str, Any]] = []
        for index in range(self.native_channel_count):
            channel = channels[index] if index < len(channels) else None
            attrs = channel.attrib if channel is not None else {}
            name = str(attrs.get("Name", "")).strip()
            fluor = str(attrs.get("Fluor", "")).strip()
            known = bool(fluor or (name and not re.fullmatch(r"Channel\s+\d+", name, re.IGNORECASE)))
            metadata.append(
                {
                    "name": name,
                    "fluor": fluor or None,
                    "color": self._ome_color(attrs.get("Color")),
                    "excitation_nm": (
                        _ome_length(
                            attrs["ExcitationWavelength"],
                            attrs.get("ExcitationWavelengthUnit"),
                            "nm",
                        )
                        if "ExcitationWavelength" in attrs
                        else None
                    ),
                    "emission_nm": (
                        _ome_length(
                            attrs["EmissionWavelength"],
                            attrs.get("EmissionWavelengthUnit"),
                            "nm",
                        )
                        if "EmissionWavelength" in attrs
                        else None
                    ),
                    "exposure": exposures.get(index),
                    "identity_source": "source_metadata" if known else "unknown",
                }
            )
        self.channel_metadata = metadata

    def _configure_aperio_semantics(self, description: str) -> None:
        dye = self._description_property(description, "Dye")
        monochrome = "JPEG/Monochrome" in description
        if not dye or not monochrome:
            return
        self.modality = "fluorescence"
        self.source_container = "fluorescence_svs"
        self.native_channel_count = 1
        self.source_channel_count = 1
        self.native_axes = "TZCYX"
        display_color = int(self._description_property(description, "DisplayColor") or "16777215")
        known = re.fullmatch(r"C\d+", dye, re.IGNORECASE) is None
        self.channel_metadata = [
            {
                "name": dye if known else "",
                "fluor": dye if known else None,
                "color": ((display_color >> 16) & 0xFF, (display_color >> 8) & 0xFF, display_color & 0xFF),
                "excitation_nm": self._optional_description_float(description, "Excitation Wavelength"),
                "emission_nm": self._optional_description_float(description, "Emission Wavelength"),
                "exposure": self._optional_description_float(description, "Exposure Time"),
                "identity_source": "source_metadata" if known else "unknown",
            }
        ]

    def _optional_description_float(self, description: str, name: str) -> float | None:
        value = self._description_property(description, name)
        return float(value) if value is not None else None

    def _extract_mpp(self) -> float:
        if self.mpp_x > 0 and self.mpp_y > 0:
            return (self.mpp_x + self.mpp_y) / 2.0
        if self.mpp_x > 0:
            return self.mpp_x
        if self.mpp_y > 0:
            return self.mpp_y
        description = self._page.description or ""
        match = re.search(r"\bMPP\s*=\s*([0-9]+(?:\.[0-9]+)?)", description, re.IGNORECASE)
        if match:
            return float(match.group(1))

        x_res = self._fraction_tag_value("XResolution")
        unit_tag = self._page.tags.get("ResolutionUnit")
        unit = unit_tag.value if unit_tag is not None else None
        if x_res and x_res > 0:
            if unit in (2, "INCH"):
                return 25400.0 / x_res
            if unit in (3, "CENTIMETER"):
                return 10000.0 / x_res
        return 0.0

    def _extract_app_mag(self) -> int:
        description = self._page.description or ""
        match = re.search(
            r"\b(?:AppMag|ObjectivePower|openslide\.objective-power|Magnification)\s*=\s*([0-9]+(?:\.[0-9]+)?)",
            description,
            re.IGNORECASE,
        )
        if not match:
            return 0
        return int(round(float(match.group(1))))

    def _fraction_tag_value(self, tag_name: str) -> float | None:
        tag = self._page.tags.get(tag_name)
        if tag is None:
            return None
        value = tag.value
        if isinstance(value, tuple) and len(value) == 2:
            numerator, denominator = value
            if denominator:
                return float(numerator) / float(denominator)
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _segment_geometry(self, index: int) -> tuple[int, int, int, int]:
        if self._page.is_tiled:
            tile_w = int(self._page.tilewidth)
            tile_h = int(self._page.tilelength)
            tiles_x = (self.width + tile_w - 1) // tile_w
            x0 = (index % tiles_x) * tile_w
            y0 = (index // tiles_x) * tile_h
            return x0, y0, min(tile_w, self.width - x0), min(tile_h, self.height - y0)

        rows_per_strip = int(getattr(self._page, "rowsperstrip", 0) or self.height)
        y0 = index * rows_per_strip
        return 0, y0, self.width, min(rows_per_strip, self.height - y0)

    def _intersecting_segment_indices(self, x: int, y: int, width: int, height: int) -> list[int]:
        if width <= 0 or height <= 0:
            return []
        x1 = min(self.width, x + width)
        y1 = min(self.height, y + height)
        if x >= x1 or y >= y1:
            return []

        if self._page.is_tiled:
            tile_w = int(self._page.tilewidth)
            tile_h = int(self._page.tilelength)
            tiles_x = (self.width + tile_w - 1) // tile_w
            first_col = max(0, x // tile_w)
            last_col = max(0, (x1 - 1) // tile_w)
            first_row = max(0, y // tile_h)
            last_row = max(0, (y1 - 1) // tile_h)
            return [
                row * tiles_x + col
                for row in range(first_row, last_row + 1)
                for col in range(first_col, last_col + 1)
            ]

        rows_per_strip = int(getattr(self._page, "rowsperstrip", 0) or self.height)
        first = max(0, y // rows_per_strip)
        last = max(0, (y1 - 1) // rows_per_strip)
        return list(range(first, last + 1))

    def _read_segment_bytes(self, index: int) -> bytes:
        offset = self._resolved_data_offset(index)
        byte_count = int(self._page.databytecounts[index])
        reader = getattr(self, "_shifted_reader", None)
        if reader is not None:
            return reader.read_actual(offset, byte_count)
        fh = self._tif.filehandle
        with fh.lock:
            fh.seek(offset)
            return fh.read(byte_count)

    def _resolved_data_offset(self, index: int) -> int:
        offset = int(self._page.dataoffsets[index])
        reader = getattr(self, "_shifted_reader", None)
        if reader is None or int(self._page.compression) not in JPEG_COMPRESSION_IDS:
            return offset

        delta = reader.delta
        shifted = offset + delta
        if shifted <= offset or shifted >= reader.size:
            return offset
        prefix = reader.read_actual(offset, 3)
        if prefix.startswith(b"\xff\xd8\xff"):
            return offset
        shifted_prefix = reader.read_actual(shifted, 3)
        if shifted_prefix.startswith(b"\xff\xd8\xff"):
            return shifted
        return offset

    def _decode_args(self) -> dict[str, Any]:
        keyframe = self._page.keyframe
        args: dict[str, Any] = {"_fullsize": bool(keyframe.is_tiled)}
        if int(keyframe.compression) in JPEG_COMPRESSION_IDS:
            args["jpegtables"] = self._page.jpegtables
            args["jpegheader"] = keyframe.jpegheader
        return args

    def _decode_segment(self, index: int) -> np.ndarray:
        cached = self._segment_cache.get(index)
        if cached is not None:
            self._segment_cache.move_to_end(index)
            return cached

        data = self._read_segment_bytes(index)
        decoded, _indices, _shape = self._page.keyframe.decode(data, index, **self._decode_args())
        if decoded is None:
            x0, y0, seg_w, seg_h = self._segment_geometry(index)
            decoded = np.full((seg_h, seg_w, 3), 255, dtype=np.uint8)
        segment = self._normalize_array(decoded)
        self._segment_cache[index] = segment
        self._segment_cache.move_to_end(index)
        while len(self._segment_cache) > self.cache_size:
            self._segment_cache.popitem(last=False)
        return segment

    @staticmethod
    def _page_dimensions(page) -> tuple[int, int]:
        keyframe = page.keyframe
        return int(keyframe.imagewidth), int(keyframe.imagelength)

    @staticmethod
    def _page_segment_geometry(page, index: int) -> tuple[int, int, int, int]:
        width, height = TiffSlideSource._page_dimensions(page)
        keyframe = page.keyframe
        if keyframe.is_tiled:
            tile_w = int(keyframe.tilewidth)
            tile_h = int(keyframe.tilelength)
            tiles_x = (width + tile_w - 1) // tile_w
            x0 = (index % tiles_x) * tile_w
            y0 = (index // tiles_x) * tile_h
            return x0, y0, min(tile_w, width - x0), min(tile_h, height - y0)
        rows_per_strip = int(getattr(keyframe, "rowsperstrip", 0) or height)
        y0 = index * rows_per_strip
        return 0, y0, width, min(rows_per_strip, height - y0)

    @staticmethod
    def _page_intersecting_segments(page, x: int, y: int, width: int, height: int) -> list[int]:
        page_width, page_height = TiffSlideSource._page_dimensions(page)
        x1 = min(page_width, x + width)
        y1 = min(page_height, y + height)
        if width <= 0 or height <= 0 or x >= x1 or y >= y1:
            return []
        keyframe = page.keyframe
        if keyframe.is_tiled:
            tile_w = int(keyframe.tilewidth)
            tile_h = int(keyframe.tilelength)
            tiles_x = (page_width + tile_w - 1) // tile_w
            first_col = max(0, x // tile_w)
            last_col = max(0, (x1 - 1) // tile_w)
            first_row = max(0, y // tile_h)
            last_row = max(0, (y1 - 1) // tile_h)
            return [
                row * tiles_x + column
                for row in range(first_row, last_row + 1)
                for column in range(first_col, last_col + 1)
            ]
        rows_per_strip = int(getattr(keyframe, "rowsperstrip", 0) or page_height)
        first = max(0, y // rows_per_strip)
        last = max(0, (y1 - 1) // rows_per_strip)
        return list(range(first, last + 1))

    def _resolved_page_data_offset(self, page, index: int) -> int:
        offset = int(page.dataoffsets[index])
        reader = getattr(self, "_shifted_reader", None)
        if reader is None or int(page.keyframe.compression) not in JPEG_COMPRESSION_IDS:
            return offset
        shifted = offset + reader.delta
        if shifted <= offset or shifted >= reader.size:
            return offset
        if reader.read_actual(offset, 3).startswith(b"\xff\xd8\xff"):
            return offset
        if reader.read_actual(shifted, 3).startswith(b"\xff\xd8\xff"):
            return shifted
        return offset

    def _read_page_segment_bytes(self, page, index: int) -> bytes:
        offset = self._resolved_page_data_offset(page, index)
        byte_count = int(page.databytecounts[index])
        reader = getattr(self, "_shifted_reader", None)
        if reader is not None:
            return reader.read_actual(offset, byte_count)
        with self._tif.filehandle.lock:
            self._tif.filehandle.seek(offset)
            return self._tif.filehandle.read(byte_count)

    @staticmethod
    def _normalize_plane_array(array: np.ndarray) -> np.ndarray:
        plane = np.asarray(array)
        while plane.ndim > 2 and plane.shape[0] == 1:
            plane = plane[0]
        if plane.ndim == 3 and plane.shape[-1] == 1:
            plane = plane[..., 0]
        if plane.ndim != 2 or plane.dtype not in (np.dtype(np.uint8), np.dtype(np.uint16)):
            raise RuntimeError("暂不支持该 OME-TIFF 原生平面布局")
        return np.ascontiguousarray(plane)

    def _decode_page_segment(self, page, index: int) -> np.ndarray:
        cache_key = (int(page.dataoffsets[0]), index)
        cached = self._plane_segment_cache.get(cache_key)
        if cached is not None:
            self._plane_segment_cache.move_to_end(cache_key)
            return cached
        keyframe = page.keyframe
        args: dict[str, Any] = {"_fullsize": bool(keyframe.is_tiled)}
        if int(keyframe.compression) in JPEG_COMPRESSION_IDS:
            args["jpegtables"] = page.jpegtables
            args["jpegheader"] = keyframe.jpegheader
        decoded, _indices, _shape = keyframe.decode(
            self._read_page_segment_bytes(page, index),
            index,
            **args,
        )
        if decoded is None:
            _x, _y, width, height = self._page_segment_geometry(page, index)
            decoded = np.zeros((height, width), dtype=keyframe.dtype)
        segment = self._normalize_plane_array(decoded)
        self._plane_segment_cache[cache_key] = segment
        self._plane_segment_cache.move_to_end(cache_key)
        while len(self._plane_segment_cache) > self.cache_size:
            self._plane_segment_cache.popitem(last=False)
        return segment

    @staticmethod
    def _level_plane_page(level, channel_index: int, z_index: int, t_index: int):
        coordinates = {"C": channel_index, "Z": z_index, "T": t_index}
        page_index = 0
        for axis, size in zip(level.axes, level.shape):
            if axis in {"Y", "X", "S"}:
                continue
            coordinate = coordinates.get(axis, 0)
            if coordinate < 0 or coordinate >= int(size):
                raise IndexError(f"OME-TIFF {axis} 轴索引越界")
            page_index = page_index * int(size) + coordinate
        if page_index >= len(level.pages):
            raise IndexError("OME-TIFF 平面索引越界")
        return level.pages[page_index]

    def _native_plane_page(
        self,
        level_index: int,
        field_index: int,
        channel_index: int,
        z_index: int,
        t_index: int,
    ):
        if field_index < 0 or field_index >= len(self._field_series):
            raise IndexError("OME-TIFF Field 索引越界")
        series = self._field_series[field_index]
        if level_index < 0 or level_index >= len(series.levels):
            raise IndexError("OME-TIFF 金字塔层索引越界")
        if channel_index < 0 or channel_index >= self.native_channel_count:
            raise IndexError("OME-TIFF 通道索引越界")
        if z_index < 0 or z_index >= self.native_z_count:
            raise IndexError("OME-TIFF Z 索引越界")
        if t_index < 0 or t_index >= self.native_t_count:
            raise IndexError("OME-TIFF T 索引越界")
        return self._level_plane_page(series.levels[level_index], channel_index, z_index, t_index)

    def read_level_field_plane_region(
        self,
        level_index: int,
        field_index: int,
        channel_index: int,
        z_index: int,
        t_index: int,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> np.ndarray:
        page = self._native_plane_page(
            level_index,
            field_index,
            channel_index,
            z_index,
            t_index,
        )
        page_width, page_height = self._page_dimensions(page)
        region = np.zeros((height, width), dtype=page.keyframe.dtype)
        if width <= 0 or height <= 0:
            return region
        request_x0 = max(0, x)
        request_y0 = max(0, y)
        request_x1 = min(page_width, x + width)
        request_y1 = min(page_height, y + height)
        if request_x0 >= request_x1 or request_y0 >= request_y1:
            return region
        for index in self._page_intersecting_segments(
            page,
            request_x0,
            request_y0,
            request_x1 - request_x0,
            request_y1 - request_y0,
        ):
            segment_x, segment_y, segment_width, segment_height = self._page_segment_geometry(page, index)
            ix0 = max(request_x0, segment_x)
            iy0 = max(request_y0, segment_y)
            ix1 = min(request_x1, segment_x + segment_width)
            iy1 = min(request_y1, segment_y + segment_height)
            if ix0 >= ix1 or iy0 >= iy1:
                continue
            segment = self._decode_page_segment(page, index)
            source_x = ix0 - segment_x
            source_y = iy0 - segment_y
            target_x = ix0 - x
            target_y = iy0 - y
            copy_width = ix1 - ix0
            copy_height = iy1 - iy0
            region[target_y : target_y + copy_height, target_x : target_x + copy_width] = segment[
                source_y : source_y + copy_height,
                source_x : source_x + copy_width,
            ]
        return region

    def read_level_plane_region(
        self,
        level_index: int,
        channel_index: int,
        z_index: int,
        t_index: int,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> np.ndarray:
        return self.read_level_field_plane_region(
            level_index,
            0,
            channel_index,
            z_index,
            t_index,
            x,
            y,
            width,
            height,
        )

    def iter_native_level_plane_jpegs(
        self,
        level_index: int,
        channel_index: int,
        z_index: int,
        t_index: int,
        field_index: int | None = None,
    ):
        selected_field = 0 if field_index is None else field_index
        page = self._native_plane_page(
            level_index,
            selected_field,
            channel_index,
            z_index,
            t_index,
        )
        keyframe = page.keyframe
        if (
            not keyframe.is_tiled
            or int(keyframe.compression) not in JPEG_COMPRESSION_IDS
            or int(getattr(keyframe, "samplesperpixel", 1) or 1) != 1
            or keyframe.dtype != np.dtype(np.uint8)
            or getattr(keyframe, "jpegtables", None) is not None
        ):
            raise RuntimeError("当前 TIFF 平面不能直接重封装为 JPEG 瓦片")
        for index in range(len(page.dataoffsets)):
            payload = self._read_page_segment_bytes(page, index)
            jpeg_dimensions(payload)
            yield payload

    @staticmethod
    def _normalize_array(array: np.ndarray) -> np.ndarray:
        arr = np.asarray(array)
        if arr.dtype != np.uint8:
            raise RuntimeError("仅支持 uint8 TIFF/SVS 输入")
        if arr.ndim == 4:
            if arr.shape[0] != 1:
                raise RuntimeError("暂不支持 3D 或多平面 TIFF/SVS 输入")
            arr = arr[0]
        if arr.ndim == 2:
            return np.repeat(arr[:, :, None], 3, axis=2)
        if arr.ndim != 3:
            raise RuntimeError("暂不支持该 TIFF/SVS 数组布局")
        if arr.shape[2] == 1:
            return np.repeat(arr, 3, axis=2)
        if arr.shape[2] == 3:
            return np.ascontiguousarray(arr)
        if arr.shape[2] == 4:
            rgb = arr[:, :, :3].astype(np.float32)
            alpha = arr[:, :, 3:4].astype(np.float32) / 255.0
            composited = rgb * alpha + 255.0 * (1.0 - alpha)
            return np.ascontiguousarray(np.clip(composited, 0, 255).astype(np.uint8))
        raise RuntimeError("仅支持 grayscale、RGB 或 RGBA TIFF/SVS 输入")

    def read_region(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        *,
        decode_workers: int | None = None,
    ) -> np.ndarray:
        region = np.full((height, width, 3), 255, dtype=np.uint8)
        if width <= 0 or height <= 0:
            return region

        request_x0 = max(0, x)
        request_y0 = max(0, y)
        request_x1 = min(self.width, x + width)
        request_y1 = min(self.height, y + height)
        if request_x0 >= request_x1 or request_y0 >= request_y1:
            return region

        for index in self._intersecting_segment_indices(request_x0, request_y0, request_x1 - request_x0, request_y1 - request_y0):
            seg_x, seg_y, seg_w, seg_h = self._segment_geometry(index)
            ix0 = max(request_x0, seg_x)
            iy0 = max(request_y0, seg_y)
            ix1 = min(request_x1, seg_x + seg_w)
            iy1 = min(request_y1, seg_y + seg_h)
            if ix0 >= ix1 or iy0 >= iy1:
                continue

            segment = self._decode_segment(index)
            src_x0 = ix0 - seg_x
            src_y0 = iy0 - seg_y
            src_x1 = src_x0 + (ix1 - ix0)
            src_y1 = src_y0 + (iy1 - iy0)
            dst_x0 = ix0 - x
            dst_y0 = iy0 - y
            dst_x1 = dst_x0 + (ix1 - ix0)
            dst_y1 = dst_y0 + (iy1 - iy0)
            region[dst_y0:dst_y1, dst_x0:dst_x1] = segment[src_y0:src_y1, src_x0:src_x1]
        return region

    def _page_to_image(self, page, max_pixels: int = 16_000_000) -> Image.Image | None:
        pixels = int(page.imagewidth) * int(page.imagelength)
        if pixels > max_pixels:
            return None
        array = np.asarray(page.asarray())
        while array.ndim > 3 and array.shape[0] == 1:
            array = array[0]
        if array.ndim == 3 and array.shape[-1] == 1:
            array = array[..., 0]
        if array.ndim == 2 and array.dtype in (np.dtype(np.uint8), np.dtype(np.uint16)):
            return Image.fromarray(array)
        return Image.fromarray(self._normalize_array(array))

    def _named_associated_image(self, name: str) -> Image.Image | None:
        series = self._associated_series.get(name)
        if series is None or not series.pages:
            return None
        return self._page_to_image(series.pages[0])

    def _smallest_tiled_page(self):
        candidates = [
            page for page in self._tif.pages[1:]
            if getattr(page, "is_tiled", False)
            and int(getattr(page, "samplesperpixel", 1) or 1) in (1, 3, 4)
            and getattr(page, "dtype", None) == np.uint8
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda page: int(page.imagewidth) * int(page.imagelength))

    def get_preview_image(self) -> Image.Image | None:
        if self._preview_image is not None:
            return self._preview_image.copy()

        page = self._smallest_tiled_page()
        if page is not None:
            image = self._page_to_image(page)
            if image is not None:
                self._preview_image = image.convert("RGB")
                return self._preview_image.copy()

        sample_w = min(1024, self.width)
        sample_h = min(1024, self.height)
        self._preview_image = Image.fromarray(self.read_region(0, 0, sample_w, sample_h))
        return self._preview_image.copy()

    def get_thumbnail_image(self) -> Image.Image | None:
        named = self._named_associated_image("thumbnail")
        if named is not None:
            return named
        if self.source_container == "ome_tiff":
            return None
        for page in self._tif.pages[1:]:
            desc = (getattr(page, "description", "") or "").lower()
            if "label " in desc or "macro " in desc:
                continue
            if ("thumbnail " in desc or "native thumbnail" in desc) and not getattr(page, "is_tiled", False):
                image = self._page_to_image(page)
                if image is not None:
                    return image.convert("RGB")
        for page in self._tif.pages[1:]:
            desc = (getattr(page, "description", "") or "").lower()
            if "label " in desc or "macro " in desc:
                continue
            if not getattr(page, "is_tiled", False):
                image = self._page_to_image(page)
                if image is not None:
                    return image.convert("RGB")
        return self.get_preview_image()

    def get_label_image(self) -> Image.Image | None:
        named = self._named_associated_image("label")
        if named is not None:
            return named
        if self.source_container == "ome_tiff":
            return None
        for page in self._tif.pages[1:]:
            desc = (getattr(page, "description", "") or "").lower()
            if "label " not in desc:
                continue
            image = self._page_to_image(page)
            if image is not None:
                return image.convert("RGB")
        return None

    def get_macro_image(self) -> Image.Image | None:
        named = self._named_associated_image("macro")
        if named is not None:
            return named
        if self.source_container == "ome_tiff":
            return None
        for page in self._tif.pages[1:]:
            desc = (getattr(page, "description", "") or "").lower()
            if "macro " not in desc and "native macro" not in desc:
                continue
            image = self._page_to_image(page)
            if image is not None:
                return image.convert("RGB")
        return None

    def get_scan_metadata(self) -> dict[str, object]:
        return {}


class _ShiftedIfdReader(io.RawIOBase):
    """Read a TIFF whose IFD area is shifted relative to stored offsets."""

    def __init__(self, path: Path, threshold: int, delta: int):
        self._fh = path.open("rb")
        self._threshold = threshold
        self._delta = delta
        self._virtual_pos = 0
        self.name = str(path)
        self._fh.seek(0, os.SEEK_END)
        self.size = self._fh.tell()
        self._fh.seek(0)

    @property
    def delta(self) -> int:
        return self._delta

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._virtual_pos

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_SET:
            self._virtual_pos = offset
        elif whence == os.SEEK_CUR:
            self._virtual_pos += offset
        elif whence == os.SEEK_END:
            self._fh.seek(0, os.SEEK_END)
            self._virtual_pos = self._fh.tell() + offset
        return self._virtual_pos

    def _actual_pos(self, virtual_pos: int) -> int:
        if virtual_pos >= self._threshold:
            return virtual_pos + self._delta
        return virtual_pos

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            self._fh.seek(self._actual_pos(self._virtual_pos))
            data = self._fh.read()
            self._virtual_pos += len(data)
            return data

        chunks: list[bytes] = []
        remaining = size
        while remaining > 0:
            if self._virtual_pos < self._threshold < self._virtual_pos + remaining:
                chunk_size = self._threshold - self._virtual_pos
            else:
                chunk_size = remaining
            self._fh.seek(self._actual_pos(self._virtual_pos))
            data = self._fh.read(chunk_size)
            if not data:
                break
            chunks.append(data)
            self._virtual_pos += len(data)
            remaining -= len(data)
            if len(data) < chunk_size:
                break
        return b"".join(chunks)

    def readinto(self, buffer) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)

    def read_actual(self, offset: int, size: int) -> bytes:
        current = self._fh.tell()
        try:
            self._fh.seek(offset)
            return self._fh.read(size)
        finally:
            self._fh.seek(current)

    def close(self) -> None:
        try:
            self._fh.close()
        finally:
            super().close()


def _recover_shifted_ifd(path: Path) -> tuple[int, int] | None:
    size = path.stat().st_size
    with path.open("rb") as fh:
        header = fh.read(TIFF_HEADER_SIZE)
        if len(header) != TIFF_HEADER_SIZE or header[:2] != b"II" or header[2:4] != b"\x2a\x00":
            return None
        stored_ifd = struct.unpack("<I", header[4:8])[0]
        if stored_ifd <= 0 or stored_ifd >= size:
            return None
        scan_size = min(IFD_SCAN_BYTES, max(0, size - TIFF_HEADER_SIZE))
        scan_start = size - scan_size
        fh.seek(scan_start)
        data = fh.read(scan_size)

    for rel in range(0, max(0, len(data) - 2)):
        actual_ifd = scan_start + rel
        if actual_ifd <= stored_ifd:
            continue
        tag_count = int.from_bytes(data[rel : rel + 2], "little")
        if not 8 <= tag_count <= 200:
            continue
        if _valid_shifted_ifd_candidate(data, rel, tag_count, stored_ifd, actual_ifd, size):
            return stored_ifd, actual_ifd - stored_ifd
    return None


def _valid_shifted_ifd_candidate(
    data: bytes,
    rel: int,
    tag_count: int,
    stored_ifd: int,
    actual_ifd: int,
    file_size: int,
) -> bool:
    end = rel + 2 + tag_count * 12 + 4
    if end > len(data):
        return False

    tags: dict[int, tuple[int, int, int]] = {}
    previous_tag = -1
    for index in range(tag_count):
        entry = data[rel + 2 + index * 12 : rel + 2 + (index + 1) * 12]
        tag, value_type, count, value = struct.unpack("<HHII", entry)
        if tag < previous_tag or value_type == 0 or value_type > 18:
            return False
        previous_tag = tag
        tags[tag] = (value_type, count, value)

    required = {256, 257, 270}
    if not required.issubset(tags):
        return False
    if 273 not in tags and 324 not in tags:
        return False

    offset_tag = 324 if 324 in tags else 273
    _value_type, count, table_offset = tags[offset_tag]
    if count <= 0:
        return False
    delta = actual_ifd - stored_ifd
    table_rel = table_offset + delta - (actual_ifd - rel)
    if table_rel < 0 or table_rel + min(count, 8) * 4 > len(data):
        return False

    sample_count = min(count, 8)
    offsets = struct.unpack("<" + "I" * sample_count, data[table_rel : table_rel + sample_count * 4])
    return all(TIFF_HEADER_SIZE <= offset < file_size for offset in offsets)
