from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import io
import os
import re
from pathlib import Path
import struct
from typing import Any

import numpy as np
from PIL import Image


JPEG_COMPRESSION_IDS = {6, 7, 33007, 34892}
PALETTE_PHOTOMETRIC = 3
TIFF_HEADER_SIZE = 8
IFD_SCAN_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class TiffBaseInfo:
    mpp: float = 0.0
    max_zoom_rate: int = 0
    background_color: int = 255


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
            self._preview_image: Image.Image | None = None
            self.base_info = TiffBaseInfo(
                mpp=self._extract_mpp(),
                max_zoom_rate=self._extract_app_mag(),
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
        if self._page.dtype != np.uint8:
            raise RuntimeError("仅支持 uint8 TIFF/SVS 输入")
        if int(getattr(self._page, "planarconfig", 1) or 1) != 1:
            raise RuntimeError("暂不支持 planar-separated TIFF/SVS 输入")
        if int(getattr(self._page, "photometric", 0) or 0) == PALETTE_PHOTOMETRIC:
            raise RuntimeError("暂不支持 palette TIFF 输入")
        samples = int(getattr(self._page, "samplesperpixel", 1) or 1)
        if samples not in (1, 3, 4):
            raise RuntimeError("仅支持 grayscale、RGB 或 RGBA TIFF/SVS 输入")

    def _extract_mpp(self) -> float:
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
        array = self._normalize_array(page.asarray())
        return Image.fromarray(array)

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
        for page in self._tif.pages[1:]:
            desc = (page.description or "").lower()
            if "label " in desc or "macro " in desc:
                continue
            if ("thumbnail " in desc or "native thumbnail" in desc) and not getattr(page, "is_tiled", False):
                image = self._page_to_image(page)
                if image is not None:
                    return image.convert("RGB")
        for page in self._tif.pages[1:]:
            desc = (page.description or "").lower()
            if "label " in desc or "macro " in desc:
                continue
            if not getattr(page, "is_tiled", False):
                image = self._page_to_image(page)
                if image is not None:
                    return image.convert("RGB")
        return self.get_preview_image()

    def get_label_image(self) -> Image.Image | None:
        for page in self._tif.pages[1:]:
            desc = (page.description or "").lower()
            if "label " not in desc:
                continue
            image = self._page_to_image(page)
            if image is not None:
                return image.convert("RGB")
        return None

    def get_macro_image(self) -> Image.Image | None:
        for page in self._tif.pages[1:]:
            desc = (page.description or "").lower()
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
