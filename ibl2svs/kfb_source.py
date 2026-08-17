from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import struct
import threading
from typing import BinaryIO

import numpy as np
from PIL import Image

from .models import BaseInfo


KFB_HEADER_MARKER = b"\xf1\x01\xee\xee"
KFB_MAGIC = b"KFB\x00"
KFBF_MAGIC = b"KFBF"
META_MARKER = b"\xff\x01\xee\xee"
IMAGE_RECORD_SIZE = 52
TILE_RECORD_SIZE = 64


class KfbFormatError(RuntimeError):
    pass


@dataclass(frozen=True)
class KfbTileRecord:
    index: int
    x: int
    y: int
    width: int
    height: int
    scale_value: float
    jpeg_offset: int
    jpeg_size: int


@dataclass(frozen=True)
class KfbAssociatedImageRecord:
    offset: int
    image_type: int
    width: int
    height: int
    channels: int
    jpeg_offset: int
    jpeg_size: int


def _u32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def _i32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<i", data, offset)[0]


def _f32(data: bytes, offset: int) -> float:
    return struct.unpack_from("<f", data, offset)[0]


def _u64(data: bytes, offset: int) -> int:
    return struct.unpack_from("<Q", data, offset)[0]


def _parse_metadata(data: bytes, offset: int, limit: int) -> dict[str, object]:
    if data[offset : offset + 4] != META_MARKER:
        return {}

    count = _u32(data, offset + 4)
    pos = offset + 8
    values: dict[str, object] = {}
    for _ in range(count):
        if pos + 8 > limit:
            break
        tag = _u32(data, pos)
        length = _u32(data, pos + 4)
        value = data[pos + 8 : pos + 8 + length]
        key = f"tag_{tag}"
        try:
            text = value.rstrip(b"\x00").decode("utf-8")
        except UnicodeDecodeError:
            text = ""
        if text and all(ch.isprintable() for ch in text):
            values[key] = text
        elif length == 4:
            values[key] = struct.unpack("<I", value)[0]
        else:
            values[key] = value.hex()
        pos += 8 + length
    return values


class KfbSlideSource:
    """JPEG-tile KFB source compatible with the existing writer API."""

    def __init__(self, path: str | Path, cache_size: int = 256):
        self.path = Path(path)
        self.cache_size = max(1, cache_size)
        self._fh = self.path.open("rb")
        self._lock = threading.RLock()
        self._tile_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._preview_image: Image.Image | None = None
        self._thumbnail_image: Image.Image | None = None
        try:
            self._file_size = self.path.stat().st_size
            self._metadata: dict[str, object] = {}
            self._associated_images: list[KfbAssociatedImageRecord] = []
            self._tiles: list[KfbTileRecord] = []
            self._tiles_by_scale: dict[float, list[KfbTileRecord]] = {}
            self._tile_maps_by_scale: dict[float, dict[tuple[int, int], KfbTileRecord]] = {}
            self._parse_header()
            self._associated_images = self._parse_associated_images()
            self._tiles = self._parse_tile_index()
            self._build_level_index()
            self.channels = 3
            self.base_info = BaseInfo(
                magic_no="KFB",
                version=str(self._metadata.get("tag_29", "")),
                focus_num=0,
                image_format=0,
                layer_size=len(self._tiles_by_scale),
                img_color=24,
                check_sum=0,
                ratio_step=2,
                max_layer_size=self.objective_power,
                slide_type=0,
                background_color=255,
                pixel_size_mm=(self.mpp / 1000.0) if self.mpp > 0 else 0.0,
                total_img_num=len(self._tiles),
                max_zoom_rate=self.objective_power,
                img_col=(self.width + self.tile_size - 1) // self.tile_size,
                img_row=(self.height + self.tile_size - 1) // self.tile_size,
                img_width=self.tile_size,
                img_height=self.tile_size,
                tile_width=self.tile_size,
                tile_height=self.tile_size,
                shrink_tile_num=0,
                total_img_width=self.width,
                total_img_height=self.height,
            )
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        fh = getattr(self, "_fh", None)
        if fh is not None:
            fh.close()
            self._fh = None

    def __enter__(self) -> "KfbSlideSource":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.height, self.width, self.channels)

    def _read_at(self, offset: int, size: int) -> bytes:
        with self._lock:
            self._fh.seek(offset)
            data = self._fh.read(size)
        if len(data) != size:
            raise KfbFormatError("KFB 文件数据被截断")
        return data

    def _parse_header(self) -> None:
        prefix = self._read_at(0, 0x60)
        if prefix[:4] != KFB_HEADER_MARKER or prefix[4:8] not in {KFB_MAGIC, KFBF_MAGIC}:
            raise KfbFormatError("不是可识别的 KFB 文件")

        self.container_variant = "kfbf" if prefix[4:8] == KFBF_MAGIC else "kfb"

        first_image_record_offset = _u32(prefix, 0x34)
        header_size = max(512, first_image_record_offset)
        data = self._read_at(0, header_size)

        self.width = _u32(data, 0x18)
        self.height = _u32(data, 0x14)
        self.tile_count = _u32(data, 0x10)
        self.objective_power = _u32(data, 0x1C)
        self.codec = data[0x20:0x24].rstrip(b"\x00").decode("ascii", errors="replace")
        self.raw_timestamp = _u32(data, 0x2C)
        self.first_image_record_offset = first_image_record_offset
        self.second_image_record_offset = _u32(data, 0x38)
        self.last_image_record_offset = _u32(data, 0x3C)
        self.tile_index_offset = _u32(data, 0x44)
        self.mpp = _f32(data, 0x4C)
        self.tile_size = _u32(data, 0x58)
        self._metadata = _parse_metadata(data, 0x5C, first_image_record_offset)

        if self.width <= 0 or self.height <= 0:
            raise KfbFormatError("KFB 主图尺寸无效")
        if self.tile_count <= 0:
            raise KfbFormatError("KFB 未包含瓦片索引")
        if self.tile_index_offset <= 0 or self.tile_index_offset >= self._file_size:
            raise KfbFormatError("KFB 瓦片索引偏移无效")
        if self.codec.upper() != "JPEG":
            raise KfbFormatError(f"暂不支持非 JPEG KFB: {self.codec}")
        if self.tile_size <= 0:
            raise KfbFormatError("KFB tile_size 无效")

    def _parse_associated_record(self, fh: BinaryIO, offset: int) -> KfbAssociatedImageRecord:
        fh.seek(offset)
        head = fh.read(IMAGE_RECORD_SIZE)
        if len(head) != IMAGE_RECORD_SIZE or head[0] != 0xF1 or head[2:4] != b"\xee\xee":
            raise KfbFormatError(f"KFB 附属图像记录无效: {offset}")

        image_type = head[1]
        if head[48:52] != bytes([0xFF, image_type, 0xEE, 0xEE]):
            raise KfbFormatError(f"KFB 附属图像结束标记无效: {offset}")

        return KfbAssociatedImageRecord(
            offset=offset,
            image_type=image_type,
            height=_u32(head, 8),
            width=_u32(head, 12),
            channels=_u32(head, 16),
            jpeg_size=_u32(head, 20),
            jpeg_offset=offset + IMAGE_RECORD_SIZE,
        )

    def _parse_associated_images(self) -> list[KfbAssociatedImageRecord]:
        records: list[KfbAssociatedImageRecord] = []
        seen: set[int] = set()
        offsets = [
            self.first_image_record_offset,
            self.second_image_record_offset,
            self.last_image_record_offset,
        ]
        with self._lock:
            for offset in offsets:
                if offset in seen or offset <= 0 or offset >= self._file_size:
                    continue
                seen.add(offset)
                records.append(self._parse_associated_record(self._fh, offset))
        return records

    def _parse_tile_index(self) -> list[KfbTileRecord]:
        records: list[KfbTileRecord] = []
        with self._lock:
            for index in range(self.tile_count):
                entry = self._read_at(
                    self.tile_index_offset + index * TILE_RECORD_SIZE,
                    TILE_RECORD_SIZE,
                )
                if entry[:4] != b"\xf1\x04\xee\xee" or entry[60:64] != b"\xff\x04\xee\xee":
                    raise KfbFormatError(f"KFB 瓦片记录标记无效: {index}")

                if self.container_variant == "kfbf":
                    offset_ref = _u32(entry, 36)
                    size_ref = _u32(entry, 44)
                    if offset_ref + 8 > self._file_size or size_ref + 8 > self._file_size:
                        raise KfbFormatError(f"KFBF 瓦片指针无效: {index}")
                    jpeg_offset = _u64(self._read_at(offset_ref, 8), 0)
                    jpeg_size = _u64(self._read_at(size_ref, 8), 0)
                    direct_size = _u32(entry, 32)
                    if jpeg_size != direct_size:
                        raise KfbFormatError(f"KFBF 瓦片长度不一致: {index}")
                else:
                    jpeg_size = _i32(entry, 32)
                    jpeg_offset = self.tile_index_offset + _i32(entry, 36)
                if jpeg_size <= 0 or jpeg_offset < 0 or jpeg_offset + jpeg_size > self._file_size:
                    raise KfbFormatError(f"{self.container_variant.upper()} 瓦片 JPEG 位置无效: {index}")

                records.append(
                    KfbTileRecord(
                        index=index,
                        x=_i32(entry, 4),
                        y=_i32(entry, 8),
                        width=_i32(entry, 12),
                        height=_i32(entry, 16),
                        scale_value=round(_f32(entry, 20), 6),
                        jpeg_size=jpeg_size,
                        jpeg_offset=jpeg_offset,
                    )
                )
        return records

    def _build_level_index(self) -> None:
        groups: dict[float, list[KfbTileRecord]] = defaultdict(list)
        for tile in self._tiles:
            groups[tile.scale_value].append(tile)
        if not groups:
            raise KfbFormatError("KFB 未包含可读取瓦片")

        self._tiles_by_scale = {
            scale: sorted(records, key=lambda tile: (tile.y, tile.x))
            for scale, records in groups.items()
        }
        self._tile_maps_by_scale = {
            scale: {(tile.x, tile.y): tile for tile in records}
            for scale, records in self._tiles_by_scale.items()
        }
        self.main_scale = max(self._tiles_by_scale)
        self.level_dimensions = {
            scale: (
                max(tile.x + tile.width for tile in records),
                max(tile.y + tile.height for tile in records),
            )
            for scale, records in self._tiles_by_scale.items()
        }

    def _decode_tile(self, record: KfbTileRecord) -> np.ndarray:
        cached = self._tile_cache.get(record.index)
        if cached is not None:
            self._tile_cache.move_to_end(record.index)
            return cached

        data = self._read_at(record.jpeg_offset, record.jpeg_size)
        with Image.open(BytesIO(data)) as image:
            arr = np.asarray(image.convert("RGB"), dtype=np.uint8)

        if arr.shape[0] != record.height or arr.shape[1] != record.width:
            normalized = np.full((record.height, record.width, 3), 255, dtype=np.uint8)
            h = min(record.height, arr.shape[0])
            w = min(record.width, arr.shape[1])
            normalized[:h, :w] = arr[:h, :w]
            arr = normalized
        else:
            arr = np.ascontiguousarray(arr)

        self._tile_cache[record.index] = arr
        self._tile_cache.move_to_end(record.index)
        while len(self._tile_cache) > self.cache_size:
            self._tile_cache.popitem(last=False)
        return arr

    def _read_region_from_scale(self, scale: float, x: int, y: int, width: int, height: int) -> np.ndarray:
        region = np.full((height, width, 3), 255, dtype=np.uint8)
        if width <= 0 or height <= 0:
            return region

        level_width, level_height = self.level_dimensions[scale]
        request_x0 = max(0, x)
        request_y0 = max(0, y)
        request_x1 = min(level_width, x + width)
        request_y1 = min(level_height, y + height)
        if request_x0 >= request_x1 or request_y0 >= request_y1:
            return region

        tile_map = self._tile_maps_by_scale[scale]
        first_x = (request_x0 // self.tile_size) * self.tile_size
        first_y = (request_y0 // self.tile_size) * self.tile_size
        for tile_y in range(first_y, request_y1, self.tile_size):
            for tile_x in range(first_x, request_x1, self.tile_size):
                record = tile_map.get((tile_x, tile_y))
                if record is None:
                    continue

                ix0 = max(request_x0, record.x)
                iy0 = max(request_y0, record.y)
                ix1 = min(request_x1, record.x + record.width)
                iy1 = min(request_y1, record.y + record.height)
                if ix0 >= ix1 or iy0 >= iy1:
                    continue

                tile = self._decode_tile(record)
                src_x0 = ix0 - record.x
                src_y0 = iy0 - record.y
                src_x1 = src_x0 + (ix1 - ix0)
                src_y1 = src_y0 + (iy1 - iy0)
                dst_x0 = ix0 - x
                dst_y0 = iy0 - y
                dst_x1 = dst_x0 + (ix1 - ix0)
                dst_y1 = dst_y0 + (iy1 - iy0)
                region[dst_y0:dst_y1, dst_x0:dst_x1] = tile[src_y0:src_y1, src_x0:src_x1]
        return region

    def read_region(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        *,
        decode_workers: int | None = None,
    ) -> np.ndarray:
        return self._read_region_from_scale(self.main_scale, x, y, width, height)

    def _associated_image(self, record: KfbAssociatedImageRecord) -> Image.Image:
        data = self._read_at(record.jpeg_offset, record.jpeg_size)
        with Image.open(BytesIO(data)) as image:
            return image.convert("RGB")

    def get_macro_image(self) -> Image.Image | None:
        type2_records = [record for record in self._associated_images if record.image_type == 2]
        candidates = [record for record in type2_records if max(record.width, record.height) >= 512]
        if not candidates:
            candidates = type2_records
        if not candidates:
            return None
        return self._associated_image(max(candidates, key=lambda record: record.width * record.height))

    def get_label_image(self) -> Image.Image | None:
        candidates = [record for record in self._associated_images if record.image_type == 3]
        if not candidates:
            return None
        return self._associated_image(max(candidates, key=lambda record: record.width * record.height))

    def _select_preview_scale(self, max_long_side: int = 2048) -> float | None:
        candidates: list[tuple[float, int]] = []
        for scale, (width, height) in self.level_dimensions.items():
            if scale == self.main_scale:
                continue
            long_side = max(width, height)
            if long_side <= max_long_side:
                candidates.append((scale, long_side))
        if candidates:
            return max(candidates, key=lambda item: item[1])[0]
        return None

    def _stitch_level(self, scale: float) -> Image.Image:
        width, height = self.level_dimensions[scale]
        canvas = Image.new("RGB", (width, height), "white")
        for record in self._tiles_by_scale[scale]:
            canvas.paste(Image.fromarray(self._decode_tile(record)), (record.x, record.y))
        return canvas

    def get_preview_image(self) -> Image.Image | None:
        if self._preview_image is None:
            scale = self._select_preview_scale()
            if scale is not None:
                self._preview_image = self._stitch_level(scale)
            else:
                self._preview_image = self.get_macro_image()
            if self._preview_image is None:
                sample_w = min(1024, self.width)
                sample_h = min(1024, self.height)
                self._preview_image = Image.fromarray(self.read_region(0, 0, sample_w, sample_h))
        return self._preview_image.copy()

    def get_thumbnail_image(self) -> Image.Image | None:
        if self._thumbnail_image is None:
            self._thumbnail_image = self.get_preview_image()
        return self._thumbnail_image.copy() if self._thumbnail_image is not None else None

    def get_scan_metadata(self) -> dict[str, object]:
        metadata = dict(self._metadata)
        metadata["scanTime"] = self.raw_timestamp
        if "tag_29" in metadata:
            metadata["deviceNo"] = metadata["tag_29"]
        return metadata
