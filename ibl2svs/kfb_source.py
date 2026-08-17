from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass, replace
from io import BytesIO
from pathlib import Path
import struct
import threading
from typing import Iterator
import zlib

import numpy as np
from PIL import Image

from .models import BaseInfo


KFB_HEADER_MARKERS = {b"\xf1\x01\xee\xee", b"\xf1\x02\xee\xee"}
CLASSIC_MAGICS = {b"KFB\x00": "kfb", b"KFBL": "kfbl", b"KFBF": "kfbf"}
META_MARKER = b"\xff\x01\xee\xee"
DATA_MARKER = b"\xf1\x04\xee\xee"
DATA_END_MARKER = b"\xff\x04\xee\xee"
IMAGE_RECORD_SIZE = 52
SUPPORTED_CLASSIC_VERSIONS = {round(1.0 + index / 10.0, 1) for index in range(14)}
JPEG_SOF_MARKERS = {
    0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
    0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF,
}
KFBX_VALUE_SIZES = {0: 1, 1: 1, 2: 4, 3: 8, 4: 4}


class KfbFormatError(RuntimeError):
    """KFB family parse error carrying machine-readable diagnostics."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "unsupported_layout",
        stage: str = "parse",
        source_container: str | None = None,
        source_version: str | None = None,
    ):
        super().__init__(message)
        self.diagnostic_code = code
        self.diagnostic_stage = stage
        self.source_container = source_container
        self.source_version = source_version


@dataclass(frozen=True)
class NativeTile:
    index: int
    level_index: int
    field_index: int
    channel_index: int
    z_index: int
    t_index: int
    x: int
    y: int
    width: int
    height: int
    scale_value: float
    codec: str
    bit_depth: int
    offset: int
    length: int
    payload: bytes | None = None


@dataclass(frozen=True)
class NativeLevel:
    index: int
    scale_value: float
    width: int
    height: int
    tile_size: int
    records: tuple[NativeTile, ...]

    @property
    def dimensions(self) -> tuple[int, int]:
        return self.width, self.height

    @property
    def columns(self) -> int:
        return (self.width + self.tile_size - 1) // self.tile_size

    @property
    def rows(self) -> int:
        return (self.height + self.tile_size - 1) // self.tile_size


@dataclass(frozen=True)
class KfbAssociatedImageRecord:
    name: str
    offset: int
    image_type: int
    width: int
    height: int
    channels: int
    payload_offset: int
    payload_size: int


@dataclass(frozen=True)
class KfbaDataItem:
    item_id: int
    data_type: int
    length: int
    value: int


@dataclass(frozen=True)
class KfbxAttribute:
    attribute_id: int
    value_type: int
    count: int
    payload: bytes


def _u32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def _i32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<i", data, offset)[0]


def _f32(data: bytes, offset: int) -> float:
    return struct.unpack_from("<f", data, offset)[0]


def _u64(data: bytes, offset: int) -> int:
    return struct.unpack_from("<Q", data, offset)[0]


def _parse_metadata(data: bytes, offset: int, limit: int) -> dict[str, object]:
    items = _parse_header_items(data, offset, limit)
    values: dict[str, object] = {}
    for tag, value in items.items():
        key = f"tag_{tag}"
        try:
            decoded = value.rstrip(b"\x00").decode("utf-8")
        except UnicodeDecodeError:
            decoded = ""
        if decoded and all(ch.isprintable() for ch in decoded):
            values[key] = decoded
        elif len(value) == 4:
            values[key] = struct.unpack("<I", value)[0]
        else:
            values[key] = value.hex()
    return values


def _parse_header_items(data: bytes, offset: int, limit: int) -> dict[int, bytes]:
    if data[offset : offset + 4] != META_MARKER:
        return {}
    count = _u32(data, offset + 4)
    pos = offset + 8
    values: dict[int, bytes] = {}
    for _ in range(count):
        if pos + 8 > limit:
            raise KfbFormatError("KFB 元数据目录被截断", code="metadata_truncated", stage="metadata")
        tag = _u32(data, pos)
        length = _u32(data, pos + 4)
        if pos + 8 + length > limit:
            raise KfbFormatError("KFB 元数据值被截断", code="metadata_truncated", stage="metadata")
        value = data[pos + 8 : pos + 8 + length]
        values[tag] = value
        pos += 8 + length
    return values


def parse_kfba_data_block(data: bytes, offset: int = 0) -> tuple[KfbaDataItem, ...]:
    """Parse the fixed CKFBADataBlock item table used by KFBA containers."""

    if offset < 0 or offset + 8 > len(data):
        raise KfbFormatError("KFBA DataBlock 被截断", code="kfba_block_truncated", stage="kfba_index")
    count = _u64(data, offset)
    end = offset + 8 + count * 20
    if end > len(data):
        raise KfbFormatError("KFBA DataItem 表被截断", code="kfba_item_truncated", stage="kfba_index")
    return tuple(
        KfbaDataItem(
            item_id=_u32(data, offset + 8 + index * 20),
            data_type=_u32(data, offset + 12 + index * 20),
            length=_u32(data, offset + 16 + index * 20),
            value=_u64(data, offset + 20 + index * 20),
        )
        for index in range(count)
    )


def parse_kfbx_attributes(data: bytes, offset: int = 0) -> tuple[KfbxAttribute, ...]:
    """Parse KAC/KFBX attributes using count × vendor value-type size."""

    attributes: list[KfbxAttribute] = []
    cursor = offset
    while cursor < len(data):
        if cursor + 8 > len(data):
            raise KfbFormatError("KFBX Attribute 头被截断", code="kfbx_attribute_truncated", stage="kfbx_index")
        attribute_id, value_type, count = struct.unpack_from("<HHI", data, cursor)
        cursor += 8
        value_size = KFBX_VALUE_SIZES.get(value_type)
        if value_size is None:
            raise KfbFormatError(
                f"KFBX Attribute value_type 不支持: {value_type}",
                code="kfbx_value_type_unsupported",
                stage="kfbx_index",
            )
        payload_size = count * value_size
        if cursor + payload_size > len(data):
            raise KfbFormatError("KFBX Attribute payload 被截断", code="kfbx_attribute_truncated", stage="kfbx_index")
        attributes.append(KfbxAttribute(attribute_id, value_type, count, data[cursor : cursor + payload_size]))
        cursor += payload_size
    return tuple(attributes)


def _kfbx_values(attribute: KfbxAttribute) -> tuple[int | float, ...]:
    if attribute.value_type in {0, 1}:
        return tuple(attribute.payload)
    format_char = {2: "i", 3: "Q", 4: "f"}[attribute.value_type]
    return tuple(value[0] for value in struct.iter_unpack(f"<{format_char}", attribute.payload))


def _single_kfbx_value(attributes: dict[int, KfbxAttribute], attribute_id: int) -> int | float:
    attribute = attributes.get(attribute_id)
    if attribute is None or attribute.count != 1:
        raise KfbFormatError(
            f"KFBX 缺少单值属性: {attribute_id}",
            code="kfbx_required_metadata_missing",
            stage="kfbx_metadata",
            source_container="kfbx",
        )
    return _kfbx_values(attribute)[0]


def _codec_from_payload(payload: bytes) -> str:
    if payload.startswith(b"\xff\xd8"):
        marker, _precision, _components = _jpeg_layout(payload)
        return "SOF3" if marker == 0xC3 else "JPEG"
    if payload.startswith(b"\xff\x0a") or payload.startswith(b"\x00\x00\x00\x0cJXL \r\n\x87\n"):
        return "JXL"
    raise KfbFormatError(
        "KFBX 瓦片压缩格式无法由 payload 签名确认",
        code="unsupported_codec",
        stage="codec",
        source_container="kfbx",
    )


def apply_vendor_lut(
    samples: np.ndarray,
    *,
    contrast: float = 1.0,
    brightness: float = 0.0,
    channel_offset: float = 0.0,
    gamma: float = 1.0,
    black: float = 0.0,
    white: float = 255.0,
) -> np.ndarray:
    """Apply the display LUT recovered from the vendor viewer."""

    if white <= black:
        raise ValueError("white must be greater than black")
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    normalized = np.clip((np.asarray(samples, dtype=np.float64) - black) / (white - black), 0.0, 1.0)
    value = np.clip(
        (normalized - 0.5) * max(float(contrast), 0.0)
        + 0.5
        + float(brightness)
        + float(channel_offset),
        0.0,
        1.0,
    )
    return np.rint(255.0 * np.power(np.abs(value), 1.0 / float(gamma))).astype(np.uint8)


def compose_vendor_channels(
    channels: list[np.ndarray],
    colors: list[tuple[int, int, int]],
    *,
    mode: int,
) -> np.ndarray:
    """Compose channels using MultiWeight or MultiMax semantics."""

    if not channels or len(channels) != len(colors):
        raise ValueError("channels and colors must be non-empty and have equal length")
    weighted = [
        np.asarray(channel, dtype=np.float32)[..., None]
        * (np.asarray(color, dtype=np.float32) / 255.0)
        for channel, color in zip(channels, colors)
    ]
    result = np.maximum.reduce(weighted) if mode == 2 else np.add.reduce(weighted)
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)


def _jpeg_layout(payload: bytes) -> tuple[int, int, int]:
    """Return JPEG SOF marker, precision, and component count."""

    if not payload.startswith(b"\xff\xd8"):
        return 0, 0, 0
    cursor = 2
    while cursor + 4 <= len(payload):
        if payload[cursor] != 0xFF:
            cursor += 1
            continue
        while cursor < len(payload) and payload[cursor] == 0xFF:
            cursor += 1
        if cursor >= len(payload):
            break
        marker = payload[cursor]
        cursor += 1
        if marker in {0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
            continue
        if cursor + 2 > len(payload):
            break
        length = struct.unpack_from(">H", payload, cursor)[0]
        if length < 2 or cursor + length > len(payload):
            break
        if marker in JPEG_SOF_MARKERS and length >= 8:
            return marker, payload[cursor + 2], payload[cursor + 7]
        cursor += length
    return 0, 0, 0


class KfbSlideSource:
    """Version-aware clean-room reader for the KFB container family."""

    def __init__(self, path: str | Path, cache_size: int = 256):
        self.path = Path(path)
        self.cache_size = max(1, int(cache_size))
        self._fh = self.path.open("rb")
        self._lock = threading.RLock()
        self._tile_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._associated_cache: dict[str, Image.Image] = {}
        self._metadata: dict[str, object] = {}
        self._header_items: dict[int, bytes] = {}
        self._associated_images: dict[str, KfbAssociatedImageRecord] = {}
        self._tiles: list[NativeTile] = []
        self._levels: list[NativeLevel] = []
        self._tile_maps: list[dict[tuple[int, int, int, int, int, int], NativeTile]] = []
        self.native_fields = (0,)
        self.native_channel_count = 1
        self.native_z_count = 1
        self.native_t_count = 1
        self.default_field_index = 0
        self.default_channel_index = 0
        self.channel_metadata: list[dict[str, object]] = []
        self.modality = "unknown"
        self.supports_native_planes = False
        self.composite_mode = 1
        self.native_axes = "YXS"
        self.source_channel_count = 3
        self.source_bit_depth = 8
        self.source_codec = ""
        self.supports_plane_jpeg_passthrough = False
        self.compatibility_level = "static_unverified"
        self.diagnostic_code: str | None = None
        self.diagnostic_stage: str | None = None
        try:
            self._file_size = self.path.stat().st_size
            self._dispatch_container()
            self._finalize_source_semantics()
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        with self._lock:
            self._tile_cache.clear()
            self._associated_cache.clear()
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
        return self.height, self.width, 3

    @property
    def levels(self) -> tuple[NativeLevel, ...]:
        return tuple(self._levels)

    @property
    def supports_native_pyramid(self) -> bool:
        return bool(self._levels)

    @property
    def native_output_ready(self) -> bool:
        return bool(self._levels)

    @property
    def native_resource_status(self) -> str:
        return "native" if self._associated_images else "unavailable"

    @property
    def native_resource_reason(self) -> str:
        return ""

    @property
    def native_resource_dimensions(self) -> dict[str, tuple[int, int] | None]:
        return {
            name: (
                (self._associated_images[name].width, self._associated_images[name].height)
                if name in self._associated_images
                else None
            )
            for name in ("thumbnail", "macro", "label")
        }

    @property
    def native_tile_mode(self) -> str:
        return "jpeg_passthrough" if self._can_passthrough_jpeg else "lossless_decoded"

    @property
    def mpp_x(self) -> float:
        return self.mpp

    @property
    def mpp_y(self) -> float:
        return self.mpp

    @property
    def source_axes(self) -> str:
        return self.native_axes

    def _error(self, message: str, *, code: str, stage: str) -> KfbFormatError:
        return KfbFormatError(
            message,
            code=code,
            stage=stage,
            source_container=getattr(self, "source_container", None),
            source_version=getattr(self, "source_version", None),
        )

    def _read_at(self, offset: int, size: int) -> bytes:
        if offset < 0 or size < 0 or offset + size > self._file_size:
            raise self._error("KFB 文件数据位置越界", code="data_out_of_bounds", stage="payload")
        with self._lock:
            fh = self._fh
            if fh is None:
                raise self._error("KFB 文件已关闭", code="source_closed", stage="payload")
            fh.seek(offset)
            data = fh.read(size)
        if len(data) != size:
            raise self._error("KFB 文件数据被截断", code="data_truncated", stage="payload")
        return data

    def _dispatch_container(self) -> None:
        if self._file_size < 0x80:
            raise self._error("江丰文件小于最小头部", code="header_truncated", stage="detect")
        prefix = self._read_at(0, min(self._file_size, 0x80))
        marker = prefix[:4]
        magic = prefix[4:8]
        if marker in KFB_HEADER_MARKERS and magic in CLASSIC_MAGICS:
            self.source_container = CLASSIC_MAGICS[magic]
            self.container_variant = self.source_container
            self._parse_classic()
            return
        if marker in KFB_HEADER_MARKERS and (magic == b"KFBA" or b"KFBA" in prefix):
            self.source_container = "kfba"
            self.container_variant = "kfba"
            self.compatibility_level = "static_unverified"
            self._parse_kfba()
            return
        if prefix.startswith((b"KAC\x00", b"KFBX")):
            self.source_container = "kfbx"
            self.source_version = "unknown"
            self.compatibility_level = "static_unverified"
            self._parse_kfbx()
            return
        raise self._error("不是可识别的江丰容器", code="unknown_container", stage="detect")

    def _finalize_source_semantics(self) -> None:
        packed_samples = int(self.source_channel_count)
        has_channel_directory = bool(self.channel_metadata)
        tile_channels = {tile.channel_index for tile in self._tiles}
        independent_planes = (
            self.native_channel_count >= 1
            and tile_channels == set(range(self.native_channel_count))
            and (self.native_channel_count > 1 or packed_samples == 1)
        )
        if has_channel_directory and independent_planes:
            self.modality = "fluorescence"
            self.supports_native_planes = True
            self.native_axes = "TZCYX"
        elif packed_samples >= 3:
            self.modality = "brightfield"
            self.supports_native_planes = False
            self.native_channel_count = 1
            self.native_axes = "YXS"
        else:
            self.modality = "unknown"
            self.supports_native_planes = False
            self.native_axes = "YX"
        self.base_info = self._build_base_info()

    def _parse_kfbx(self) -> None:
        first_attribute_offset = _u64(self._read_at(68, 8), 0)
        if first_attribute_offset < 76 or first_attribute_offset >= self._file_size:
            raise self._error(
                "KFBX 首个 Attribute 偏移无效",
                code="kfbx_attribute_offset_invalid",
                stage="kfbx_index",
            )
        top_level = parse_kfbx_attributes(
            self._read_at(first_attribute_offset, self._file_size - first_attribute_offset)
        )
        global_attributes = {
            attribute.attribute_id: attribute
            for attribute in top_level
            if attribute.attribute_id != 123
        }
        level_attributes = [attribute for attribute in top_level if attribute.attribute_id == 123]
        if not level_attributes:
            raise self._error(
                "KFBX 未包含金字塔层属性",
                code="kfbx_required_metadata_missing",
                stage="kfbx_metadata",
            )

        tile_width = int(_single_kfbx_value(global_attributes, 1))
        tile_height = int(_single_kfbx_value(global_attributes, 2))
        compression_type = int(_single_kfbx_value(global_attributes, 2123))
        if tile_width <= 0 or tile_height <= 0 or tile_width != tile_height:
            raise self._error(
                "KFBX 当前仅支持方形正尺寸原生瓦片",
                code="kfbx_tile_geometry_unsupported",
                stage="kfbx_metadata",
            )

        parsed_levels: list[tuple[float, int, int, list[tuple[int, int, int, int]]]] = []
        for level_attribute in level_attributes:
            nested = {
                attribute.attribute_id: attribute
                for attribute in parse_kfbx_attributes(level_attribute.payload)
            }
            width = int(_single_kfbx_value(nested, 3))
            height = int(_single_kfbx_value(nested, 4))
            magnification = float(_single_kfbx_value(nested, 5))
            offsets_attribute = nested.get(123)
            lengths_attribute = nested.get(135)
            coordinates_attribute = nested.get(134)
            if offsets_attribute is None or lengths_attribute is None or coordinates_attribute is None:
                raise self._error(
                    "KFBX 层缺少 offset/length/coordinate 属性",
                    code="kfbx_required_metadata_missing",
                    stage="kfbx_index",
                )
            offsets = tuple(int(value) for value in _kfbx_values(offsets_attribute))
            lengths = tuple(int(value) for value in _kfbx_values(lengths_attribute))
            coordinates = tuple(int(value) for value in _kfbx_values(coordinates_attribute))
            if len(offsets) != len(lengths) or len(coordinates) != len(offsets) * 2:
                raise self._error(
                    "KFBX offset/length/coordinate 数量不一致",
                    code="kfbx_index_count_mismatch",
                    stage="kfbx_index",
                )
            if width <= 0 or height <= 0 or magnification <= 0:
                raise self._error(
                    "KFBX 层几何或倍率无效",
                    code="tile_geometry_invalid",
                    stage="kfbx_index",
                )
            entries: list[tuple[int, int, int, int]] = []
            for index, (offset, length) in enumerate(zip(offsets, lengths)):
                x, y = coordinates[index * 2 : index * 2 + 2]
                if length <= 0 or offset < 0 or offset + length > self._file_size:
                    raise self._error(
                        f"KFBX 瓦片位置无效: {index}",
                        code="tile_payload_invalid",
                        stage="kfbx_index",
                    )
                if x < 0 or y < 0 or x >= width or y >= height:
                    raise self._error(
                        f"KFBX 瓦片坐标无效: {index}",
                        code="tile_geometry_invalid",
                        stage="kfbx_index",
                    )
                entries.append((x, y, offset, length))
            parsed_levels.append((magnification, width, height, entries))

        parsed_levels.sort(key=lambda item: item[0], reverse=True)
        first_entry = parsed_levels[0][3][0]
        first_payload = self._read_at(first_entry[2], first_entry[3])
        source_codec = _codec_from_payload(first_payload)
        tiles: list[NativeTile] = []
        levels: list[NativeLevel] = []
        tile_maps: list[dict[tuple[int, int, int, int, int, int], NativeTile]] = []
        for level_index, (magnification, width, height, entries) in enumerate(parsed_levels):
            records: list[NativeTile] = []
            for x, y, offset, length in entries:
                codec = _codec_from_payload(self._read_at(offset, length))
                if codec != source_codec:
                    raise self._error(
                        "KFBX 同一文件包含未声明的混合 codec",
                        code="mixed_codec_unsupported",
                        stage="codec",
                    )
                tile = NativeTile(
                    index=len(tiles),
                    level_index=level_index,
                    field_index=0,
                    channel_index=0,
                    z_index=0,
                    t_index=0,
                    x=x,
                    y=y,
                    width=min(tile_width, width - x),
                    height=min(tile_height, height - y),
                    scale_value=magnification,
                    codec=source_codec,
                    bit_depth=8,
                    offset=offset,
                    length=length,
                )
                tiles.append(tile)
                records.append(tile)
            records.sort(key=lambda tile: (tile.y, tile.x))
            level = NativeLevel(level_index, magnification, width, height, tile_width, tuple(records))
            levels.append(level)
            tile_maps.append({(0, tile.x, tile.y, 0, 0, 0): tile for tile in records})

        self.header_marker = self._read_at(0, 4)
        self.record_size = 0
        self.tile_size = tile_width
        self.tile_count = len(tiles)
        self._tiles = tiles
        self._levels = levels
        self._tile_maps = tile_maps
        self.level_dimensions = [level.dimensions for level in levels]
        self.level_grids = [(level.columns, level.rows) for level in levels]
        self.main_scale = levels[0].scale_value
        self.width, self.height = levels[0].dimensions
        self.objective_power = max(1, int(round(self.main_scale)))
        self.source_codec = source_codec
        self.codec = source_codec
        self.raw_timestamp = 0
        self.mpp = 0.0
        self.channels = 3
        self.source_channel_count = 3
        self.source_bit_depth = 8
        self.native_axes = "YXS"
        self._metadata = {"compressionType": compression_type}
        self._inspect_first_tile()
        self.native_axes = "YX" if self.source_channel_count == 1 else "YXS"
        self.compatibility_level = "static_unverified"
        self.base_info = self._build_base_info()

    def _parse_kfba(self) -> None:
        prefix = self._read_at(0, 0x80)
        version_value = round(float(_f32(prefix, 0x0C)), 1)
        self.source_version = f"{version_value:.1f}"
        if version_value != 1.4:
            raise self._error(
                f"当前静态确认的 KFBA Header 版本仅为 1.4，收到: {self.source_version}",
                code="unsupported_version",
                stage="header",
            )

        self.header_marker = prefix[:4]
        self.record_size = 424
        self.block_count = _u32(prefix, 0x10)
        self.height = _u32(prefix, 0x14)
        self.width = _u32(prefix, 0x18)
        self.objective_power = _u32(prefix, 0x1C)
        self.source_codec = prefix[0x20:0x28].split(b"\x00", 1)[0].decode("ascii", errors="replace").upper()
        self.codec = self.source_codec
        self.raw_timestamp = _u32(prefix, 0x2C)
        self.first_image_record_offset = _u32(prefix, 0x34)
        self.second_image_record_offset = _u32(prefix, 0x38)
        self.last_image_record_offset = _u64(prefix, 0x3C)
        self.tile_index_offset = _u64(prefix, 0x44)
        self.mpp = float(_f32(prefix, 0x4C))
        self.tile_size = _u32(prefix, 0x58)

        if self.width <= 0 or self.height <= 0 or self.block_count <= 0 or self.tile_size <= 0:
            raise self._error("KFBA 主图或瓦片参数无效", code="invalid_header_dimensions", stage="header")
        if self.tile_index_offset < 0x80 or self.tile_index_offset + self.block_count * self.record_size > self._file_size:
            raise self._error("KFBA 索引偏移或长度无效", code="invalid_index_offset", stage="header")
        if self.source_codec not in {"JPEG", "JXL", "LJPEG", "SOF3"}:
            raise self._error(
                f"不支持的 KFBA codec: {self.source_codec or '<empty>'}",
                code="unsupported_codec",
                stage="header",
            )

        metadata_end = min(
            [
                value
                for value in (
                    self.first_image_record_offset,
                    self.second_image_record_offset,
                    self.last_image_record_offset,
                    self.tile_index_offset,
                )
                if value > 0x5C
            ],
            default=self.tile_index_offset,
        )
        header = self._read_at(0, metadata_end)
        self._header_items = _parse_header_items(header, 0x5C, metadata_end)
        self._metadata = _parse_metadata(header, 0x5C, metadata_end)
        self.native_channel_count = self._parse_kfba_channel_metadata()
        self._parse_associated_images()

        required_ids = {0, 1, 2, 3, 4, 5, 6, 7, 10}
        tiles: list[NativeTile] = []
        for block_index in range(self.block_count):
            block_offset = self.tile_index_offset + block_index * self.record_size
            items = parse_kfba_data_block(self._read_at(block_offset, self.record_size))
            item_map = {item.item_id: item for item in items}
            if len(item_map) != len(items):
                raise self._error(
                    f"KFBA DataBlock {block_index} 包含重复 DataItem",
                    code="kfba_duplicate_item",
                    stage="kfba_index",
                )
            missing = sorted(required_ids.difference(item_map))
            if missing:
                raise self._error(
                    f"KFBA DataBlock {block_index} 缺少必要 DataItem: {missing}",
                    code="kfba_required_item_missing",
                    stage="kfba_index",
                )
            x = int(item_map[0].value)
            y = int(item_map[1].value)
            height = int(item_map[2].value)
            width = int(item_map[3].value)
            field_index = int(item_map[6].value)
            native_level_index = int(item_map[7].value)
            if width <= 0 or height <= 0 or width > self.tile_size or height > self.tile_size:
                raise self._error(
                    f"KFBA DataBlock {block_index} 瓦片尺寸无效",
                    code="tile_geometry_invalid",
                    stage="kfba_index",
                )
            if x < 0 or y < 0 or field_index < 0 or native_level_index < 0:
                raise self._error(
                    f"KFBA DataBlock {block_index} 坐标、字段或层索引无效",
                    code="tile_geometry_invalid",
                    stage="kfba_index",
                )
            offset_table = int(item_map[4].value)
            length_table = int(item_map[5].value)
            table_bytes = self.native_channel_count * 8
            offsets = struct.unpack(f"<{self.native_channel_count}Q", self._read_at(offset_table, table_bytes))
            lengths = struct.unpack(f"<{self.native_channel_count}Q", self._read_at(length_table, table_bytes))
            scale_value = float(self.objective_power) / (2 ** native_level_index)
            for channel_index, (payload_offset, payload_length) in enumerate(zip(offsets, lengths)):
                if payload_length <= 0 or payload_offset + payload_length > self._file_size:
                    raise self._error(
                        f"KFBA DataBlock {block_index} 通道 {channel_index} payload 无效",
                        code="tile_payload_invalid",
                        stage="kfba_index",
                    )
                payload = self._read_at(payload_offset, payload_length)
                payload_codec = _codec_from_payload(payload)
                if self.source_codec == "JXL" and payload_codec != "JXL":
                    raise self._error("KFBA Header 与瓦片 codec 不一致", code="mixed_codec_unsupported", stage="codec")
                if self.source_codec != "JXL" and payload_codec not in {"JPEG", "SOF3"}:
                    raise self._error("KFBA Header 与瓦片 codec 不一致", code="mixed_codec_unsupported", stage="codec")
                bit_depth = 8
                if payload_codec in {"JPEG", "SOF3"}:
                    _marker, bit_depth, components = _jpeg_layout(payload)
                    if self.native_channel_count > 1 and components != 1:
                        raise self._error(
                            "KFBA 多通道 DataBlock 包含非灰度通道 payload",
                            code="kfba_channel_layout_unsupported",
                            stage="codec",
                        )
                tiles.append(
                    NativeTile(
                        index=len(tiles),
                        level_index=native_level_index,
                        field_index=field_index,
                        channel_index=channel_index,
                        z_index=0,
                        t_index=0,
                        x=x,
                        y=y,
                        width=width,
                        height=height,
                        scale_value=scale_value,
                        codec=payload_codec,
                        bit_depth=bit_depth,
                        offset=payload_offset,
                        length=payload_length,
                    )
                )

        self._tiles = tiles
        self.tile_count = len(tiles)
        self.native_fields = tuple(sorted({tile.field_index for tile in tiles}))
        self.default_field_index = self.native_fields[0]
        self._build_levels(group_by_native_index=True)
        self.channels = 3
        self.source_bit_depth = max(tile.bit_depth for tile in tiles)
        first_payload = self._read_at(tiles[0].offset, tiles[0].length)
        if tiles[0].codec == "JXL":
            first_decoded = self._decode_payload(tiles[0])
            packed_components = first_decoded.shape[2] if first_decoded.ndim == 3 else 1
            self.source_bit_depth = first_decoded.dtype.itemsize * 8
        else:
            marker, precision, packed_components = _jpeg_layout(first_payload)
        if self.native_channel_count > 1 and packed_components != 1:
            raise self._error(
                "KFBA 多通道 payload 无法映射到原始 C 轴",
                code="kfba_channel_layout_unsupported",
                stage="codec",
            )
        self.source_channel_count = self.native_channel_count if self.native_channel_count > 1 else packed_components
        if self.source_bit_depth > 8:
            full_range = float((1 << self.source_bit_depth) - 1)
            for metadata in self.channel_metadata:
                metadata["white"] = full_range
        self._can_passthrough_jpeg = (
            len(self.native_fields) == 1
            and self.native_channel_count == 1
            and tiles[0].codec == "JPEG"
            and self.source_bit_depth == 8
            and packed_components in {1, 3}
        )
        self.supports_plane_jpeg_passthrough = (
            tiles[0].codec == "JPEG"
            and marker == 0xC0
            and precision == 8
            and packed_components == 1
        ) if tiles[0].codec != "JXL" else False
        self.native_axes = "TZCYX" if self.requires_raw_ome else ("YX" if packed_components == 1 else "YXS")
        self.compatibility_level = "static_unverified"
        self.base_info = self._build_base_info()

    def _parse_kfba_channel_metadata(self) -> int:
        count_blob = self._header_items.get(75)
        if count_blob is None or len(count_blob) != 4:
            raise self._error(
                "KFBA Header 缺少有效通道数量 DataItem 75",
                code="kfba_required_metadata_missing",
                stage="kfba_metadata",
            )
        channel_count = _u32(count_blob, 0)
        if channel_count <= 0:
            raise self._error("KFBA 通道数量无效", code="kfba_required_metadata_missing", stage="kfba_metadata")

        item_sizes = {76: 20, 77: 40, 78: 4, 79: 12, 84: 8}
        for item_id, item_size in item_sizes.items():
            blob = self._header_items.get(item_id)
            if blob is None or len(blob) != channel_count * item_size:
                raise self._error(
                    f"KFBA Header 缺少有效通道 DataItem {item_id}",
                    code="kfba_required_metadata_missing",
                    stage="kfba_metadata",
                )

        vendor_ids = self._header_items[76]
        names = self._header_items[77]
        equipment = self._header_items[78]
        colors = self._header_items[79]
        exposures = self._header_items[84]
        self.channel_metadata = []
        for channel_index in range(channel_count):
            vendor_id_blob = vendor_ids[channel_index * 20 : (channel_index + 1) * 20]
            vendor_id = vendor_id_blob.split(b"\x00", 1)[0].decode("utf-8", errors="replace").strip()
            name_blob = names[channel_index * 40 : (channel_index + 1) * 40]
            name = name_blob.split(b"\x00", 1)[0].decode("utf-8", errors="replace")
            color = struct.unpack_from("<iii", colors, channel_index * 12)
            if any(component < 0 or component > 255 for component in color):
                raise self._error(
                    f"KFBA 通道 {channel_index} RGB 颜色无效",
                    code="kfba_channel_metadata_invalid",
                    stage="kfba_metadata",
                )
            self.channel_metadata.append(
                {
                    "name": name or f"Channel {channel_index + 1}",
                    "fluor": name or None,
                    "vendor_channel_id": vendor_id or None,
                    "equipment_channel": _i32(equipment, channel_index * 4),
                    "color": tuple(int(component) for component in color),
                    "exposure": struct.unpack_from("<d", exposures, channel_index * 8)[0],
                    "identity_source": "source_metadata" if name else "unknown",
                    "enabled": True,
                    "contrast": 1.0,
                    "brightness": 0.0,
                    "channel_offset": 0.0,
                    "gamma": 1.0,
                    "black": 0.0,
                    "white": 255.0,
                }
            )
        return channel_count

    @property
    def requires_raw_ome(self) -> bool:
        return (
            self.modality == "fluorescence"
            or len(self.native_fields) > 1
            or self.native_channel_count > 1
            or self.native_z_count > 1
            or self.native_t_count > 1
            or self.source_bit_depth > 8
        )

    def _parse_classic(self) -> None:
        prefix = self._read_at(0, 0x60)
        version_value = round(float(_f32(prefix, 0x0C)), 1)
        self.source_version = f"{version_value:.1f}"
        if version_value not in SUPPORTED_CLASSIC_VERSIONS:
            raise self._error(
                f"不支持的 KFB 版本: {self.source_version}",
                code="unsupported_version",
                stage="header",
            )
        self.header_marker = prefix[:4]
        self.record_size = 68 if version_value == 1.0 else 64
        self.tile_count = _u32(prefix, 0x10)
        self.height = _u32(prefix, 0x14)
        self.width = _u32(prefix, 0x18)
        self.objective_power = _u32(prefix, 0x1C)
        self.source_codec = prefix[0x20:0x28].split(b"\x00", 1)[0].decode("ascii", errors="replace").upper()
        self.codec = self.source_codec
        self.raw_timestamp = _u32(prefix, 0x2C)
        self.first_image_record_offset = _u32(prefix, 0x34)
        self.second_image_record_offset = _u32(prefix, 0x38)
        self.last_image_record_offset = _u32(prefix, 0x3C)
        self.tile_index_offset = _u32(prefix, 0x44)
        self.mpp = float(_f32(prefix, 0x4C))
        self.tile_size = _u32(prefix, 0x58)

        if self.width <= 0 or self.height <= 0 or self.tile_count <= 0 or self.tile_size <= 0:
            raise self._error("KFB 主图或瓦片参数无效", code="invalid_header_dimensions", stage="header")
        if self.tile_index_offset < 0x60 or self.tile_index_offset >= self._file_size:
            raise self._error("KFB 索引偏移无效", code="invalid_index_offset", stage="header")
        if self.source_codec not in {"JPEG", "JXL", "LJPEG", "SOF3"}:
            raise self._error(
                f"不支持的 KFB codec: {self.source_codec or '<empty>'}",
                code="unsupported_codec",
                stage="header",
            )

        metadata_end = min(
            [value for value in (self.first_image_record_offset, self.tile_index_offset) if value > 0],
            default=self.tile_index_offset,
        )
        header = self._read_at(0, metadata_end)
        self._header_items = _parse_header_items(header, 0x5C, metadata_end)
        self._metadata = _parse_metadata(header, 0x5C, metadata_end)
        self._parse_classic_channel_metadata()
        self._parse_associated_images()
        self._tiles = self._parse_tile_index(version_value)
        self._build_levels()
        self.channels = 3
        self.source_channel_count = 3
        self.native_axes = "YXS"
        self._inspect_first_tile()
        self.native_axes = "YX" if self.source_channel_count == 1 else "YXS"
        self.compatibility_level = (
            "sample_verified"
            if self.source_container == "kfbf"
            and self.source_version == "2.1"
            and self.source_codec == "JPEG"
            and self.source_bit_depth == 8
            and self.source_channel_count in {1, 3}
            else "static_unverified"
        )
        self.base_info = self._build_base_info()

    def _parse_classic_channel_metadata(self) -> None:
        count_blob = self._header_items.get(75)
        if count_blob is None:
            return
        if len(count_blob) != 4:
            raise self._error(
                "KFBF Header 的通道数量 DataItem 75 无效",
                code="kfbf_channel_metadata_invalid",
                stage="kfbf_metadata",
            )
        channel_count = _u32(count_blob, 0)
        if channel_count <= 0:
            raise self._error(
                "KFBF Header 的通道数量无效",
                code="kfbf_channel_metadata_invalid",
                stage="kfbf_metadata",
            )

        item_sizes = {76: 20, 77: 40, 78: 4, 79: 12, 84: 8}
        values: dict[int, bytes] = {}
        for item_id, item_size in item_sizes.items():
            pointer = self._header_items.get(item_id)
            if pointer is None or len(pointer) != 8:
                raise self._error(
                    f"KFBF Header 缺少有效通道指针 DataItem {item_id}",
                    code="kfbf_channel_metadata_invalid",
                    stage="kfbf_metadata",
                )
            values[item_id] = self._read_at(_u64(pointer, 0), channel_count * item_size)

        vendor_ids = values[76]
        names = values[77]
        equipment = values[78]
        colors = values[79]
        exposures = values[84]
        self.channel_metadata = []
        for channel_index in range(channel_count):
            vendor_id_blob = vendor_ids[channel_index * 20 : (channel_index + 1) * 20]
            vendor_id = vendor_id_blob.split(b"\x00", 1)[0].decode("utf-8", errors="replace").strip()
            name_blob = names[channel_index * 40 : (channel_index + 1) * 40]
            name = name_blob.split(b"\x00", 1)[0].decode("utf-8", errors="replace").strip()
            color = struct.unpack_from("<iii", colors, channel_index * 12)
            if any(component < 0 or component > 255 for component in color):
                raise self._error(
                    f"KFBF 通道 {channel_index} RGB 颜色无效",
                    code="kfbf_channel_metadata_invalid",
                    stage="kfbf_metadata",
                )
            self.channel_metadata.append(
                {
                    "name": name or f"Channel {channel_index + 1}",
                    "fluor": name or None,
                    "vendor_channel_id": vendor_id or None,
                    "equipment_channel": _i32(equipment, channel_index * 4),
                    "color": tuple(int(component) for component in color),
                    "exposure": struct.unpack_from("<d", exposures, channel_index * 8)[0],
                    "identity_source": "source_metadata" if name else "unknown",
                    "enabled": True,
                    "contrast": 1.0,
                    "brightness": 0.0,
                    "channel_offset": 0.0,
                    "gamma": 1.0,
                    "black": 0.0,
                    "white": 255.0,
                }
            )
        self.native_channel_count = channel_count

    def _parse_associated_record(self, offset: int, name: str) -> KfbAssociatedImageRecord:
        head = self._read_at(offset, IMAGE_RECORD_SIZE)
        image_type = head[1]
        if head[0] != 0xF1 or head[2:4] != b"\xee\xee":
            raise self._error(f"KFB {name} 记录头无效", code="associated_marker_invalid", stage="associated")
        if head[48:52] != bytes([0xFF, image_type, 0xEE, 0xEE]):
            raise self._error(f"KFB {name} 记录尾无效", code="associated_marker_invalid", stage="associated")
        width = _u32(head, 12)
        height = _u32(head, 8)
        direct_size = _u32(head, 20)
        if self.source_container in {"kfbf", "kfba"} and float(self.source_version) >= 1.4:
            data_position = _u64(head, 24)
            length_position = _u64(head, 32)
        else:
            data_position = _u32(head, 24)
            length_position = _u32(head, 32)
        if data_position == IMAGE_RECORD_SIZE:
            payload_offset = offset + IMAGE_RECORD_SIZE
            payload_size = direct_size
        elif (
            self.source_container == "kfba"
            or (self.source_container == "kfbf" and float(self.source_version) >= 2.1)
        ):
            payload_offset = _u64(self._read_at(data_position, 8), 0)
            payload_size = _u64(self._read_at(length_position, 8), 0)
            if payload_size != direct_size:
                raise self._error(
                    f"KFBF {name} 长度不一致",
                    code="associated_length_mismatch",
                    stage="associated",
                )
        else:
            raise self._error(
                f"KFB {name} DataBlock 位置语义无法识别",
                code="associated_position_invalid",
                stage="associated",
            )
        if width <= 0 or height <= 0 or payload_size <= 0 or payload_offset + payload_size > self._file_size:
            raise self._error(f"KFB {name} 记录尺寸无效", code="associated_payload_invalid", stage="associated")
        return KfbAssociatedImageRecord(
            name=name,
            offset=offset,
            image_type=image_type,
            width=width,
            height=height,
            channels=_u32(head, 16),
            payload_offset=payload_offset,
            payload_size=payload_size,
        )

    def _parse_associated_images(self) -> None:
        positions = (
            ("macro", self.first_image_record_offset),
            ("label", self.second_image_record_offset),
            ("thumbnail", self.last_image_record_offset),
        )
        seen: set[int] = set()
        for name, offset in positions:
            if offset <= 0 or offset in seen:
                continue
            if offset >= self._file_size:
                raise self._error(f"KFB {name} 偏移无效", code="associated_offset_invalid", stage="associated")
            seen.add(offset)
            self._associated_images[name] = self._parse_associated_record(offset, name)

    def _looks_like_raw_index(self, blob: bytes) -> bool:
        if len(blob) != self.tile_count * self.record_size:
            return False
        end_offset = self.record_size - 4
        return all(
            blob[index * self.record_size : index * self.record_size + 4] == DATA_MARKER
            and blob[index * self.record_size + end_offset : (index + 1) * self.record_size] == DATA_END_MARKER
            for index in range(self.tile_count)
        )

    def _read_index_blob(self) -> bytes:
        expected = self.tile_count * self.record_size
        remaining = self._file_size - self.tile_index_offset
        if remaining >= expected:
            candidate = self._read_at(self.tile_index_offset, expected)
            if self._looks_like_raw_index(candidate):
                return candidate
        if remaining < 12:
            raise self._error("KFB 压缩索引头被截断", code="compressed_index_truncated", stage="index")
        header = self._read_at(self.tile_index_offset, 12)
        marker, compressed_size, uncompressed_size = struct.unpack("<4sII", header)
        if marker != DATA_MARKER or uncompressed_size != expected:
            raise self._error("KFB 索引结构无法识别", code="index_layout_invalid", stage="index")
        compressed = self._read_at(self.tile_index_offset + 12, compressed_size)
        try:
            blob = zlib.decompress(compressed)
        except zlib.error as exc:
            raise self._error("KFB 压缩索引解压失败", code="compressed_index_invalid", stage="index") from exc
        if len(blob) != expected or not self._looks_like_raw_index(blob):
            raise self._error("KFB 压缩索引长度或记录标记无效", code="compressed_index_invalid", stage="index")
        return blob

    def _parse_tile_index(self, version_value: float) -> list[NativeTile]:
        index_blob = self._read_index_blob()
        records: list[NativeTile] = []
        for index in range(self.tile_count):
            entry = index_blob[index * self.record_size : (index + 1) * self.record_size]
            x = _i32(entry, 4)
            y = _i32(entry, 8)
            width = _i32(entry, 12)
            height = _i32(entry, 16)
            scale_value = round(_f32(entry, 20), 6)
            direct_length = _u32(entry, 32)
            if version_value >= 2.1:
                position = _u64(entry, 36)
                length_position = _u64(entry, 44)
                if self.source_container == "kfbf":
                    payload_offset = _u64(self._read_at(position, 8), 0)
                    payload_length = _u64(self._read_at(length_position, 8), 0)
                else:
                    payload_offset = position
                    payload_length = direct_length
            elif self.source_container == "kfbf":
                position = _u32(entry, 36)
                length_position = _u32(entry, 44)
                payload_offset = _u64(self._read_at(position, 8), 0)
                payload_length = _u64(self._read_at(length_position, 8), 0)
            else:
                payload_offset = self.tile_index_offset + _i32(entry, 36)
                payload_length = direct_length
            if width <= 0 or height <= 0 or scale_value <= 0:
                raise self._error(f"KFB 瓦片几何信息无效: {index}", code="tile_geometry_invalid", stage="index")
            if payload_length != direct_length:
                raise self._error(f"KFBF 瓦片长度不一致: {index}", code="tile_length_mismatch", stage="index")
            if payload_offset < 0 or payload_offset + payload_length > self._file_size:
                raise self._error(f"KFB 瓦片位置无效: {index}", code="tile_payload_invalid", stage="index")
            records.append(
                NativeTile(
                    index=index, level_index=-1, field_index=0, channel_index=0, z_index=0, t_index=0,
                    x=x, y=y, width=width, height=height, scale_value=scale_value,
                    codec=self.source_codec, bit_depth=8, offset=payload_offset, length=payload_length,
                )
            )
        return records

    def _build_levels(self, *, group_by_native_index: bool = False) -> None:
        groups: dict[int | float, list[NativeTile]] = defaultdict(list)
        for tile in self._tiles:
            key: int | float = tile.level_index if group_by_native_index else tile.scale_value
            groups[key].append(tile)
        if not groups:
            raise self._error("KFB 未包含可读取瓦片", code="empty_index", stage="index")
        self._levels = []
        self._tile_maps = []
        ordered_keys = sorted(groups) if group_by_native_index else sorted(groups, reverse=True)
        all_records: list[NativeTile] = []
        for level_index, group_key in enumerate(ordered_keys):
            source_records = sorted(
                groups[group_key],
                key=lambda tile: (tile.field_index, tile.y, tile.x, tile.channel_index, tile.z_index, tile.t_index),
            )
            records = tuple(replace(tile, level_index=level_index) for tile in source_records)
            scale_value = records[0].scale_value
            width = max(tile.x + tile.width for tile in records)
            height = max(tile.y + tile.height for tile in records)
            level = NativeLevel(level_index, scale_value, width, height, self.tile_size, records)
            self._levels.append(level)
            self._tile_maps.append(
                {
                    (tile.field_index, tile.x, tile.y, tile.channel_index, tile.z_index, tile.t_index): tile
                    for tile in records
                }
            )
            all_records.extend(records)
        self._tiles = all_records
        self.level_dimensions = [level.dimensions for level in self._levels]
        self.level_grids = [(level.columns, level.rows) for level in self._levels]
        self.main_scale = self._levels[0].scale_value
        if self.level_dimensions[0] != (self.width, self.height):
            raise self._error(
                "KFB 最大原生层尺寸与 Header 不一致",
                code="main_level_dimension_mismatch",
                stage="index",
            )

    def _inspect_first_tile(self) -> None:
        first = self._levels[0].records[0]
        payload = self._read_at(first.offset, first.length)
        if self.source_codec in {"JPEG", "LJPEG", "SOF3"}:
            marker, precision, components = _jpeg_layout(payload)
            if not marker:
                raise self._error("KFB JPEG 瓦片缺少有效 SOF", code="jpeg_header_invalid", stage="codec")
            self.source_bit_depth = precision
            self.source_channel_count = components
            self._can_passthrough_jpeg = marker == 0xC0 and precision == 8 and components in {1, 3}
            self.supports_plane_jpeg_passthrough = marker == 0xC0 and precision == 8 and components == 1
        else:
            try:
                import imagecodecs

                decoded = np.asarray(imagecodecs.jpegxl_decode(payload))
            except Exception as exc:
                raise self._error("KFB JXL 瓦片头无法解码", code="jxl_header_invalid", stage="codec") from exc
            self.source_bit_depth = decoded.dtype.itemsize * 8
            self.source_channel_count = decoded.shape[2] if decoded.ndim == 3 else 1
            self._can_passthrough_jpeg = False
            self.supports_plane_jpeg_passthrough = False

    def _build_base_info(self) -> BaseInfo:
        return BaseInfo(
            magic_no=self.source_container.upper(), version=self.source_version,
            focus_num=0, image_format=0, layer_size=len(self._levels),
            img_color=self.source_bit_depth * max(1, self.source_channel_count),
            check_sum=0, ratio_step=2, max_layer_size=self.objective_power,
            slide_type=0, background_color=255,
            pixel_size_mm=(self.mpp / 1000.0) if self.mpp > 0 else 0.0,
            total_img_num=len(self._tiles), max_zoom_rate=self.objective_power,
            img_col=self._levels[0].columns, img_row=self._levels[0].rows,
            img_width=self.tile_size, img_height=self.tile_size,
            tile_width=self.tile_size, tile_height=self.tile_size,
            shrink_tile_num=max(0, len(self._tiles) - len(self._levels[0].records)),
            total_img_width=self.width, total_img_height=self.height,
        )

    def _decode_payload(self, tile: NativeTile) -> np.ndarray:
        payload = self._read_at(tile.offset, tile.length)
        try:
            import imagecodecs
            if tile.codec == "JXL":
                decoded = imagecodecs.jpegxl_decode(payload)
            elif tile.codec == "LJPEG":
                decoded = imagecodecs.ljpeg_decode(payload)
            elif tile.codec == "SOF3":
                decoded = imagecodecs.jpegsof3_decode(payload)
            else:
                decoded = imagecodecs.jpeg_decode(payload)
        except Exception as exc:
            raise self._error(f"KFB 瓦片解码失败: {tile.index}", code="tile_decode_failed", stage="codec") from exc
        return np.asarray(decoded)

    def _decode_tile(self, tile: NativeTile) -> np.ndarray:
        with self._lock:
            cached = self._tile_cache.get(tile.index)
            if cached is not None:
                self._tile_cache.move_to_end(tile.index)
                decoded = cached
            else:
                decoded = np.ascontiguousarray(self._decode_payload(tile))
                if decoded.shape[:2] != (tile.height, tile.width):
                    raise self._error(
                        f"KFB 瓦片解码尺寸不一致: {tile.index}",
                        code="tile_dimension_mismatch",
                        stage="codec",
                    )
                self._tile_cache[tile.index] = decoded
                self._tile_cache.move_to_end(tile.index)
                while len(self._tile_cache) > self.cache_size:
                    self._tile_cache.popitem(last=False)
        display = decoded
        if display.ndim == 2:
            display = np.repeat(display[..., None], 3, axis=2)
        elif display.ndim == 3 and display.shape[2] == 1:
            display = np.repeat(display, 3, axis=2)
        elif display.ndim != 3 or display.shape[2] < 3:
            raise self._error(f"KFB 瓦片通道布局无效: {tile.index}", code="tile_channel_layout_invalid", stage="codec")
        display = display[..., :3]
        if display.dtype != np.uint8:
            max_value = float((1 << max(1, tile.bit_depth, self.source_bit_depth)) - 1)
            display = np.rint(np.clip(display.astype(np.float64) / max_value, 0.0, 1.0) * 255.0).astype(np.uint8)
        return np.ascontiguousarray(display)

    def read_level_region(self, level_index: int, x: int, y: int, width: int, height: int) -> np.ndarray:
        if self.native_channel_count > 1:
            display_channels: list[np.ndarray] = []
            colors: list[tuple[int, int, int]] = []
            for channel_index, metadata in enumerate(self.channel_metadata):
                if not bool(metadata.get("enabled", True)):
                    continue
                plane = self.read_level_field_plane_region(
                    level_index,
                    self.default_field_index,
                    channel_index,
                    0,
                    0,
                    x,
                    y,
                    width,
                    height,
                )
                display_channels.append(
                    apply_vendor_lut(
                        plane,
                        contrast=float(metadata.get("contrast", 1.0)),
                        brightness=float(metadata.get("brightness", 0.0)),
                        channel_offset=float(metadata.get("channel_offset", 0.0)),
                        gamma=float(metadata.get("gamma", 1.0)),
                        black=float(metadata.get("black", 0.0)),
                        white=float(metadata.get("white", (1 << self.source_bit_depth) - 1)),
                    )
                )
                colors.append(tuple(metadata.get("color", (255, 255, 255))))
            if not display_channels:
                return np.zeros((max(0, height), max(0, width), 3), dtype=np.uint8)
            return compose_vendor_channels(display_channels, colors, mode=self.composite_mode)

        if level_index < 0 or level_index >= len(self._levels):
            raise IndexError(f"KFB 金字塔层索引越界: {level_index}")
        if width <= 0 or height <= 0:
            return np.empty((max(0, height), max(0, width), 3), dtype=np.uint8)
        level = self._levels[level_index]
        region = np.full((height, width, 3), self.base_info.background_color, dtype=np.uint8)
        request_x0 = max(0, x)
        request_y0 = max(0, y)
        request_x1 = min(level.width, x + width)
        request_y1 = min(level.height, y + height)
        if request_x0 >= request_x1 or request_y0 >= request_y1:
            return region
        tile_map = self._tile_maps[level_index]
        first_x = (request_x0 // self.tile_size) * self.tile_size
        first_y = (request_y0 // self.tile_size) * self.tile_size
        for tile_y in range(first_y, request_y1, self.tile_size):
            for tile_x in range(first_x, request_x1, self.tile_size):
                tile = tile_map.get((self.default_field_index, tile_x, tile_y, 0, 0, 0))
                if tile is None:
                    continue
                ix0 = max(request_x0, tile.x)
                iy0 = max(request_y0, tile.y)
                ix1 = min(request_x1, tile.x + tile.width)
                iy1 = min(request_y1, tile.y + tile.height)
                if ix0 >= ix1 or iy0 >= iy1:
                    continue
                decoded = self._decode_tile(tile)
                src_x0, src_y0 = ix0 - tile.x, iy0 - tile.y
                dst_x0, dst_y0 = ix0 - x, iy0 - y
                copy_width, copy_height = ix1 - ix0, iy1 - iy0
                region[dst_y0 : dst_y0 + copy_height, dst_x0 : dst_x0 + copy_width] = decoded[
                    src_y0 : src_y0 + copy_height,
                    src_x0 : src_x0 + copy_width,
                ]
        return region

    def read_region(
        self, x: int, y: int, width: int, height: int, *, decode_workers: int | None = None,
    ) -> np.ndarray:
        del decode_workers
        return self.read_level_region(0, x, y, width, height)

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
            self.default_field_index,
            channel_index,
            z_index,
            t_index,
            x,
            y,
            width,
            height,
        )

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
        if level_index < 0 or level_index >= len(self._levels):
            raise IndexError(f"KFB 金字塔层索引越界: {level_index}")
        if field_index not in self.native_fields:
            raise IndexError(f"KFB 字段索引不存在: {field_index}")
        if z_index < 0 or z_index >= self.native_z_count or t_index < 0 or t_index >= self.native_t_count:
            raise IndexError("KFB Z/T 索引越界")
        if channel_index < 0 or channel_index >= self.source_channel_count:
            raise IndexError(f"KFB 通道索引不存在: {channel_index}")
        if width <= 0 or height <= 0:
            dtype = np.uint16 if self.source_bit_depth > 8 else np.uint8
            return np.empty((max(0, height), max(0, width)), dtype=dtype)

        level = self._levels[level_index]
        dtype = np.uint16 if self.source_bit_depth > 8 else np.uint8
        background = 0 if self.modality == "fluorescence" else self.base_info.background_color
        region = np.full((height, width), background, dtype=dtype)
        request_x0 = max(0, x)
        request_y0 = max(0, y)
        request_x1 = min(level.width, x + width)
        request_y1 = min(level.height, y + height)
        if request_x0 >= request_x1 or request_y0 >= request_y1:
            return region

        tile_map = self._tile_maps[level_index]
        logical_channel = channel_index if self.native_channel_count > 1 else 0
        first_x = (request_x0 // self.tile_size) * self.tile_size
        first_y = (request_y0 // self.tile_size) * self.tile_size
        for tile_y in range(first_y, request_y1, self.tile_size):
            for tile_x in range(first_x, request_x1, self.tile_size):
                tile = tile_map.get((field_index, tile_x, tile_y, logical_channel, z_index, t_index))
                if tile is None:
                    continue
                with self._lock:
                    decoded = self._tile_cache.get(tile.index)
                    if decoded is not None:
                        self._tile_cache.move_to_end(tile.index)
                if decoded is None:
                    decoded = np.ascontiguousarray(self._decode_payload(tile))
                    if decoded.shape[:2] != (tile.height, tile.width):
                        raise self._error(
                            f"KFB 瓦片解码尺寸不一致: {tile.index}",
                            code="tile_dimension_mismatch",
                            stage="codec",
                        )
                    with self._lock:
                        self._tile_cache[tile.index] = decoded
                        self._tile_cache.move_to_end(tile.index)
                        while len(self._tile_cache) > self.cache_size:
                            self._tile_cache.popitem(last=False)
                if decoded.ndim == 3:
                    sample_index = 0 if self.native_channel_count > 1 else channel_index
                    if sample_index >= decoded.shape[2]:
                        raise self._error(
                            f"KFB 瓦片缺少请求的样本通道: {tile.index}",
                            code="tile_channel_layout_invalid",
                            stage="codec",
                        )
                    decoded = decoded[..., sample_index]
                elif decoded.ndim != 2:
                    raise self._error(
                        f"KFB 瓦片通道布局无效: {tile.index}",
                        code="tile_channel_layout_invalid",
                        stage="codec",
                    )

                ix0 = max(request_x0, tile.x)
                iy0 = max(request_y0, tile.y)
                ix1 = min(request_x1, tile.x + tile.width)
                iy1 = min(request_y1, tile.y + tile.height)
                if ix0 >= ix1 or iy0 >= iy1:
                    continue
                src_x0, src_y0 = ix0 - tile.x, iy0 - tile.y
                dst_x0, dst_y0 = ix0 - x, iy0 - y
                copy_width, copy_height = ix1 - ix0, iy1 - iy0
                region[dst_y0 : dst_y0 + copy_height, dst_x0 : dst_x0 + copy_width] = decoded[
                    src_y0 : src_y0 + copy_height,
                    src_x0 : src_x0 + copy_width,
                ]
        return region

    def iter_native_level_tiles(
        self,
        level_index: int,
        channel_index: int = 0,
        z_index: int = 0,
        t_index: int = 0,
        field_index: int | None = None,
    ) -> Iterator[NativeTile]:
        if level_index < 0 or level_index >= len(self._levels):
            raise IndexError(f"KFB 金字塔层索引越界: {level_index}")
        selected_field = self.default_field_index if field_index is None else field_index
        for tile in self._levels[level_index].records:
            if (
                tile.field_index,
                tile.channel_index,
                tile.z_index,
                tile.t_index,
            ) == (selected_field, channel_index, z_index, t_index):
                yield replace(tile, payload=self._read_at(tile.offset, tile.length))

    def iter_native_level_jpegs(self, level_index: int) -> Iterator[bytes | None]:
        if not self._can_passthrough_jpeg:
            raise self._error(
                "当前 KFB 瓦片不能直接重封装为 TIFF JPEG",
                code="jpeg_passthrough_unavailable",
                stage="write",
            )
        level = self._levels[level_index]
        tile_map = self._tile_maps[level_index]
        for row in range(level.rows):
            for column in range(level.columns):
                tile = tile_map.get(
                    (self.default_field_index, column * self.tile_size, row * self.tile_size, 0, 0, 0)
                )
                yield None if tile is None else self._read_at(tile.offset, tile.length)

    def iter_native_level_plane_jpegs(
        self,
        level_index: int,
        channel_index: int,
        z_index: int,
        t_index: int,
        field_index: int | None = None,
    ) -> Iterator[bytes | None]:
        if self.source_codec != "JPEG" or self.source_bit_depth != 8:
            raise self._error(
                "当前 KFB 平面不能直接重封装为 TIFF JPEG",
                code="jpeg_passthrough_unavailable",
                stage="write",
            )
        level = self._levels[level_index]
        selected_field = self.default_field_index if field_index is None else field_index
        tile_map = self._tile_maps[level_index]
        for row in range(level.rows):
            for column in range(level.columns):
                tile = tile_map.get(
                    (
                        selected_field,
                        column * self.tile_size,
                        row * self.tile_size,
                        channel_index,
                        z_index,
                        t_index,
                    )
                )
                if tile is None:
                    yield None
                    continue
                payload = self._read_at(tile.offset, tile.length)
                marker, precision, components = _jpeg_layout(payload)
                if marker != 0xC0 or precision != 8 or components != 1:
                    raise self._error(
                        "KFB 平面包含不能直接重封装的 JPEG 瓦片",
                        code="jpeg_passthrough_unavailable",
                        stage="write",
                    )
                yield payload

    def _associated_image(self, name: str) -> Image.Image | None:
        cached = self._associated_cache.get(name)
        if cached is not None:
            return cached.copy()
        record = self._associated_images.get(name)
        if record is None:
            return None
        payload = self._read_at(record.payload_offset, record.payload_size)
        try:
            with Image.open(BytesIO(payload)) as image:
                image.load()
                decoded = image.copy()
        except Exception as exc:
            raise self._error(f"KFB {name} 解码失败", code="associated_decode_failed", stage="associated") from exc
        if decoded.size != (record.width, record.height):
            raise self._error(f"KFB {name} 解码尺寸不一致", code="associated_dimension_mismatch", stage="associated")
        self._associated_cache[name] = decoded
        return decoded.copy()

    def get_macro_image(self) -> Image.Image | None:
        return self._associated_image("macro")

    def get_preview_image(self) -> Image.Image | None:
        candidates = [
            level
            for level in self._levels[1:]
            if max(level.width, level.height) <= 2048
        ]
        if candidates:
            level = max(candidates, key=lambda item: item.width * item.height)
            return Image.fromarray(
                self.read_level_region(level.index, 0, 0, level.width, level.height),
                mode="RGB",
            )
        return self.get_macro_image()

    def get_label_image(self) -> Image.Image | None:
        return self._associated_image("label")

    def get_thumbnail_image(self) -> Image.Image | None:
        return self._associated_image("thumbnail")

    def get_scan_metadata(self) -> dict[str, object]:
        metadata = dict(self._metadata)
        metadata.update(
            {
                "format": f"jiangfeng.{self.source_container}",
                "sourceContainer": self.source_container,
                "sourceVersion": self.source_version,
                "sourceCodec": self.source_codec,
                "sourceBitDepth": self.source_bit_depth,
                "sourceChannelCount": self.source_channel_count,
                "sourceAxes": self.native_axes,
                "modality": self.modality,
                "supportsNativePlanes": self.supports_native_planes,
                "nativeFields": self.native_fields,
                "nativeChannelCount": self.native_channel_count,
                "nativeZCount": self.native_z_count,
                "nativeTCount": self.native_t_count,
                "channelMetadata": self.channel_metadata,
                "compatibilityLevel": self.compatibility_level,
                "scanTime": self.raw_timestamp,
                "mpp": self.mpp,
                "mppX": self.mpp_x,
                "mppY": self.mpp_y,
                "physicalWidthUm": self.width * self.mpp_x,
                "physicalHeightUm": self.height * self.mpp_y,
                "width": self.width,
                "height": self.height,
                "levelDimensions": self.level_dimensions,
                "nativeResources": self.native_resource_dimensions,
                "nativeTileMode": self.native_tile_mode,
            }
        )
        return metadata
