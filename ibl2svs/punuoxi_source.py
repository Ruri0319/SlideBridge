from __future__ import annotations

from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import re
import struct
import threading

import numpy as np
from PIL import Image

from .jpeg_transform import transpose_jpeg
from .models import BaseInfo


HEADER_MIN_SIZE = 0xAC
THUMBNAIL_OFFSET_OFFSET = 0x00
LABEL_OFFSET_OFFSET = 0x08
INDEX_OFFSET_OFFSET = 0x10
PHYSICAL_WIDTH_OFFSET = 0x18
PHYSICAL_HEIGHT_OFFSET = 0x1C
IMAGE_WIDTH_OFFSET = 0x20
IMAGE_HEIGHT_OFFSET = 0x24
SCAN_TIME_OFFSET = 0x40
DEVICE_LENGTH_OFFSET = 0x53
DEVICE_OFFSET = 0x57
INSTITUTION_OFFSET = 0x84
CASE_OFFSET = 0x98
TILE_SIZE = 256
TAIL_SIZE = 33
LEVEL_HEADER_SIZE = 8
INDEX_RECORD_SIZE = 12
HEX_ID_RE = re.compile(rb"^[0-9a-fA-F]{32}\x00$")
NATIVE_THUMBNAIL_WIDTH = 300
NATIVE_MACRO_WIDTH = 1152
NATIVE_MACRO_HEIGHT = 625
NATIVE_MACRO_PIXEL_BYTES = NATIVE_MACRO_WIDTH * NATIVE_MACRO_HEIGHT * 3
NATIVE_MACRO_TRAILER_BYTES = 24
NATIVE_LABEL_WIDTH = 300


class PunuoxiFormatError(RuntimeError):
    """Raised when a private Punuoxi ``.image`` container is invalid."""


@dataclass(frozen=True)
class PunuoxiTileRecord:
    index: int
    column: int
    row: int
    jpeg_length: int
    jpeg_offset: int


@dataclass(frozen=True)
class PunuoxiLevel:
    index: int
    columns: int
    rows: int
    tile_size: int
    index_offset: int
    records: tuple[PunuoxiTileRecord, ...]

    @property
    def width(self) -> int:
        return self.columns * self.tile_size

    @property
    def height(self) -> int:
        return self.rows * self.tile_size

    @property
    def dimensions(self) -> tuple[int, int]:
        return self.width, self.height


def _unpack_from(fmt: str, data: bytes, offset: int):
    size = struct.calcsize(fmt)
    if offset < 0 or offset + size > len(data):
        raise PunuoxiFormatError("IMAGE 文件头或索引被截断")
    return struct.unpack_from(fmt, data, offset)[0]


def _decode_header_text(data: bytes, offset: int, size: int) -> str:
    if offset < 0 or offset + size > len(data):
        return ""
    raw = bytes(data[offset : offset + size]).split(b"\x00", 1)[0]
    if not raw:
        return ""
    try:
        return raw.decode("gbk").strip()
    except UnicodeDecodeError:
        return raw.decode("utf-8", errors="replace").strip()


class PunuoxiImageSource:
    """Read the JPEG-tile pyramid stored by the investigated ``.image`` format.

    The container keeps axis-transposed JPEG tiles before a compact,
    little-endian, column-major index. Only the index is loaded into Python
    objects; JPEG payloads are decoded and reoriented on demand, then kept in
    a bounded LRU cache.
    """

    def __init__(self, path: str | Path, cache_size: int = 256):
        self.path = Path(path)
        self.cache_size = max(1, int(cache_size))
        self._fh = self.path.open("rb")
        self._lock = threading.RLock()
        self._tile_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._preview_image: Image.Image | None = None
        self._thumbnail_image: Image.Image | None = None
        self._macro_image: Image.Image | None = None
        self._label_image: Image.Image | None = None
        self._thumbnail_offset = 0
        self._label_offset = 0
        self._native_resource_status = "unavailable"
        self._native_resource_reason = "文件未声明原生附属图资源"
        self._native_tile_mode = "uninitialized"
        self._native_resources: dict[str, tuple[int, int, int] | None] = {
            "thumbnail": None,
            "macro": None,
            "label": None,
        }

        try:
            self._file_size = self.path.stat().st_size
            if self._file_size < HEADER_MIN_SIZE + TAIL_SIZE:
                raise PunuoxiFormatError("IMAGE 文件过小，缺少固定头部或尾部标识")

            self._parse_header()
            self._levels = self._parse_levels()
            if not self._levels:
                raise PunuoxiFormatError("IMAGE 未包含金字塔索引")
            self._parse_native_resources()

            self.width = self._header_width
            self.height = self._header_height
            self.channels = 3
            self.modality = "brightfield"
            self.native_fields = (0,)
            self.native_channel_count = 1
            self.native_z_count = 1
            self.native_t_count = 1
            self.source_channel_count = 3
            self.source_bit_depth = 8
            self.channel_metadata = []
            self.supports_native_planes = False
            self.tile_size = TILE_SIZE
            self.level_dimensions = [level.dimensions for level in self._levels]
            self.level_grids = [(level.columns, level.rows) for level in self._levels]
            self.base_info = self._build_base_info()
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        with self._lock:
            self._tile_cache.clear()
            fh = getattr(self, "_fh", None)
            if fh is not None:
                fh.close()
                self._fh = None

    def __enter__(self) -> "PunuoxiImageSource":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.height, self.width, self.channels

    @property
    def mpp_x(self) -> float:
        return self._physical_width_um / self.width

    @property
    def mpp_y(self) -> float:
        return self._physical_height_um / self.height

    @property
    def levels(self) -> tuple[PunuoxiLevel, ...]:
        return tuple(self._levels)

    @property
    def index_offset(self) -> int:
        return self._index_offset

    @property
    def tail_identifier(self) -> str:
        return self._tail_identifier

    @property
    def native_resource_status(self) -> str:
        return self._native_resource_status

    @property
    def native_resource_reason(self) -> str:
        return self._native_resource_reason

    @property
    def native_tile_mode(self) -> str:
        return self._native_tile_mode

    @property
    def native_resource_dimensions(self) -> dict[str, tuple[int, int] | None]:
        return {
            name: (value[1], value[2]) if value else None
            for name, value in self._native_resources.items()
        }

    @property
    def supports_native_pyramid(self) -> bool:
        return bool(self._levels)

    def _read_at(self, offset: int, size: int) -> bytes:
        with self._lock:
            fh = self._fh
            if fh is None:
                raise PunuoxiFormatError("IMAGE 文件已关闭")
            fh.seek(offset)
            data = fh.read(size)
        if len(data) != size:
            raise PunuoxiFormatError("IMAGE 文件头、索引或瓦片数据被截断")
        return data

    def _parse_header(self) -> None:
        data = self._read_at(0, HEADER_MIN_SIZE)

        tail = self._read_at(self._file_size - TAIL_SIZE, TAIL_SIZE)
        if not HEX_ID_RE.fullmatch(tail):
            raise PunuoxiFormatError("IMAGE 尾部 32 字节标识无效")
        self._tail_identifier = tail[:-1].decode("ascii").lower()

        self._thumbnail_offset = int(_unpack_from("<Q", data, THUMBNAIL_OFFSET_OFFSET))
        self._label_offset = int(_unpack_from("<Q", data, LABEL_OFFSET_OFFSET))
        self._index_offset = int(_unpack_from("<Q", data, INDEX_OFFSET_OFFSET))
        self._physical_width_um = float(_unpack_from("<f", data, PHYSICAL_WIDTH_OFFSET))
        self._physical_height_um = float(_unpack_from("<f", data, PHYSICAL_HEIGHT_OFFSET))
        self._header_width = int(_unpack_from("<I", data, IMAGE_WIDTH_OFFSET))
        self._header_height = int(_unpack_from("<I", data, IMAGE_HEIGHT_OFFSET))
        self._max_zoom_rate = int(_unpack_from("<I", data, 0x38))

        index_end = self._file_size - TAIL_SIZE
        if self._index_offset < HEADER_MIN_SIZE or self._index_offset >= index_end:
            raise PunuoxiFormatError("IMAGE 金字塔索引偏移无效")
        if self._header_width <= 0 or self._header_height <= 0:
            raise PunuoxiFormatError("IMAGE 主图尺寸无效")
        if self._physical_width_um <= 0 or self._physical_height_um <= 0:
            raise PunuoxiFormatError("IMAGE 物理尺寸无效")

        # The field is a fixed 19-byte ASCII timestamp. Bytes immediately
        # after it belong to the following public/header fields.
        self._scan_time = _decode_header_text(data, SCAN_TIME_OFFSET, 0x13)
        device_length = int(_unpack_from("<I", data, DEVICE_LENGTH_OFFSET))
        if DEVICE_OFFSET + device_length > INSTITUTION_OFFSET:
            raise PunuoxiFormatError("IMAGE 设备号长度无效")
        self._device_no = _decode_header_text(data, DEVICE_OFFSET, device_length)
        self._institution = _decode_header_text(data, INSTITUTION_OFFSET, 0x0C)
        self._case_no = _decode_header_text(data, CASE_OFFSET, 0x14)

    def _parse_levels(self) -> list[PunuoxiLevel]:
        index_end = self._file_size - TAIL_SIZE
        cursor = self._index_offset
        levels: list[PunuoxiLevel] = []
        tile_index = 0
        while cursor < index_end:
            if len(levels) >= 32:
                raise PunuoxiFormatError("IMAGE 金字塔层数异常")
            if cursor + LEVEL_HEADER_SIZE > index_end:
                raise PunuoxiFormatError("IMAGE 金字塔层头被截断")

            level_header = self._read_at(cursor, LEVEL_HEADER_SIZE)
            columns = int(_unpack_from("<I", level_header, 0))
            rows = int(_unpack_from("<I", level_header, 4))
            if columns <= 0 or rows <= 0:
                raise PunuoxiFormatError(f"IMAGE 第 {len(levels)} 层网格尺寸无效")

            record_count = columns * rows
            records_size = record_count * INDEX_RECORD_SIZE
            records_start = cursor + LEVEL_HEADER_SIZE
            records_end = records_start + records_size
            if records_end > index_end:
                raise PunuoxiFormatError(f"IMAGE 第 {len(levels)} 层索引被截断")
            index_data = self._read_at(records_start, records_size)

            physical_records: list[PunuoxiTileRecord] = []
            # The vendor stores the index column-major: all rows for column 0,
            # then all rows for column 1, and so on.  Keep the decoded records
            # in row-major coordinate order below so read_region() can use the
            # same coordinate lookup as the other slide sources.
            for column in range(columns):
                for row in range(rows):
                    physical_index = column * rows + row
                    offset = physical_index * INDEX_RECORD_SIZE
                    jpeg_length = int(_unpack_from("<I", index_data, offset))
                    jpeg_offset = int(_unpack_from("<Q", index_data, offset + 4))
                    if jpeg_offset < HEADER_MIN_SIZE or jpeg_offset + jpeg_length > self._index_offset:
                        raise PunuoxiFormatError(
                            f"IMAGE 第 {len(levels)} 层瓦片 ({column},{row}) 偏移无效"
                        )
                    # Real scanner files use zero-length index records for
                    # blank rows that pad the declared slide canvas.
                    if (
                        jpeg_length
                        and self._read_at(jpeg_offset, 3) != b"\xff\xd8\xff"
                    ):
                        raise PunuoxiFormatError(
                            f"IMAGE 第 {len(levels)} 层瓦片 ({column},{row}) 不是 JPEG"
                        )
                    if (
                        jpeg_length
                        and self._read_at(jpeg_offset + jpeg_length - 2, 2) != b"\xff\xd9"
                    ):
                        raise PunuoxiFormatError(
                            f"IMAGE 第 {len(levels)} 层瓦片 ({column},{row}) JPEG 尾标记无效"
                        )
                    physical_records.append(
                        PunuoxiTileRecord(
                            index=tile_index,
                            column=column,
                            row=row,
                            jpeg_length=jpeg_length,
                            jpeg_offset=jpeg_offset,
                        )
                    )
                    tile_index += 1

            records = tuple(
                physical_records[column * rows + row]
                for row in range(rows)
                for column in range(columns)
            )

            levels.append(
                PunuoxiLevel(
                    index=len(levels),
                    columns=columns,
                    rows=rows,
                    tile_size=TILE_SIZE,
                    index_offset=cursor,
                    records=records,
                )
            )
            cursor = records_end

        if cursor != index_end:
            raise PunuoxiFormatError("IMAGE 金字塔索引与尾部标识之间存在未知数据")
        first = levels[0]
        if first.dimensions != (self._header_width, self._header_height):
            raise PunuoxiFormatError(
                "IMAGE 主层网格尺寸与头部尺寸不一致: "
                f"{first.width}x{first.height} != {self._header_width}x{self._header_height}"
            )
        return levels

    def _first_tile_offset(self) -> int | None:
        offsets = [
            record.jpeg_offset
            for level in self._levels
            for record in level.records
            if record.jpeg_length > 0
        ]
        return min(offsets) if offsets else None

    def _parse_native_resources(self) -> None:
        """Parse the raw RGB resources used by the investigated IMAGE files.

        The resource pointers are optional in older/synthetic files. A failed
        resource probe must not prevent the main JPEG pyramid from being read;
        callers can then use the legacy preview/associated-image path.
        """

        if not self._thumbnail_offset or not self._label_offset:
            return
        if not (
            HEADER_MIN_SIZE <= self._thumbnail_offset < self._label_offset < self._index_offset
        ):
            self._native_resource_status = "legacy_fallback"
            self._native_resource_reason = "原生附属图偏移不满足已验证布局"
            return

        thumbnail_bytes = (
            self._label_offset
            - self._thumbnail_offset
            - NATIVE_MACRO_PIXEL_BYTES
            - NATIVE_MACRO_TRAILER_BYTES
        )
        if thumbnail_bytes <= 0 or thumbnail_bytes % (NATIVE_THUMBNAIL_WIDTH * 3) != 0:
            self._native_resource_status = "legacy_fallback"
            self._native_resource_reason = "无法从资源偏移推导 thumbnail 尺寸"
            return
        thumbnail_height = thumbnail_bytes // (NATIVE_THUMBNAIL_WIDTH * 3)
        macro_offset = self._thumbnail_offset + thumbnail_bytes
        label_pixels_end = self._first_tile_offset()
        if label_pixels_end is None or label_pixels_end <= self._label_offset:
            self._native_resource_status = "legacy_fallback"
            self._native_resource_reason = "无法确定 label 资源结束位置"
            return
        label_bytes = label_pixels_end - self._label_offset
        if label_bytes <= 0 or label_bytes % (NATIVE_LABEL_WIDTH * 3) != 0:
            self._native_resource_status = "legacy_fallback"
            self._native_resource_reason = "无法从首个 JPEG 偏移推导 label 尺寸"
            return
        label_height = label_bytes // (NATIVE_LABEL_WIDTH * 3)
        macro_end = macro_offset + NATIVE_MACRO_PIXEL_BYTES + NATIVE_MACRO_TRAILER_BYTES
        if macro_end != self._label_offset or macro_end > self._file_size:
            self._native_resource_status = "legacy_fallback"
            self._native_resource_reason = "imageThumb 宏观图长度不符合已验证布局"
            return

        self._native_resources = {
            "thumbnail": (self._thumbnail_offset, NATIVE_THUMBNAIL_WIDTH, thumbnail_height),
            "macro": (macro_offset, NATIVE_MACRO_WIDTH, NATIVE_MACRO_HEIGHT),
            "label": (self._label_offset, NATIVE_LABEL_WIDTH, label_height),
        }
        self._native_resource_status = "native"
        self._native_resource_reason = ""

    def _read_rgb_resource(self, name: str) -> Image.Image | None:
        resource = self._native_resources.get(name)
        if resource is None:
            return None
        offset, width, height = resource
        size = width * height * 3
        raw = self._read_at(offset, size)
        array = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3)).copy()
        return Image.fromarray(array, mode="RGB")

    def _build_base_info(self) -> BaseInfo:
        mpp_x = self.mpp_x
        mpp_y = self.mpp_y
        mpp = (mpp_x + mpp_y) / 2.0
        total_tiles = sum(len(level.records) for level in self._levels)
        return BaseInfo(
            magic_no="image",
            version="1",
            focus_num=0,
            image_format=1,
            layer_size=len(self._levels),
            img_color=24,
            check_sum=0,
            ratio_step=2,
            max_layer_size=self._max_zoom_rate,
            slide_type=0,
            background_color=250,
            pixel_size_mm=mpp / 1000.0,
            total_img_num=total_tiles,
            max_zoom_rate=self._max_zoom_rate,
            img_col=self._levels[0].columns,
            img_row=self._levels[0].rows,
            img_width=TILE_SIZE,
            img_height=TILE_SIZE,
            tile_width=TILE_SIZE,
            tile_height=TILE_SIZE,
            shrink_tile_num=max(0, total_tiles - len(self._levels[0].records)),
            total_img_width=self.width,
            total_img_height=self.height,
        )

    def _decode_tile(self, record: PunuoxiTileRecord) -> np.ndarray:
        if record.jpeg_length == 0:
            return np.full(
                (TILE_SIZE, TILE_SIZE, 3),
                self.base_info.background_color,
                dtype=np.uint8,
            )

        with self._lock:
            cached = self._tile_cache.get(record.index)
            if cached is not None:
                self._tile_cache.move_to_end(record.index)
                return cached
        blob = self._read_at(record.jpeg_offset, record.jpeg_length)

        try:
            with Image.open(BytesIO(blob)) as image:
                decoded = np.asarray(image.convert("RGB"), dtype=np.uint8)
        except Exception as exc:
            raise PunuoxiFormatError(f"IMAGE JPEG 瓦片解码失败: {record.index}") from exc

        if decoded.shape != (TILE_SIZE, TILE_SIZE, 3):
            normalized = np.full(
                (TILE_SIZE, TILE_SIZE, 3),
                self.base_info.background_color,
                dtype=np.uint8,
            )
            height = min(TILE_SIZE, decoded.shape[0])
            width = min(TILE_SIZE, decoded.shape[1])
            normalized[:height, :width] = decoded[:height, :width]
            decoded = normalized
        else:
            decoded = np.ascontiguousarray(decoded)

        # The JPEG payload stores each tile with its pixel axes exchanged.
        # This is independent of the column-major index order handled in
        # _parse_levels(): the index determines where a tile belongs, while
        # this transpose restores the orientation inside that tile.  It is a
        # lossless pixel permutation and performs no interpolation.
        decoded = np.ascontiguousarray(np.transpose(decoded, (1, 0, 2)))

        with self._lock:
            self._tile_cache[record.index] = decoded
            self._tile_cache.move_to_end(record.index)
            while len(self._tile_cache) > self.cache_size:
                self._tile_cache.popitem(last=False)
        return decoded

    def _encoded_level_tile(self, record: PunuoxiTileRecord) -> tuple[bytes | None, str | None]:
        if record.jpeg_length == 0:
            return None, None
        blob = self._read_at(record.jpeg_offset, record.jpeg_length)
        try:
            transformed, mode = transpose_jpeg(blob)
        except Exception as exc:
            raise PunuoxiFormatError(f"IMAGE JPEG 瓦片方向转换失败: {record.index}") from exc
        with self._lock:
            if self._native_tile_mode == "uninitialized":
                self._native_tile_mode = mode
            elif self._native_tile_mode != mode:
                self._native_tile_mode = "mixed"
        return transformed, mode

    def read_region(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        *,
        decode_workers: int | None = None,
    ) -> np.ndarray:
        del decode_workers
        if width <= 0 or height <= 0:
            return np.empty((max(0, height), max(0, width), 3), dtype=np.uint8)

        region = np.full(
            (height, width, 3),
            self.base_info.background_color,
            dtype=np.uint8,
        )
        request_x0 = max(0, x)
        request_y0 = max(0, y)
        request_x1 = min(self.width, x + width)
        request_y1 = min(self.height, y + height)
        if request_x0 >= request_x1 or request_y0 >= request_y1:
            return region

        level = self._levels[0]
        first_column = request_x0 // TILE_SIZE
        last_column = (request_x1 - 1) // TILE_SIZE
        first_row = request_y0 // TILE_SIZE
        last_row = (request_y1 - 1) // TILE_SIZE
        for row in range(first_row, last_row + 1):
            for column in range(first_column, last_column + 1):
                record = level.records[row * level.columns + column]
                tile = self._decode_tile(record)
                tile_x = column * TILE_SIZE
                tile_y = row * TILE_SIZE
                ix0 = max(request_x0, tile_x)
                iy0 = max(request_y0, tile_y)
                ix1 = min(request_x1, tile_x + TILE_SIZE)
                iy1 = min(request_y1, tile_y + TILE_SIZE)
                src_x0 = ix0 - tile_x
                src_y0 = iy0 - tile_y
                src_x1 = src_x0 + ix1 - ix0
                src_y1 = src_y0 + iy1 - iy0
                dst_x0 = ix0 - x
                dst_y0 = iy0 - y
                dst_x1 = dst_x0 + ix1 - ix0
                dst_y1 = dst_y0 + iy1 - iy0
                region[dst_y0:dst_y1, dst_x0:dst_x1] = tile[src_y0:src_y1, src_x0:src_x1]
        return region

    def read_level_region(
        self,
        level_index: int,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> np.ndarray:
        if level_index < 0 or level_index >= len(self._levels):
            raise IndexError(f"IMAGE 金字塔层索引越界: {level_index}")
        if width <= 0 or height <= 0:
            return np.empty((max(0, height), max(0, width), 3), dtype=np.uint8)

        level = self._levels[level_index]
        region = np.full(
            (height, width, 3),
            self.base_info.background_color,
            dtype=np.uint8,
        )
        request_x0 = max(0, x)
        request_y0 = max(0, y)
        request_x1 = min(level.width, x + width)
        request_y1 = min(level.height, y + height)
        if request_x0 >= request_x1 or request_y0 >= request_y1:
            return region

        first_column = request_x0 // TILE_SIZE
        last_column = (request_x1 - 1) // TILE_SIZE
        first_row = request_y0 // TILE_SIZE
        last_row = (request_y1 - 1) // TILE_SIZE
        for row in range(first_row, last_row + 1):
            for column in range(first_column, last_column + 1):
                record = level.records[row * level.columns + column]
                tile = self._decode_tile(record)
                tile_x = column * TILE_SIZE
                tile_y = row * TILE_SIZE
                ix0 = max(request_x0, tile_x)
                iy0 = max(request_y0, tile_y)
                ix1 = min(request_x1, tile_x + TILE_SIZE)
                iy1 = min(request_y1, tile_y + TILE_SIZE)
                src_x0 = ix0 - tile_x
                src_y0 = iy0 - tile_y
                src_x1 = src_x0 + ix1 - ix0
                src_y1 = src_y0 + iy1 - iy0
                dst_x0 = ix0 - x
                dst_y0 = iy0 - y
                dst_x1 = dst_x0 + ix1 - ix0
                dst_y1 = dst_y0 + iy1 - iy0
                region[dst_y0:dst_y1, dst_x0:dst_x1] = tile[src_y0:src_y1, src_x0:src_x1]
        return region

    def iter_native_level_jpegs(self, level_index: int):
        if level_index < 0 or level_index >= len(self._levels):
            raise IndexError(f"IMAGE 金字塔层索引越界: {level_index}")
        for record in self._levels[level_index].records:
            encoded, _mode = self._encoded_level_tile(record)
            yield encoded

    def iter_native_level_jpegs_parallel(
        self,
        level_index: int,
        *,
        workers: int,
        cancel_event=None,
    ):
        if level_index < 0 or level_index >= len(self._levels):
            raise IndexError(f"IMAGE 金字塔层索引越界: {level_index}")
        if workers <= 1:
            yield from self.iter_native_level_jpegs(level_index)
            return

        records = iter(self._levels[level_index].records)
        pending = deque()
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="image-jpeg-transpose") as executor:
            for _ in range(workers * 2):
                record = next(records, None)
                if record is None:
                    break
                pending.append(executor.submit(self._encoded_level_tile, record))

            while pending:
                if cancel_event is not None and cancel_event.is_set():
                    raise RuntimeError("转换已取消")
                encoded, _mode = pending.popleft().result()
                yield encoded
                record = next(records, None)
                if record is not None:
                    pending.append(executor.submit(self._encoded_level_tile, record))

    def _select_preview_level(self, max_long_side: int = 2048) -> PunuoxiLevel:
        for level in self._levels[1:]:
            if max(level.width, level.height) <= max_long_side:
                return level
        return self._levels[-1]

    def _stitch_level(self, level: PunuoxiLevel) -> Image.Image:
        canvas = Image.new(
            "RGB",
            level.dimensions,
            (self.base_info.background_color,) * 3,
        )
        for record in level.records:
            canvas.paste(
                Image.fromarray(self._decode_tile(record), mode="RGB"),
                (record.column * TILE_SIZE, record.row * TILE_SIZE),
            )
        return canvas

    def get_preview_image(self) -> Image.Image | None:
        if self._preview_image is None:
            self._preview_image = self._stitch_level(self._select_preview_level())
        return self._preview_image.copy()

    def get_thumbnail_image(self) -> Image.Image | None:
        if self._thumbnail_image is not None:
            return self._thumbnail_image.copy()
        native = self._read_rgb_resource("thumbnail")
        if native is not None:
            self._thumbnail_image = native
            return native.copy()
        self._thumbnail_image = self.get_preview_image()
        return self._thumbnail_image.copy() if self._thumbnail_image is not None else None

    def get_label_image(self) -> Image.Image | None:
        if self._label_image is None:
            self._label_image = self._read_rgb_resource("label")
        return self._label_image.copy() if self._label_image is not None else None

    def get_macro_image(self) -> Image.Image | None:
        if self._macro_image is None:
            self._macro_image = self._read_rgb_resource("macro")
        return self._macro_image.copy() if self._macro_image is not None else None

    def get_scan_metadata(self) -> dict[str, object]:
        mpp_x = self.mpp_x
        mpp_y = self.mpp_y
        return {
            "format": "punuoxi.image",
            "scanTime": self._scan_time,
            "deviceNo": self._device_no,
            "institution": self._institution,
            "caseNo": self._case_no,
            "physicalWidthUm": self._physical_width_um,
            "physicalHeightUm": self._physical_height_um,
            "mppX": mpp_x,
            "mppY": mpp_y,
            "mpp": (mpp_x + mpp_y) / 2.0,
            "width": self.width,
            "height": self.height,
            "levelCount": len(self._levels),
            "levelGrids": self.level_grids,
            "levelDimensions": self.level_dimensions,
            "backgroundColor": self.base_info.background_color,
            "nativeResourceStatus": self._native_resource_status,
            "nativeResourceReason": self._native_resource_reason,
            "nativeResources": {
                name: {"width": value[1], "height": value[2]} if value else None
                for name, value in self._native_resources.items()
            },
            "nativeTileMode": self._native_tile_mode,
            "tailIdentifier": self._tail_identifier,
        }
