from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from io import BytesIO
import mmap
from pathlib import Path
import re
import struct
import threading

import numpy as np
from PIL import Image

from .models import BaseInfo


HEADER_MIN_SIZE = 0xAC
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


class PunuoxiFormatError(RuntimeError):
    """Raised when a private Punuoxi ``.image`` container is invalid."""


@dataclass(frozen=True)
class PunuoxiTileRecord:
    index: int
    column: int
    row: int
    jpeg_length: int
    jpeg_offset: int
    reserved: int = 0


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


def _unpack_from(fmt: str, data: mmap.mmap | bytes, offset: int):
    size = struct.calcsize(fmt)
    if offset < 0 or offset + size > len(data):
        raise PunuoxiFormatError("IMAGE 文件头或索引被截断")
    return struct.unpack_from(fmt, data, offset)[0]


def _decode_header_text(data: mmap.mmap | bytes, offset: int, size: int) -> str:
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
        self._mmap: mmap.mmap | None = None
        self._lock = threading.RLock()
        self._tile_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._preview_image: Image.Image | None = None
        self._thumbnail_image: Image.Image | None = None

        try:
            self._file_size = self.path.stat().st_size
            if self._file_size < HEADER_MIN_SIZE + TAIL_SIZE:
                raise PunuoxiFormatError("IMAGE 文件过小，缺少固定头部或尾部标识")

            self._mmap = mmap.mmap(self._fh.fileno(), length=0, access=mmap.ACCESS_READ)
            self._parse_header()
            self._levels = self._parse_levels()
            if not self._levels:
                raise PunuoxiFormatError("IMAGE 未包含金字塔索引")

            self.width = self._header_width
            self.height = self._header_height
            self.channels = 3
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
            mapped = getattr(self, "_mmap", None)
            if mapped is not None:
                mapped.close()
                self._mmap = None
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
    def levels(self) -> tuple[PunuoxiLevel, ...]:
        return tuple(self._levels)

    @property
    def index_offset(self) -> int:
        return self._index_offset

    @property
    def tail_identifier(self) -> str:
        return self._tail_identifier

    def _parse_header(self) -> None:
        data = self._mmap
        if data is None:
            raise PunuoxiFormatError("IMAGE 文件未打开")

        tail = bytes(data[self._file_size - TAIL_SIZE : self._file_size])
        if not HEX_ID_RE.fullmatch(tail):
            raise PunuoxiFormatError("IMAGE 尾部 32 字节标识无效")
        self._tail_identifier = tail[:-1].decode("ascii").lower()

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
        data = self._mmap
        if data is None:
            raise PunuoxiFormatError("IMAGE 文件未打开")

        index_end = self._file_size - TAIL_SIZE
        cursor = self._index_offset
        levels: list[PunuoxiLevel] = []
        tile_index = 0
        while cursor < index_end:
            if len(levels) >= 32:
                raise PunuoxiFormatError("IMAGE 金字塔层数异常")
            if cursor + LEVEL_HEADER_SIZE > index_end:
                raise PunuoxiFormatError("IMAGE 金字塔层头被截断")

            columns = int(_unpack_from("<I", data, cursor))
            rows = int(_unpack_from("<I", data, cursor + 4))
            if columns <= 0 or rows <= 0:
                raise PunuoxiFormatError(f"IMAGE 第 {len(levels)} 层网格尺寸无效")

            record_count = columns * rows
            records_size = record_count * INDEX_RECORD_SIZE
            records_start = cursor + LEVEL_HEADER_SIZE
            records_end = records_start + records_size
            if records_end > index_end:
                raise PunuoxiFormatError(f"IMAGE 第 {len(levels)} 层索引被截断")

            physical_records: list[PunuoxiTileRecord] = []
            # The vendor stores the index column-major: all rows for column 0,
            # then all rows for column 1, and so on.  Keep the decoded records
            # in row-major coordinate order below so read_region() can use the
            # same coordinate lookup as the other slide sources.
            for column in range(columns):
                for row in range(rows):
                    physical_index = column * rows + row
                    offset = records_start + physical_index * INDEX_RECORD_SIZE
                    jpeg_length = int(_unpack_from("<I", data, offset))
                    jpeg_offset = int(_unpack_from("<I", data, offset + 4))
                    reserved = int(_unpack_from("<I", data, offset + 8))
                    if jpeg_length <= 0:
                        raise PunuoxiFormatError(
                            f"IMAGE 第 {len(levels)} 层瓦片 ({column},{row}) 长度无效"
                        )
                    if jpeg_offset < HEADER_MIN_SIZE or jpeg_offset + jpeg_length > self._index_offset:
                        raise PunuoxiFormatError(
                            f"IMAGE 第 {len(levels)} 层瓦片 ({column},{row}) 偏移无效"
                        )
                    if bytes(data[jpeg_offset : jpeg_offset + 3]) != b"\xff\xd8\xff":
                        raise PunuoxiFormatError(
                            f"IMAGE 第 {len(levels)} 层瓦片 ({column},{row}) 不是 JPEG"
                        )
                    if bytes(data[jpeg_offset + jpeg_length - 2 : jpeg_offset + jpeg_length]) != b"\xff\xd9":
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
                            reserved=reserved,
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

    def _build_base_info(self) -> BaseInfo:
        mpp_x = self._physical_width_um / self.width
        mpp_y = self._physical_height_um / self.height
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
        with self._lock:
            cached = self._tile_cache.get(record.index)
            if cached is not None:
                self._tile_cache.move_to_end(record.index)
                return cached
            data = self._mmap
            if data is None:
                raise PunuoxiFormatError("IMAGE 文件已关闭")
            blob = bytes(data[record.jpeg_offset : record.jpeg_offset + record.jpeg_length])

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
        if self._thumbnail_image is None:
            self._thumbnail_image = self.get_preview_image()
        return self._thumbnail_image.copy() if self._thumbnail_image is not None else None

    def get_label_image(self) -> Image.Image | None:
        return None

    def get_macro_image(self) -> Image.Image | None:
        return None

    def get_scan_metadata(self) -> dict[str, object]:
        mpp_x = self._physical_width_um / self.width
        mpp_y = self._physical_height_um / self.height
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
            "tailIdentifier": self._tail_identifier,
        }
