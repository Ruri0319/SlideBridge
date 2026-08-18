from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import Any

import numpy as np


SOF_MARKERS = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}


def jpeg_dimensions(payload: bytes, *, baseline_only: bool = True) -> tuple[int, int]:
    """Return JPEG width and height after parsing marker segment boundaries."""
    if len(payload) < 4 or payload[:2] != b"\xff\xd8":
        raise RuntimeError("JPEG 瓦片缺少 SOI 标记")
    position = 2
    while position < len(payload):
        if payload[position] != 0xFF:
            raise RuntimeError("JPEG marker 结构无效")
        while position < len(payload) and payload[position] == 0xFF:
            position += 1
        if position >= len(payload):
            break
        marker = payload[position]
        position += 1
        if marker in {0x01, 0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
            if marker == 0xD9:
                break
            continue
        if position + 2 > len(payload):
            raise RuntimeError("JPEG marker 长度被截断")
        segment_length = int.from_bytes(payload[position : position + 2], "big")
        if segment_length < 2 or position + segment_length > len(payload):
            raise RuntimeError("JPEG marker 长度无效")
        if marker in SOF_MARKERS:
            if baseline_only and marker != 0xC0:
                raise RuntimeError("JPEG 瓦片不是 8-bit baseline JPEG")
            if segment_length < 8:
                raise RuntimeError("JPEG SOF 段被截断")
            precision = payload[position + 2]
            height = int.from_bytes(payload[position + 3 : position + 5], "big")
            width = int.from_bytes(payload[position + 5 : position + 7], "big")
            if baseline_only and precision != 8:
                raise RuntimeError("JPEG 瓦片不是 8-bit baseline JPEG")
            if width <= 0 or height <= 0:
                raise RuntimeError("JPEG SOF 尺寸无效")
            return width, height
        if marker == 0xDA:
            break
        position += segment_length
    raise RuntimeError("JPEG 瓦片缺少 SOF 标记")


def select_viewer_compatible_levels(levels: Sequence[Any], tile_size: int) -> list[Any]:
    """Keep native levels through one statistically useful overview level."""
    selected: list[Any] = []
    overview_limit = tile_size * 2
    for level in levels:
        selected.append(level)
        width, height = level.dimensions
        if len(selected) > 1 and width <= overview_limit and height <= overview_limit:
            break
    return selected


def iter_full_size_jpeg_tiles(
    payloads: Iterable[bytes | None],
    *,
    tile_size: int,
    blank_tile: bytes,
    background: int,
    quality: int,
    cancel_event=None,
) -> Iterator[bytes]:
    """Pass through full JPEG tiles and pad encoded edge tiles to the TIFF tile size."""
    import imagecodecs

    for payload in payloads:
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("转换已取消")
        if payload is None:
            yield blank_tile
            continue

        jpeg_width, jpeg_height = jpeg_dimensions(payload)
        if (jpeg_width, jpeg_height) == (tile_size, tile_size):
            yield payload
            continue

        decoded = np.asarray(imagecodecs.jpeg8_decode(payload))
        padded_shape = (tile_size, tile_size, *decoded.shape[2:])
        padded = np.full(padded_shape, background, dtype=np.uint8)
        padded[: decoded.shape[0], : decoded.shape[1]] = decoded
        yield imagecodecs.jpeg8_encode(np.ascontiguousarray(padded), level=quality)
