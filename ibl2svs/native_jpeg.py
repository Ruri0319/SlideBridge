from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import Any

import numpy as np


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

        sof = payload.find(b"\xff\xc0")
        jpeg_height = int.from_bytes(payload[sof + 5 : sof + 7], "big")
        jpeg_width = int.from_bytes(payload[sof + 7 : sof + 9], "big")
        if (jpeg_width, jpeg_height) == (tile_size, tile_size):
            yield payload
            continue

        decoded = np.asarray(imagecodecs.jpeg8_decode(payload))
        padded_shape = (tile_size, tile_size, *decoded.shape[2:])
        padded = np.full(padded_shape, background, dtype=np.uint8)
        padded[: decoded.shape[0], : decoded.shape[1]] = decoded
        yield imagecodecs.jpeg8_encode(np.ascontiguousarray(padded), level=quality)
