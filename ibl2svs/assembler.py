from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from PIL import Image

from .reader import IBLSlide


@dataclass
class ProgressState:
    done: int = 0
    total: int = 0


class PILImageSource:
    def __init__(self, image: Image.Image):
        self.image = image.convert("RGB")
        self.width, self.height = self.image.size
        self.channels = 3
        self.background_color = 255

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.height, self.width, self.channels)

    def read_region(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        right = min(self.width, x + width)
        bottom = min(self.height, y + height)
        crop = self.image.crop((x, y, right, bottom))
        region = np.full((height, width, 3), 255, dtype=np.uint8)
        crop_arr = np.array(crop, dtype=np.uint8)
        region[: crop_arr.shape[0], : crop_arr.shape[1]] = crop_arr
        return region


class SlideSource:
    def __init__(self, slide: IBLSlide, decode_workers: int | None = None):
        self.slide = slide
        self.decode_workers = decode_workers
        self.width = slide.width
        self.height = slide.height
        self.channels = 3
        self.background_color = int(getattr(slide.base_info, "background_color", 255))

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.height, self.width, self.channels)

    def read_region(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        return self.slide.read_region(x, y, width, height, decode_workers=self.decode_workers)


class PyramidSource:
    def __init__(self, source: SlideSource, downsample: int):
        self.source = source
        self.downsample = downsample
        self.width = max(1, (source.width + downsample - 1) // downsample)
        self.height = max(1, (source.height + downsample - 1) // downsample)
        self.channels = 3
        self.background_color = getattr(source, "background_color", 255)

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.height, self.width, self.channels)

    def read_region(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        src = self.source.read_region(
            x * self.downsample,
            y * self.downsample,
            width * self.downsample,
            height * self.downsample,
        )
        image = Image.fromarray(src)
        resized = image.resize((width, height), resample=Image.Resampling.BILINEAR)
        return np.array(resized, dtype=np.uint8)


class NativeLevelSource:
    """Expose one native pyramid level through the regular region interface."""

    def __init__(self, slide, level_index: int):
        self.slide = slide
        self.level_index = level_index
        self.width, self.height = slide.level_dimensions[level_index]
        self.channels = 3
        self.background_color = int(getattr(slide.base_info, "background_color", 255))

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.height, self.width, self.channels)

    def read_region(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        return self.slide.read_level_region(self.level_index, x, y, width, height)


class ResizedSource:
    """Stream a source at a different canvas size without materializing it."""

    def __init__(self, source, width: int, height: int):
        self.source = source
        self.width = max(1, int(width))
        self.height = max(1, int(height))
        self.channels = 3
        self.background_color = getattr(source, "background_color", 255)

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.height, self.width, self.channels)

    def read_region(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        if width <= 0 or height <= 0:
            return np.empty((max(0, height), max(0, width), 3), dtype=np.uint8)

        sx0 = (max(0, x) * self.source.width) // self.width
        sy0 = (max(0, y) * self.source.height) // self.height
        sx1 = min(
            self.source.width,
            max(sx0 + 1, ((max(0, x) + width) * self.source.width + self.width - 1) // self.width),
        )
        sy1 = min(
            self.source.height,
            max(sy0 + 1, ((max(0, y) + height) * self.source.height + self.height - 1) // self.height),
        )
        source_region = self.source.read_region(sx0, sy0, sx1 - sx0, sy1 - sy0)
        resized = Image.fromarray(source_region).resize(
            (width, height),
            resample=Image.Resampling.BILINEAR,
        )
        return np.asarray(resized, dtype=np.uint8)


def tile_count(width: int, height: int, tile_size: int) -> int:
    tiles_x = (width + tile_size - 1) // tile_size
    tiles_y = (height + tile_size - 1) // tile_size
    return tiles_x * tiles_y


def _emit_tiles_from_region(
    region: np.ndarray,
    region_width: int,
    region_height: int,
    tile_size: int,
    progress: ProgressState | None,
    fill_value: int = 255,
):
    for tile_y in range(0, region_height, tile_size):
        tile_h = min(tile_size, region_height - tile_y)
        for tile_x in range(0, region_width, tile_size):
            tile_w = min(tile_size, region_width - tile_x)
            tile = region[tile_y : tile_y + tile_h, tile_x : tile_x + tile_w]
            if tile_h != tile_size or tile_w != tile_size:
                padded = np.full((tile_size, tile_size, 3), fill_value, dtype=np.uint8)
                padded[:tile_h, :tile_w] = tile
                tile = padded
            else:
                tile = np.ascontiguousarray(tile)

            if progress is not None:
                progress.done += 1
            yield tile


def _iter_chunk_tiles(
    source,
    tile_size: int,
    *,
    chunk_size: int,
    progress: ProgressState | None = None,
    cancel_event=None,
    stats: dict | None = None,
    memory_tracker=None,
    notify=None,
):
    total = tile_count(source.width, source.height, tile_size)
    if progress is not None:
        progress.total = total
        progress.done = 0
    if stats is not None:
        stats.setdefault("read_decode_sec", 0.0)

    # TIFF tile writers expect the iterator in global row-major tile order.
    # Emitting tiles chunk-by-chunk can scramble rows once chunk_size > tile_size.
    strip_height = max(tile_size, min(chunk_size, tile_size * 2))
    strip_height = max(tile_size, (strip_height // tile_size) * tile_size)

    for y in range(0, source.height, strip_height):
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("转换已取消")

        strip_h = min(strip_height, source.height - y)
        started = time.perf_counter()
        region = source.read_region(0, y, source.width, strip_h)
        if stats is not None:
            stats["read_decode_sec"] += time.perf_counter() - started
        if memory_tracker is not None:
            memory_tracker.sample()

        yield from _emit_tiles_from_region(
            region,
            source.width,
            strip_h,
            tile_size,
            progress,
            fill_value=getattr(source, "background_color", 255),
        )

        if notify is not None and progress is not None:
            notify(progress)
        if memory_tracker is not None:
            memory_tracker.sample()


def iter_slide_tiles(
    slide: IBLSlide,
    tile_size: int,
    *,
    downsample: int = 1,
    chunk_size: int = 2048,
    decode_workers: int | None = None,
    progress: ProgressState | None = None,
    cancel_event=None,
    stats: dict | None = None,
    memory_tracker=None,
    notify=None,
):
    source = SlideSource(slide, decode_workers=decode_workers)
    if downsample > 1:
        source = PyramidSource(source, downsample=downsample)
    yield from _iter_chunk_tiles(
        source,
        tile_size,
        chunk_size=chunk_size,
        progress=progress,
        cancel_event=cancel_event,
        stats=stats,
        memory_tracker=memory_tracker,
        notify=notify,
    )


def iter_source_tiles(
    source,
    tile_size: int,
    *,
    chunk_size: int = 2048,
    progress: ProgressState | None = None,
    cancel_event=None,
    stats: dict | None = None,
    memory_tracker=None,
    notify=None,
):
    yield from _iter_chunk_tiles(
        source,
        tile_size,
        chunk_size=chunk_size,
        progress=progress,
        cancel_event=cancel_event,
        stats=stats,
        memory_tracker=memory_tracker,
        notify=notify,
    )


class StripDownsampleDrive:
    """Read IBL strips, yield main-page tiles, and accumulate a 4x-downsampled buffer.

    Each strip is read once from *slide*, emitted tile-by-tile for the main
    page, and then downsampled into *level4_buffer* at the appropriate row
    range.  This lets the SVS writer build the first pyramid level in memory
    without a second pass or a temporary file.
    """

    def __init__(
        self,
        slide: IBLSlide,
        tile_size: int,
        strip_height: int,
        level4_buffer: "np.ndarray",
        progress: ProgressState | None = None,
        cancel_event=None,
        stats: dict | None = None,
        notify=None,
    ):
        self._slide = slide
        self._tile_size = tile_size
        self._strip_height = strip_height
        self._l4 = level4_buffer
        self._l4_h = level4_buffer.shape[0]
        self._l4_w = level4_buffer.shape[1]
        self._progress = progress
        self._cancel_event = cancel_event
        self._stats = stats if stats is not None else {}
        self._notify = notify

        total = tile_count(slide.width, slide.height, tile_size)
        if progress is not None:
            progress.total = total
            progress.done = 0
        self._stats.setdefault("read_decode_sec", 0.0)

    def __iter__(self):
        tile_size = self._tile_size
        w = self._slide.width
        h = self._slide.height
        l4_w = self._l4_w
        l4_h_max = self._l4_h

        for y in range(0, h, self._strip_height):
            if self._cancel_event is not None and self._cancel_event.is_set():
                raise RuntimeError("转换已取消")

            strip_h = min(self._strip_height, h - y)
            started = time.perf_counter()
            region = self._slide.read_region(0, y, w, strip_h)
            self._stats["read_decode_sec"] += time.perf_counter() - started

            # -- emit main-page tiles (row-major) --
            for ty in range(0, strip_h, tile_size):
                th = min(tile_size, strip_h - ty)
                for tx in range(0, w, tile_size):
                    tw = min(tile_size, w - tx)
                    tile = region[ty : ty + th, tx : tx + tw]
                    if th != tile_size or tw != tile_size:
                        padded = np.full(
                            (tile_size, tile_size, 3),
                            self._slide.base_info.background_color,
                            dtype=np.uint8,
                        )
                        padded[:th, :tw] = tile
                        tile = padded
                    else:
                        tile = np.ascontiguousarray(tile)

                    if self._progress is not None:
                        self._progress.done += 1
                    yield tile

            # -- downsample this strip into the 4x accumulation buffer --
            l4_y0 = y // 4
            l4_y1 = min(l4_h_max, (y + strip_h + 3) // 4)
            if l4_y1 > l4_y0:
                l4_strip_h = l4_y1 - l4_y0
                strip_pil = Image.fromarray(region)
                l4_strip = strip_pil.resize((l4_w, l4_strip_h), resample=Image.Resampling.BILINEAR)
                self._l4[l4_y0:l4_y1, :] = np.array(l4_strip, dtype=np.uint8)

            if self._notify is not None and self._progress is not None:
                self._notify(self._progress)


class DensePyramidDrive:
    """Read IBL strips, yield main-page tiles, and accumulate a downsampled buffer.

    Like *StripDownsampleDrive* but targets a 2× buffer by default for dense
    pyramid generation.  When the 2× buffer would exceed the *memory_budget*
    the class silently falls back to a 4× buffer so the cascade still fits in
    the available RAM.
    """

    def __init__(
        self,
        slide: IBLSlide,
        tile_size: int,
        strip_height: int,
        *,
        memory_budget_mb: int = 6144,
        progress: ProgressState | None = None,
        cancel_event=None,
        stats: dict | None = None,
        notify=None,
    ):
        self._slide = slide
        self._tile_size = tile_size
        self._strip_height = strip_height
        self._progress = progress
        self._cancel_event = cancel_event
        self._stats = stats if stats is not None else {}
        self._notify = notify

        # -- choose 2× or 4× accumulation target --
        w, h = slide.width, slide.height
        two_x_mb = (max(1, w // 2) * max(1, h // 2) * 3) / (1024 * 1024)
        four_x_mb = (max(1, w // 4) * max(1, h // 4) * 3) / (1024 * 1024)
        headroom_mb = memory_budget_mb * 0.55  # 55 % for the accumulation buffer

        if two_x_mb <= headroom_mb:
            self._ds = 2
            self._buf_w = max(1, w // 2)
            self._buf_h = max(1, h // 2)
        else:
            self._ds = 4
            self._buf_w = max(1, w // 4)
            self._buf_h = max(1, h // 4)

        self._buffer = np.zeros((self._buf_h, self._buf_w, 3), dtype=np.uint8)
        self.downsample_factor = self._ds

        total = tile_count(w, h, tile_size)
        if progress is not None:
            progress.total = total
            progress.done = 0
        self._stats.setdefault("read_decode_sec", 0.0)

    @property
    def accumulation_buffer(self) -> "np.ndarray":
        return self._buffer

    def __iter__(self):
        tile_size = self._tile_size
        w = self._slide.width
        h = self._slide.height
        ds = self._ds
        buf_w = self._buf_w
        buf_h_max = self._buf_h

        for y in range(0, h, self._strip_height):
            if self._cancel_event is not None and self._cancel_event.is_set():
                raise RuntimeError("转换已取消")

            strip_h = min(self._strip_height, h - y)
            started = time.perf_counter()
            region = self._slide.read_region(0, y, w, strip_h)
            self._stats["read_decode_sec"] += time.perf_counter() - started

            # -- emit main-page tiles --
            for ty in range(0, strip_h, tile_size):
                th = min(tile_size, strip_h - ty)
                for tx in range(0, w, tile_size):
                    tw = min(tile_size, w - tx)
                    tile = region[ty : ty + th, tx : tx + tw]
                    if th != tile_size or tw != tile_size:
                        padded = np.full(
                            (tile_size, tile_size, 3),
                            self._slide.base_info.background_color,
                            dtype=np.uint8,
                        )
                        padded[:th, :tw] = tile
                        tile = padded
                    else:
                        tile = np.ascontiguousarray(tile)

                    if self._progress is not None:
                        self._progress.done += 1
                    yield tile

            # -- downsample this strip into the accumulation buffer --
            buf_y0 = y // ds
            buf_y1 = min(buf_h_max, (y + strip_h + ds - 1) // ds)
            if buf_y1 > buf_y0:
                buf_strip_h = buf_y1 - buf_y0
                strip_pil = Image.fromarray(region)
                buf_strip = strip_pil.resize(
                    (buf_w, buf_strip_h), resample=Image.Resampling.BILINEAR
                )
                self._buffer[buf_y0:buf_y1, :] = np.array(buf_strip, dtype=np.uint8)

            if self._notify is not None and self._progress is not None:
                self._notify(self._progress)


def compute_pyramid_shapes(width: int, height: int, min_size: int = 512) -> list[tuple[int, int]]:
    shapes: list[tuple[int, int]] = []
    current_w = width
    current_h = height
    while min(current_w, current_h) >= min_size:
        current_w = max(1, (current_w + 1) // 2)
        current_h = max(1, (current_h + 1) // 2)
        shapes.append((current_w, current_h))
    return shapes


def iter_tiles(source, tile_size: int, progress: ProgressState | None = None, cancel_event=None):
    yield from iter_source_tiles(
        source,
        tile_size,
        progress=progress,
        cancel_event=cancel_event,
    )
