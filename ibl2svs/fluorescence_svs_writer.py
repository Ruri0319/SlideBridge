from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Iterator

import numpy as np

from .app_meta import APP_NAME, APP_VERSION
from .models import ConvertOptions
from .native_jpeg import iter_full_size_jpeg_tiles, select_viewer_compatible_levels


def _channel(slide, channel_index: int) -> dict[str, Any]:
    metadata = list(getattr(slide, "channel_metadata", []) or [])
    if channel_index < len(metadata):
        return dict(metadata[channel_index])
    return {
        "name": f"Channel {channel_index + 1}",
        "color": (255, 255, 255),
        "identity_source": "unknown",
    }


def _dye_name(channel: dict[str, Any], channel_index: int) -> str:
    if channel.get("identity_source") == "unknown":
        return f"C{channel_index + 1}"
    return str(channel.get("fluor") or channel.get("name") or f"C{channel_index + 1}")


def _display_color(channel: dict[str, Any]) -> int:
    red, green, blue = tuple(channel.get("color", (255, 255, 255)))
    return (int(red) << 16) | (int(green) << 8) | int(blue)


def build_fluorescence_description(slide, channel_index: int, options: ConvertOptions) -> str:
    channel = _channel(slide, channel_index)
    properties = [
        f"AppMag = {int(getattr(slide.base_info, 'max_zoom_rate', 0) or 0)}",
        f"MPP = {float(getattr(slide.base_info, 'mpp', 0.0) or 0.0):.6f}",
        f"Dye = {_dye_name(channel, channel_index)}",
        f"DisplayColor = {_display_color(channel)}",
        f"Filename = {slide.path.stem}",
        f"OriginalWidth = {slide.width}",
        f"OriginalHeight = {slide.height}",
    ]
    if channel.get("excitation_nm") is not None:
        properties.append(f"Excitation Wavelength = {float(channel['excitation_nm']):.6f}")
    if channel.get("emission_nm") is not None:
        properties.append(f"Emission Wavelength = {float(channel['emission_nm']):.6f}")
    if channel.get("exposure") is not None:
        properties.append(f"Exposure Time = {float(channel['exposure']):.6f}")
    return (
        "Aperio Image Library v12.0.0 \r\n"
        f"{slide.width}x{slide.height} [0,0 {slide.width}x{slide.height}] "
        f"(256x256) JPEG/Monochrome Q={options.main_quality}|"
        + "|".join(properties)
    )


def _level_description(slide, level_index: int) -> str:
    width, height = slide.level_dimensions[level_index]
    return (
        "Aperio Image Library v12.0.0 \r\n"
        f"{slide.width}x{slide.height} [0,0 {slide.width}x{slide.height}] "
        f"(256x256) -> {width}x{height} JPEG/Monochrome"
    )


def _encoded_tiles(
    slide,
    level_index: int,
    channel_index: int,
    blank_tile: bytes,
    cancel_event,
) -> Iterator[bytes]:
    iterator = getattr(slide, "iter_native_level_plane_jpegs")
    yield from iter_full_size_jpeg_tiles(
        iterator(level_index, channel_index, 0, 0, int(slide.native_fields[0])),
        tile_size=256,
        blank_tile=blank_tile,
        background=0,
        quality=100,
        cancel_event=cancel_event,
    )


def _decoded_tiles(
    slide,
    level_index: int,
    channel_index: int,
    quality: int,
    cancel_event,
) -> Iterator[bytes]:
    import imagecodecs

    width, height = slide.level_dimensions[level_index]
    for y in range(0, height, 256):
        for x in range(0, width, 256):
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("转换已取消")
            tile = np.asarray(
                slide.read_level_field_plane_region(
                    level_index,
                    int(slide.native_fields[0]),
                    channel_index,
                    0,
                    0,
                    x,
                    y,
                    min(256, width - x),
                    min(256, height - y),
                ),
                dtype=np.uint8,
            )
            if tile.shape != (256, 256):
                padded = np.zeros((256, 256), dtype=np.uint8)
                padded[: tile.shape[0], : tile.shape[1]] = tile
                tile = padded
            yield imagecodecs.jpeg8_encode(np.ascontiguousarray(tile), level=quality)


def _write_associated_page(tif, image, name: str, subfiletype: int) -> tuple[int, int]:
    array = np.asarray(image.convert("RGB" if image.mode not in {"1", "L"} else "L"))
    height, width = array.shape[:2]
    tif.write(
        array,
        photometric="rgb" if array.ndim == 3 else "minisblack",
        compression="lzw",
        rowsperstrip=max(1, min(height, 64)),
        description=f"Aperio Image Library v12.0.0 \n{name} {width}x{height}",
        metadata=None,
        software="",
        subfiletype=subfiletype,
    )
    return width, height


def validate_fluorescence_svs(path: str | Path, expected_levels: int) -> None:
    import tifffile

    with tifffile.TiffFile(str(path)) as tif:
        tiled = [page for page in tif.pages if page.is_tiled]
        if len(tiled) < expected_levels:
            raise RuntimeError("荧光 SVS 缺少原生金字塔页面")
        for page in tiled[:expected_levels]:
            if int(page.tilewidth) != 256 or int(page.tilelength) != 256:
                raise RuntimeError("荧光 SVS 瓦片尺寸不是 256×256")
            if int(getattr(page, "samplesperpixel", 1) or 1) != 1:
                raise RuntimeError("荧光 SVS 页面不是单通道")


def write_fluorescence_svs(
    slide,
    output_path: str | Path,
    options: ConvertOptions,
    perf: dict[str, Any],
    *,
    channel_index: int = 0,
    progress_callback=None,
    cancel_event=None,
) -> int:
    import imagecodecs
    import tifffile

    if getattr(slide, "modality", "unknown") != "fluorescence":
        raise RuntimeError("荧光 SVS 仅接受明确识别的荧光输入")
    if int(getattr(slide, "source_bit_depth", 0) or 0) != 8:
        raise RuntimeError("荧光 SVS 仅支持 8-bit；请改用 OME-TIFF")
    if len(tuple(getattr(slide, "native_fields", ()))) != 1:
        raise RuntimeError("荧光 SVS 仅支持 Field=1；请改用 OME-TIFF")
    if int(getattr(slide, "native_z_count", 1)) != 1 or int(getattr(slide, "native_t_count", 1)) != 1:
        raise RuntimeError("荧光 SVS 仅支持 Z=1、T=1；请改用 OME-TIFF")
    if not bool(getattr(slide, "supports_native_planes", False)):
        raise RuntimeError("输入缺少可独立读取的原生荧光通道平面")
    channel_count = int(getattr(slide, "native_channel_count", 1) or 1)
    if channel_index < 0 or channel_index >= channel_count:
        raise IndexError("荧光通道索引越界")

    levels = select_viewer_compatible_levels(list(slide.levels), 256)
    passthrough = (
        str(getattr(slide, "source_codec", "")).upper() == "JPEG"
        and bool(getattr(slide, "supports_plane_jpeg_passthrough", False))
        and callable(getattr(slide, "iter_native_level_plane_jpegs", None))
        and int(getattr(slide, "tile_size", 256)) == 256
    )
    blank_tile = imagecodecs.jpeg8_encode(np.zeros((256, 256), dtype=np.uint8), level=100)
    perf["native_path"] = True
    perf["native_level_dimensions"] = list(slide.level_dimensions)
    perf["native_resource_dimensions"] = dict(getattr(slide, "native_resource_dimensions", {}))
    perf["native_tile_mode"] = "jpeg_passthrough" if passthrough else "svs_reencoded"
    perf["level_dimensions"] = [level.dimensions for level in levels]
    perf["current_stage"] = "写出荧光 SVS"
    common = {
        "dtype": np.uint8,
        "photometric": "minisblack",
        "tile": (256, 256),
        "compression": "jpeg",
        "metadata": None,
        "software": "",
    }

    output_path = Path(output_path)
    started = time.perf_counter()
    with tifffile.TiffWriter(str(output_path), bigtiff=True) as tif:
        for level_index, level in enumerate(levels):
            width, height = level.dimensions
            tiles = (
                _encoded_tiles(slide, level_index, channel_index, blank_tile, cancel_event)
                if passthrough
                else _decoded_tiles(
                    slide,
                    level_index,
                    channel_index,
                    options.main_quality,
                    cancel_event,
                )
            )
            tif.write(
                data=tiles,
                shape=(height, width),
                description=(
                    build_fluorescence_description(slide, channel_index, options)
                    if level_index == 0
                    else _level_description(slide, level_index)
                ),
                subfiletype=0,
                **common,
            )
            if progress_callback is not None:
                progress_callback(
                    "写出荧光 SVS",
                    level_index + 1,
                    len(levels),
                    level_index + 1,
                    len(levels) + 3,
                )
            if level_index == 0:
                thumbnail_getter = getattr(slide, "get_thumbnail_image", None)
                thumbnail = thumbnail_getter() if callable(thumbnail_getter) else None
                if thumbnail is not None:
                    _write_associated_page(tif, thumbnail, "thumbnail", 1)

        label_getter = getattr(slide, "get_label_image", None)
        label = label_getter() if callable(label_getter) else None
        if label is not None:
            perf["svs_label_dimensions"] = _write_associated_page(tif, label, "label", 1)

        macro_getter = getattr(slide, "get_macro_image", None)
        macro = macro_getter() if callable(macro_getter) else None
        if macro is not None:
            perf["svs_macro_dimensions"] = _write_associated_page(tif, macro, "macro", 9)

    perf["main_write_sec"] += time.perf_counter() - started
    validate_fluorescence_svs(output_path, len(levels))
    perf["backend"] = "tifffile-fluorescence-svs"
    return len(levels)
