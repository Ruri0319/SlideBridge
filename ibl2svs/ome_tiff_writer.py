from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Iterator
import xml.etree.ElementTree as ET

import numpy as np

from .app_meta import APP_NAME, APP_VERSION
from .models import ConvertOptions
from .native_jpeg import iter_full_size_jpeg_tiles, select_viewer_compatible_levels


def _ome_color(color: tuple[int, int, int]) -> int:
    rgba = (
        (int(color[0]) << 24)
        | (int(color[1]) << 16)
        | (int(color[2]) << 8)
        | 255
    )
    return rgba - (1 << 32) if rgba >= (1 << 31) else rgba


def _mpp_xy(slide) -> tuple[float, float]:
    mpp = float(getattr(slide.base_info, "mpp", 0.0) or 0.0)
    return (
        float(getattr(slide, "mpp_x", 0.0) or mpp),
        float(getattr(slide, "mpp_y", 0.0) or mpp),
    )


def _channel_metadata(slide) -> list[dict[str, Any]]:
    count = int(getattr(slide, "native_channel_count", 1) or 1)
    metadata = list(getattr(slide, "channel_metadata", []) or [])
    if len(metadata) != count:
        metadata = [
            {
                "name": f"Channel {index + 1}",
                "color": (255, 255, 255),
                "identity_source": "unknown",
            }
            for index in range(count)
        ]
    return metadata


def _field_metadata(slide, field_index: int) -> dict[str, Any]:
    channels = _channel_metadata(slide)
    mpp_x, mpp_y = _mpp_xy(slide)
    channel: list[dict[str, Any]] = []
    for index, item in enumerate(channels):
        entry: dict[str, Any] = {
            "Name": str(item.get("name") or f"Channel {index + 1}"),
            "Fluor": str(item.get("fluor") or ""),
            "Color": _ome_color(tuple(item.get("color", (255, 255, 255)))),
        }
        if item.get("excitation_nm") is not None:
            entry.update({"ExcitationWavelength": float(item["excitation_nm"]), "ExcitationWavelengthUnit": "nm"})
        if item.get("emission_nm") is not None:
            entry.update({"EmissionWavelength": float(item["emission_nm"]), "EmissionWavelengthUnit": "nm"})
        channel.append(entry)

    metadata: dict[str, Any] = {
        "axes": "TZCYX",
        "Name": f"{slide.path.stem} field {field_index}",
        "SignificantBits": int(getattr(slide, "source_bit_depth", 8) or 8),
        "Channel": channel,
    }
    if getattr(slide, "modality", "unknown") == "fluorescence":
        metadata["Description"] = "Modality=fluorescence"
    if mpp_x > 0:
        metadata.update({"PhysicalSizeX": mpp_x, "PhysicalSizeXUnit": "µm"})
    if mpp_y > 0:
        metadata.update({"PhysicalSizeY": mpp_y, "PhysicalSizeYUnit": "µm"})

    exposures = [item.get("exposure") for item in channels]
    if any(value is not None for value in exposures):
        metadata["Plane"] = [
            (
                {"ExposureTime": float(exposures[channel_index]), "ExposureTimeUnit": "s"}
                if exposures[channel_index] is not None
                else {}
            )
            for _t_index in range(int(getattr(slide, "native_t_count", 1) or 1))
            for _z_index in range(int(getattr(slide, "native_z_count", 1) or 1))
            for channel_index in range(len(channels))
        ]
    return metadata


def _encoded_plane_tiles(
    slide,
    level_index: int,
    field_index: int,
    tile_size: int,
    blank_tile: bytes,
    cancel_event,
) -> Iterator[bytes]:
    channel_count = int(getattr(slide, "native_channel_count", 1) or 1)
    z_count = int(getattr(slide, "native_z_count", 1) or 1)
    t_count = int(getattr(slide, "native_t_count", 1) or 1)
    iterator = getattr(slide, "iter_native_level_plane_jpegs")
    for t_index in range(t_count):
        for z_index in range(z_count):
            for channel_index in range(channel_count):
                yield from iter_full_size_jpeg_tiles(
                    iterator(
                        level_index,
                        channel_index,
                        z_index,
                        t_index,
                        field_index,
                    ),
                    tile_size=tile_size,
                    blank_tile=blank_tile,
                    background=0,
                    quality=100,
                    cancel_event=cancel_event,
                )


def _decoded_plane_tiles(
    slide,
    level_index: int,
    field_index: int,
    tile_size: int,
    dtype,
    cancel_event,
) -> Iterator[np.ndarray]:
    width, height = slide.level_dimensions[level_index]
    channel_count = int(getattr(slide, "native_channel_count", 1) or 1)
    z_count = int(getattr(slide, "native_z_count", 1) or 1)
    t_count = int(getattr(slide, "native_t_count", 1) or 1)
    for t_index in range(t_count):
        for z_index in range(z_count):
            for channel_index in range(channel_count):
                for y in range(0, height, tile_size):
                    for x in range(0, width, tile_size):
                        if cancel_event is not None and cancel_event.is_set():
                            raise RuntimeError("转换已取消")
                        tile = np.asarray(
                            slide.read_level_field_plane_region(
                                level_index,
                                field_index,
                                channel_index,
                                z_index,
                                t_index,
                                x,
                                y,
                                min(tile_size, width - x),
                                min(tile_size, height - y),
                            ),
                            dtype=dtype,
                        )
                        if tile.shape != (tile_size, tile_size):
                            padded = np.zeros((tile_size, tile_size), dtype=dtype)
                            padded[: tile.shape[0], : tile.shape[1]] = tile
                            tile = padded
                        yield np.ascontiguousarray(tile)


def _associated_images(slide):
    for name, getter_name in (
        ("thumbnail", "get_thumbnail_image"),
        ("macro", "get_macro_image"),
        ("label", "get_label_image"),
    ):
        getter = getattr(slide, getter_name, None)
        image = getter() if callable(getter) else None
        if image is not None:
            yield name, image


def validate_ome_tiff(path: str | Path) -> None:
    import tifffile

    with tifffile.TiffFile(str(path)) as tif:
        omexml = tif.ome_metadata
    if not omexml:
        raise RuntimeError("输出文件缺少 OME-XML")
    ET.fromstring(omexml)
    try:
        tifffile.OmeXml.validate(omexml)
    except ModuleNotFoundError as exc:
        if exc.name != "lxml":
            raise


def _write_fluorescence_ome(
    slide,
    output_path: Path,
    options: ConvertOptions,
    perf: dict[str, Any],
    *,
    progress_callback=None,
    cancel_event=None,
) -> int:
    import imagecodecs
    import tifffile

    fields = tuple(getattr(slide, "native_fields", (0,)))
    source_levels = list(getattr(slide, "levels", ()))
    levels = select_viewer_compatible_levels(source_levels, int(getattr(slide, "tile_size", 256) or 256))
    if not fields or not levels or not bool(getattr(slide, "supports_native_planes", False)):
        raise RuntimeError("荧光 OME-TIFF 需要可独立读取的原生通道平面")
    channel_count = int(getattr(slide, "native_channel_count", 1) or 1)
    z_count = int(getattr(slide, "native_z_count", 1) or 1)
    t_count = int(getattr(slide, "native_t_count", 1) or 1)
    bit_depth = int(getattr(slide, "source_bit_depth", 8) or 8)
    dtype = np.uint8 if bit_depth <= 8 else np.uint16
    tile_size = int(getattr(slide, "tile_size", options.resolved_tile_size()) or 256)
    passthrough = (
        bit_depth == 8
        and str(getattr(slide, "source_codec", "")).upper() == "JPEG"
        and bool(getattr(slide, "supports_plane_jpeg_passthrough", False))
        and callable(getattr(slide, "iter_native_level_plane_jpegs", None))
    )
    blank_tile = imagecodecs.jpeg8_encode(
        np.zeros((tile_size, tile_size), dtype=np.uint8),
        level=100,
    )
    perf["native_path"] = True
    perf["native_level_dimensions"] = list(slide.level_dimensions)
    perf["native_resource_dimensions"] = dict(getattr(slide, "native_resource_dimensions", {}))
    perf["native_tile_mode"] = "jpeg_passthrough" if passthrough else "lossless_decoded"
    perf["level_dimensions"] = [level.dimensions for level in levels]
    perf["current_stage"] = "写出荧光 OME-TIFF"

    with tifffile.TiffWriter(str(output_path), bigtiff=True, ome=True) as tif:
        for field_position, field_index in enumerate(fields):
            for level_index, level in enumerate(levels):
                width, height = level.dimensions
                shape = (t_count, z_count, channel_count, height, width)
                metadata = _field_metadata(slide, field_index) if level_index == 0 else None
                common = {
                    "shape": shape,
                    "dtype": dtype,
                    "photometric": "minisblack",
                    "tile": (tile_size, tile_size),
                    "metadata": metadata,
                    "software": f"{APP_NAME} {APP_VERSION}",
                    "subfiletype": 0 if level_index == 0 else 1,
                }
                started = time.perf_counter()
                if passthrough:
                    tif.write(
                        data=_encoded_plane_tiles(
                            slide,
                            level_index,
                            field_index,
                            tile_size,
                            blank_tile,
                            cancel_event,
                        ),
                        compression="jpeg",
                        subifds=len(levels) - 1 if level_index == 0 else None,
                        **common,
                    )
                else:
                    tif.write(
                        data=_decoded_plane_tiles(
                            slide,
                            level_index,
                            field_index,
                            tile_size,
                            dtype,
                            cancel_event,
                        ),
                        compression="deflate",
                        predictor=True,
                        subifds=len(levels) - 1 if level_index == 0 else None,
                        **common,
                    )
                elapsed = time.perf_counter() - started
                if level_index == 0:
                    perf["main_write_sec"] += elapsed
                else:
                    perf["pyramid_sec"] += elapsed
                if progress_callback is not None:
                    progress_callback(
                        "写出荧光 OME-TIFF",
                        field_position * len(levels) + level_index + 1,
                        len(fields) * len(levels),
                        field_position * len(levels) + level_index + 1,
                        len(fields) * len(levels) + 1,
                    )

        for name, image in _associated_images(slide):
            array = np.asarray(image.convert("RGB" if image.mode not in {"L", "I;16", "I"} else "L"))
            tif.write(
                array,
                photometric="rgb" if array.ndim == 3 else "minisblack",
                compression="lzw",
                rowsperstrip=max(1, min(array.shape[0], 64)),
                metadata={"axes": "YXS" if array.ndim == 3 else "YX", "Name": name},
                software=f"{APP_NAME} {APP_VERSION}",
            )

    validate_ome_tiff(output_path)
    return len(levels)


def write_ome_tiff(
    slide,
    output_path: str | Path,
    options: ConvertOptions,
    perf: dict[str, Any],
    *,
    progress_callback=None,
    cancel_event=None,
) -> int:
    output_path = Path(output_path)
    if (
        getattr(slide, "modality", "brightfield") == "fluorescence"
        or (
            getattr(slide, "source_container", "") == "ome_tiff"
            and bool(getattr(slide, "supports_native_planes", False))
        )
    ):
        return _write_fluorescence_ome(
            slide,
            output_path,
            options,
            perf,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )

    from .writer import _write_brightfield_ome_tiff_streaming

    levels = _write_brightfield_ome_tiff_streaming(
        slide,
        output_path,
        options,
        perf,
        progress_callback=progress_callback,
        cancel_event=cancel_event,
    )
    validate_ome_tiff(output_path)
    return levels
