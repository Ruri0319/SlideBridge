from __future__ import annotations

from pathlib import Path
import re
import xml.etree.ElementTree as ET
from typing import Any

from .fluorescence_svs_writer import write_fluorescence_svs
from .models import ConvertOptions


def _safe_component(value: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", value.strip())
    cleaned = re.sub(r"\s+", "_", cleaned).strip("._")
    return cleaned


def _channel_suffix(metadata: dict[str, Any], index: int) -> str:
    prefix = f"C{index + 1:02d}"
    if metadata.get("identity_source") == "unknown":
        return prefix
    name = _safe_component(str(metadata.get("fluor") or metadata.get("name") or ""))
    return f"{prefix}_{name}" if name else prefix


def _candidate_files(base: Path, channel_metadata: list[dict[str, Any]]) -> tuple[Path, list[Path]]:
    afi = base.with_suffix(".afi")
    children = [
        base.with_name(f"{base.name}_{_channel_suffix(metadata, index)}.svs")
        for index, metadata in enumerate(channel_metadata)
    ]
    return afi, children


def reserve_afi_fileset(
    requested_path: str | Path,
    channel_metadata: list[dict[str, Any]],
) -> tuple[Path, list[Path]]:
    requested = Path(requested_path).with_suffix(".afi")
    base = requested.with_suffix("")
    index = 0
    while True:
        candidate_base = base if index == 0 else base.with_name(f"{base.name}_{index}")
        afi, children = _candidate_files(candidate_base, channel_metadata)
        temporary = [path.with_suffix(f"{path.suffix}.part") for path in [afi, *children]]
        if not any(path.exists() for path in [afi, *children, *temporary]):
            return afi, children
        index += 1


def _validate_geometry(slide) -> None:
    channel_count = int(getattr(slide, "native_channel_count", 1) or 1)
    fields = tuple(getattr(slide, "native_fields", ()))
    if len(fields) != 1:
        raise RuntimeError("AFI 仅支持 Field=1；请改用 OME-TIFF")
    if channel_count <= 1:
        return
    field_index = int(fields[0])
    levels = tuple(getattr(slide, "levels", ()))
    if not levels:
        raise RuntimeError("AFI 输入缺少原生金字塔层")
    for level in levels:
        expected = None
        for channel_index in range(channel_count):
            geometry = [
                (tile.x, tile.y, tile.width, tile.height)
                for tile in level.records
                if (
                    tile.field_index,
                    tile.channel_index,
                    tile.z_index,
                    tile.t_index,
                ) == (field_index, channel_index, 0, 0)
            ]
            if expected is None:
                expected = geometry
            elif geometry != expected:
                raise RuntimeError("AFI 各通道的层尺寸或瓦片网格不一致；请改用 OME-TIFF")


def write_afi(
    slide,
    requested_path: str | Path,
    options: ConvertOptions,
    perf: dict[str, Any],
    *,
    progress_callback=None,
    cancel_event=None,
) -> tuple[int, list[Path]]:
    if getattr(slide, "modality", "unknown") != "fluorescence":
        raise RuntimeError("AFI 仅接受明确识别的荧光输入")
    if int(getattr(slide, "source_bit_depth", 0) or 0) != 8:
        raise RuntimeError("AFI 仅支持 8-bit；请改用 OME-TIFF")
    if int(getattr(slide, "native_z_count", 1)) != 1 or int(getattr(slide, "native_t_count", 1)) != 1:
        raise RuntimeError("AFI 仅支持 Z=1、T=1；请改用 OME-TIFF")
    if not bool(getattr(slide, "supports_native_planes", False)):
        raise RuntimeError("输入缺少可独立读取的原生荧光通道平面")
    _validate_geometry(slide)

    channel_count = int(getattr(slide, "native_channel_count", 1) or 1)
    channel_metadata = list(getattr(slide, "channel_metadata", []) or [])
    if len(channel_metadata) != channel_count:
        raise RuntimeError("AFI 通道定义数量与源文件不一致")
    afi_path, child_paths = reserve_afi_fileset(requested_path, channel_metadata)
    afi_temp = afi_path.with_suffix(".afi.part")
    child_temps = [path.with_suffix(".svs.part") for path in child_paths]
    published: list[Path] = []
    written_levels = 0
    try:
        for channel_index, (temp_path, final_path) in enumerate(zip(child_temps, child_paths)):
            child_perf = {
                **perf,
                "main_write_sec": 0.0,
                "pyramid_sec": 0.0,
                "thumbnail_sec": 0.0,
            }

            def channel_progress(level, done, total, overall_done, overall_total):
                if progress_callback is not None:
                    progress_callback(
                        f"通道 {channel_index + 1}/{channel_count}: {level}",
                        done,
                        total,
                        channel_index * overall_total + overall_done,
                        channel_count * overall_total,
                    )

            written_levels = write_fluorescence_svs(
                slide,
                temp_path,
                options,
                child_perf,
                channel_index=channel_index,
                progress_callback=channel_progress,
                cancel_event=cancel_event,
            )
            perf["main_write_sec"] += float(child_perf.get("main_write_sec", 0.0))
            perf["pyramid_sec"] += float(child_perf.get("pyramid_sec", 0.0))
            perf["level_dimensions"] = list(child_perf.get("level_dimensions", []))

        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("转换已取消")
        for temp_path, final_path in zip(child_temps, child_paths):
            temp_path.replace(final_path)
            published.append(final_path)

        root = ET.Element("AFI")
        for child_path in child_paths:
            ET.SubElement(root, "Path").text = child_path.name
        ET.ElementTree(root).write(afi_temp, encoding="utf-8", xml_declaration=True)
        afi_temp.replace(afi_path)
        published.append(afi_path)
    except Exception:
        for path in [afi_temp, *child_temps, *published]:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        raise

    perf["backend"] = "tifffile-afi"
    perf["native_tile_mode"] = "jpeg_passthrough" if getattr(slide, "source_codec", "") == "JPEG" else "svs_reencoded"
    perf["output_files"] = [str(path) for path in [afi_path, *child_paths]]
    perf["primary_output_path"] = str(afi_path)
    return written_levels, [afi_path, *child_paths]
