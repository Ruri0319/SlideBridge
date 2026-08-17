from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable
import xml.etree.ElementTree as ET

from .afi_source import AfiSlideSource, read_afi_paths
from .kfb_source import KfbSlideSource
from .models import (
    BatchInspection,
    ChannelDefinition,
    InputInspection,
    OutputFormat,
    SourceModality,
)
from .punuoxi_source import PunuoxiImageSource
from .reader import IBLSlide
from .tiff_source import TiffSlideSource


SUPPORTED_INPUT_SUFFIXES = {
    ".ibl",
    ".svs",
    ".tif",
    ".tiff",
    ".kfb",
    ".kfbl",
    ".kfbf",
    ".kfba",
    ".kfbx",
    ".image",
    ".afi",
}


class ChannelOverrideError(RuntimeError):
    diagnostic_code = "channel_override_mismatch"
    diagnostic_stage = "preflight"


def detect_input_format(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".ibl":
        return "ibl"
    if suffix == ".svs":
        return "svs"
    if suffix in {".tif", ".tiff"}:
        return "generic_tiff"
    if suffix in {".kfb", ".kfbl", ".kfbf", ".kfba", ".kfbx"}:
        return "kfb"
    if suffix == ".image":
        return "image"
    if suffix == ".afi":
        return "afi"
    return "unsupported"


def open_slide(path: str | Path, *, cache_size: int | None = None):
    path = Path(path)
    input_format = detect_input_format(path)
    if input_format == "ibl":
        return IBLSlide(path, cache_size=cache_size)
    if input_format == "kfb":
        return KfbSlideSource(path, cache_size=cache_size or 256)
    if input_format == "image":
        return PunuoxiImageSource(path, cache_size=cache_size or 256)
    if input_format == "afi":
        return AfiSlideSource(path, cache_size=cache_size or 64)
    if input_format in {"svs", "generic_tiff"}:
        return TiffSlideSource(path, cache_size=cache_size or 64)
    raise RuntimeError(f"不支持的输入格式: {path.suffix or path.name}")


def find_inspectable_files(input_dir: str | Path, recursive: bool = True) -> list[Path]:
    root = Path(input_dir)
    pattern = "**/*" if recursive else "*"
    candidates = sorted(
        path
        for path in root.glob(pattern)
        if path.is_file() and path.suffix.lower() in SUPPORTED_INPUT_SUFFIXES
    )
    referenced_children: set[Path] = set()
    for afi_path in (path for path in candidates if path.suffix.lower() == ".afi"):
        try:
            referenced_children.update(path.resolve() for path in read_afi_paths(afi_path))
        except (OSError, ET.ParseError, RuntimeError):
            continue
    return [path for path in candidates if path.resolve() not in referenced_children]


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _color(value: Any) -> tuple[int, int, int]:
    if isinstance(value, str):
        cleaned = value.strip().lstrip("#")
        if len(cleaned) == 6:
            return tuple(int(cleaned[index : index + 2], 16) for index in (0, 2, 4))  # type: ignore[return-value]
    if isinstance(value, (tuple, list)) and len(value) >= 3:
        return tuple(max(0, min(255, int(item))) for item in value[:3])  # type: ignore[return-value]
    return (255, 255, 255)


def channel_definitions(slide, overrides: list[dict[str, Any]] | None = None) -> list[ChannelDefinition]:
    if getattr(slide, "modality", "brightfield") != "fluorescence":
        if overrides:
            raise ChannelOverrideError("明场或未知模态文件不能应用荧光通道定义")
        return []
    count = int(getattr(slide, "native_channel_count", 0) or 0)
    if count <= 0:
        source_count = int(getattr(slide, "source_channel_count", getattr(slide, "channels", 3)) or 1)
        count = source_count if getattr(slide, "modality", "brightfield") == "fluorescence" else 0
    raw_metadata = list(getattr(slide, "channel_metadata", []) or [])
    override_by_index: dict[int, dict[str, Any]] = {}
    if overrides is not None:
        override_indexes = [int(item.get("index", index)) for index, item in enumerate(overrides)]
        if len(override_indexes) != count or set(override_indexes) != set(range(count)):
            raise ChannelOverrideError("通道覆盖定义与当前文件的通道结构不匹配，请重新预检")
        override_by_index = dict(zip(override_indexes, overrides))
    definitions: list[ChannelDefinition] = []
    for index in range(count):
        source = raw_metadata[index] if index < len(raw_metadata) else {}
        override = override_by_index.get(index)
        name = str(source.get("name") or "").strip()
        identity_source = str(source.get("identity_source") or ("source_metadata" if name else "unknown"))
        fluor = str(source.get("fluor") or name).strip() or None
        color = _color(source.get("color"))
        excitation_nm = _float_or_none(source.get("excitation_nm"))
        emission_nm = _float_or_none(source.get("emission_nm"))
        exposure = _float_or_none(source.get("exposure"))
        if override is not None:
            name = str(override.get("name", name) or "").strip()
            fluor = str(override.get("fluor", name) or "").strip() or None
            color = _color(override["color"]) if "color" in override else color
            excitation_nm = (
                _float_or_none(override.get("excitation_nm"))
                if "excitation_nm" in override
                else excitation_nm
            )
            emission_nm = (
                _float_or_none(override.get("emission_nm"))
                if "emission_nm" in override
                else emission_nm
            )
            exposure = _float_or_none(override.get("exposure")) if "exposure" in override else exposure
            identity_source = "user_supplied"
        if not name:
            name = f"Channel {index + 1}"
            fluor = None
            identity_source = "unknown"
        definitions.append(
            ChannelDefinition(
                index=index,
                name=name,
                fluor=fluor,
                color=color,
                excitation_nm=excitation_nm,
                emission_nm=emission_nm,
                exposure=exposure,
                identity_source=identity_source,  # type: ignore[arg-type]
            )
        )
    return definitions


def apply_channel_overrides(slide, overrides: list[dict[str, Any]] | None) -> list[ChannelDefinition]:
    definitions = channel_definitions(slide, overrides)
    slide.channel_definitions = definitions
    slide.channel_metadata = [asdict(item) for item in definitions]
    slide.channel_override_applied = bool(overrides)
    return definitions


def output_eligibility(slide) -> tuple[tuple[OutputFormat, ...], dict[str, str]]:
    modality: SourceModality = getattr(slide, "modality", "brightfield")
    fields = tuple(getattr(slide, "native_fields", (0,)))
    channels = int(getattr(slide, "native_channel_count", 1) or 1)
    z_count = int(getattr(slide, "native_z_count", 1) or 1)
    t_count = int(getattr(slide, "native_t_count", 1) or 1)
    bit_depth = int(getattr(slide, "source_bit_depth", 8) or 8)
    native_planes = bool(getattr(slide, "supports_native_planes", False))
    allowed: list[OutputFormat] = ["ome_tiff"]
    reasons: dict[str, str] = {}

    if modality == "brightfield":
        allowed.append("svs")
        reasons["fluorescence_svs"] = "输入不是荧光切片"
        reasons["afi"] = "输入不是荧光切片"
        return tuple(allowed), reasons

    reasons["svs"] = (
        "荧光输入不能写为明场 RGB SVS"
        if modality == "fluorescence"
        else "输入模态未知，不能写为明场 RGB SVS"
    )
    if modality != "fluorescence" or not native_planes:
        reasons["fluorescence_svs"] = "缺少可独立读取的原生荧光通道平面"
        reasons["afi"] = "缺少可独立读取的原生荧光通道平面"
        return tuple(allowed), reasons
    if bit_depth != 8:
        reasons["fluorescence_svs"] = "荧光 SVS 仅支持 8-bit；请改用 OME-TIFF"
        reasons["afi"] = "AFI 仅支持 8-bit；请改用 OME-TIFF"
        return tuple(allowed), reasons
    if len(fields) != 1 or z_count != 1 or t_count != 1:
        reason = "AFI/荧光 SVS 仅支持 Field=1、Z=1、T=1；请改用 OME-TIFF"
        reasons["fluorescence_svs"] = reason
        reasons["afi"] = reason
        return tuple(allowed), reasons
    allowed.append("afi")
    if channels == 1:
        allowed.append("fluorescence_svs")
    else:
        reasons["fluorescence_svs"] = "多通道荧光请输入 AFI；独立荧光 SVS 仅支持 C=1"
    return tuple(allowed), reasons


def inspect_file(path: str | Path) -> InputInspection:
    path = Path(path)
    input_format = detect_input_format(path)
    file_size = 0
    file_mtime_ns = 0
    try:
        stat = path.stat()
        file_size = stat.st_size
        file_mtime_ns = stat.st_mtime_ns
        with open_slide(path) as slide:
            definitions = tuple(channel_definitions(slide))
            allowed, reasons = output_eligibility(slide)
            return InputInspection(
                input_path=path,
                file_size=file_size,
                file_mtime_ns=file_mtime_ns,
                input_format=input_format,
                source_modality=getattr(slide, "modality", "brightfield"),
                source_container=getattr(slide, "source_container", None),
                source_version=getattr(slide, "source_version", None),
                source_codec=getattr(slide, "source_codec", None),
                source_bit_depth=int(getattr(slide, "source_bit_depth", 8) or 8),
                field_count=len(tuple(getattr(slide, "native_fields", (0,)))),
                channel_count=len(definitions),
                z_count=int(getattr(slide, "native_z_count", 1) or 1),
                t_count=int(getattr(slide, "native_t_count", 1) or 1),
                channel_definitions=definitions,
                allowed_output_formats=allowed,
                incompatible_reasons=reasons,
            )
    except Exception as exc:
        return InputInspection(
            input_path=path,
            file_size=file_size,
            file_mtime_ns=file_mtime_ns,
            input_format=input_format,
            source_modality="unknown",
            source_container=getattr(exc, "source_container", None),
            source_version=getattr(exc, "source_version", None),
            source_codec=None,
            source_bit_depth=0,
            field_count=0,
            channel_count=0,
            z_count=0,
            t_count=0,
            channel_definitions=(),
            allowed_output_formats=(),
            incompatible_reasons={},
            error=str(exc),
        )


def inspect_inputs(
    input_dir: str | Path,
    recursive: bool = True,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> BatchInspection:
    root = Path(input_dir)
    files = find_inspectable_files(root, recursive=recursive)
    inspected: list[InputInspection] = []
    for index, path in enumerate(files, start=1):
        if progress_callback is not None:
            progress_callback(index - 1, len(files), str(path))
        inspected.append(inspect_file(path))
    if progress_callback is not None:
        progress_callback(len(files), len(files), "")
    return BatchInspection(root, recursive, tuple(inspected))
