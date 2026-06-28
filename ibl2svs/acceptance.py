from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _image_to_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _occupancy_mask(image: Image.Image, background: int, tolerance: int = 12) -> np.ndarray:
    array = _image_to_array(image)
    diff = np.abs(array.astype(np.int16) - int(background))
    return np.any(diff > tolerance, axis=2)


def _is_meaningful_resolution_tag(page) -> bool:
    x_tag = page.tags.get("XResolution")
    y_tag = page.tags.get("YResolution")
    unit_tag = page.tags.get("ResolutionUnit")
    if x_tag is None and y_tag is None and unit_tag is None:
        return False

    def _fraction_value(tag) -> tuple[int, int] | None:
        if tag is None:
            return None
        value = tag.value
        if isinstance(value, tuple) and len(value) == 2 and all(isinstance(v, int) for v in value):
            return value
        return None

    x_value = _fraction_value(x_tag)
    y_value = _fraction_value(y_tag)
    unit_value = unit_tag.value if unit_tag is not None else None

    # Treat the common writer residue (1/1 + NONE/no-unit) as non-meaningful.
    if x_value == (1, 1) and y_value == (1, 1) and unit_value in (None, 1, "NONE"):
        return False
    return True


def compare_preview_geometry(
    layer0: Image.Image,
    layer1: Image.Image,
    *,
    background: int,
    tolerance: int = 12,
    min_iou: float = 0.75,
    max_xor_ratio: float = 0.25,
) -> dict[str, Any]:
    layer0_mask = _occupancy_mask(layer0, background, tolerance=tolerance)
    layer1_mask = _occupancy_mask(layer1, background, tolerance=tolerance)
    if layer0_mask.shape != layer1_mask.shape:
        raise ValueError(
            f"preview shapes differ: {layer0_mask.shape!r} vs {layer1_mask.shape!r}"
        )

    intersection = np.logical_and(layer0_mask, layer1_mask).sum(dtype=np.int64)
    union = np.logical_or(layer0_mask, layer1_mask).sum(dtype=np.int64)
    xor = np.logical_xor(layer0_mask, layer1_mask).sum(dtype=np.int64)
    pixels0 = layer0_mask.sum(dtype=np.int64)
    pixels1 = layer1_mask.sum(dtype=np.int64)
    total = layer0_mask.size
    iou = float(intersection / union) if union else 1.0
    xor_ratio = float(xor / max(1, total))
    return {
        "width": int(layer0.width),
        "height": int(layer0.height),
        "occupied_layer0": int(pixels0),
        "occupied_layer1": int(pixels1),
        "intersection": int(intersection),
        "union": int(union),
        "xor_pixels": int(xor),
        "iou": iou,
        "xor_ratio": xor_ratio,
        "passed": iou >= min_iou and xor_ratio <= max_xor_ratio,
    }


def summarize_wsi_output(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    summary: dict[str, Any] = {
        "path": str(path),
        "name": path.name,
        "suffix": path.suffix.lower(),
        "exists": path.exists(),
    }
    if not path.exists():
        return summary

    summary["size_bytes"] = path.stat().st_size
    try:
        import tifffile
    except ImportError:
        summary["tifffile_available"] = False
        return summary

    summary["tifffile_available"] = True
    with tifffile.TiffFile(path) as tif:
        pages: list[dict[str, Any]] = []
        svs_thumbnail_dimensions = None
        svs_pyramid_dimensions: list[list[int]] = []
        svs_label_dimensions = None
        svs_macro_dimensions = None
        svs_extra_series_candidate_pages: list[int] = []
        svs_photometric_pages: list[str] = []
        svs_default_resolution_tags: list[int] = []
        for index, page in enumerate(tif.pages):
            shape = tuple(int(x) for x in page.shape)
            description = page.description or ""
            tile = None
            if getattr(page, "is_tiled", False):
                tile = [int(page.tilewidth), int(page.tilelength)]
            rowsperstrip = getattr(page, "rowsperstrip", None)
            photometric = getattr(page, "photometric", None)
            resolution_tag_present = _is_meaningful_resolution_tag(page)
            pages.append(
                {
                    "index": index,
                    "shape": list(shape),
                    "is_tiled": bool(getattr(page, "is_tiled", False)),
                    "tile": tile,
                    "rowsperstrip": int(rowsperstrip) if rowsperstrip else None,
                    "compression": int(page.compression) if getattr(page, "compression", None) else None,
                    "subfiletype": int(getattr(page, "subfiletype", 0) or 0),
                    "photometric": str(photometric.name if hasattr(photometric, "name") else photometric),
                    "has_default_resolution_tags": bool(resolution_tag_present),
                    "description": description[:240],
                }
            )
            if path.suffix.lower() == ".svs":
                svs_photometric_pages.append(str(photometric.name if hasattr(photometric, "name") else photometric))
                if resolution_tag_present:
                    svs_default_resolution_tags.append(index)
                if index == 1:
                    svs_thumbnail_dimensions = [shape[1], shape[0]]
                elif index >= 2 and bool(getattr(page, "is_tiled", False)):
                    svs_pyramid_dimensions.append([shape[1], shape[0]])
                elif "label " in description.lower():
                    svs_label_dimensions = [shape[1], shape[0]]
                elif "macro " in description.lower():
                    svs_macro_dimensions = [shape[1], shape[0]]
        summary["page_count"] = len(pages)
        summary["pages"] = pages
        if path.suffix.lower() == ".svs":
            if len(svs_pyramid_dimensions) > 2:
                svs_extra_series_candidate_pages = [
                    page["index"]
                    for page in pages[4:]
                    if page["is_tiled"]
                ]
            summary["svs_thumbnail_dimensions"] = svs_thumbnail_dimensions
            summary["svs_pyramid_dimensions"] = svs_pyramid_dimensions
            summary["svs_label_dimensions"] = svs_label_dimensions
            summary["svs_macro_dimensions"] = svs_macro_dimensions
            summary["svs_extra_series_candidate_pages"] = svs_extra_series_candidate_pages
            summary["svs_is_bigtiff"] = bool(tif.is_bigtiff)
            summary["svs_photometric_pages"] = svs_photometric_pages
            summary["svs_default_resolution_tag_pages"] = svs_default_resolution_tags

    if path.suffix.lower() == ".svs":
        try:
            proc = subprocess.run(
                ["tiffinfo", str(path)],
                capture_output=True,
                text=True,
                check=False,
            )
            summary["tiffinfo_available"] = proc.returncode == 0
            if proc.stdout:
                summary["tiffinfo_excerpt"] = proc.stdout[:2000]
        except Exception:
            summary["tiffinfo_available"] = False

    try:
        import openslide
    except ImportError:
        summary["openslide_available"] = False
        return summary

    summary["openslide_available"] = True
    slide = openslide.OpenSlide(str(path))
    try:
        summary["openslide_vendor"] = slide.properties.get(openslide.PROPERTY_NAME_VENDOR)
        summary["openslide_level_count"] = slide.level_count
        summary["openslide_level_dimensions"] = [
            [int(width), int(height)] for width, height in slide.level_dimensions
        ]
    finally:
        slide.close()
    return summary


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def render_acceptance_markdown(
    *,
    input_path: Path,
    tiff_result: dict[str, Any] | None,
    svs_result: dict[str, Any] | None,
    geometry_result: dict[str, Any],
    roi_exports: list[dict[str, Any]],
) -> str:
    def _result_block(title: str, result: dict[str, Any] | None) -> str:
        if result is None:
            return f"## {title}\n\n未执行。\n"
        lines = [
            f"## {title}",
            "",
            f"- 文件: `{result.get('path') or result.get('output_path') or ''}`",
            f"- 大小: `{result.get('size_bytes', '')}`",
            f"- 页数: `{result.get('page_count', '')}`",
            f"- OpenSlide vendor: `{result.get('openslide_vendor', '')}`",
            f"- Level 数: `{result.get('openslide_level_count', '')}`",
            f"- SVS thumbnail: `{result.get('svs_thumbnail_dimensions', '')}`",
            f"- SVS pyramid: `{result.get('svs_pyramid_dimensions', '')}`",
            f"- SVS label: `{result.get('svs_label_dimensions', '')}`",
            f"- SVS macro: `{result.get('svs_macro_dimensions', '')}`",
            f"- SVS extra series 候选页: `{result.get('svs_extra_series_candidate_pages', '')}`",
            f"- SVS BigTIFF: `{result.get('svs_is_bigtiff', '')}`",
            f"- SVS photometric: `{result.get('svs_photometric_pages', '')}`",
            f"- SVS 默认分辨率标签页: `{result.get('svs_default_resolution_tag_pages', '')}`",
            "",
            "### Windows 人工验收",
            "",
            "- QuPath 打开结果: ",
            "- OpenSlide 打开结果: ",
            "- 低倍全局是否正确: ",
            "- 接缝区域是否连续: ",
            "- 高倍细胞结构是否连续: ",
            "- 备注: ",
            "",
        ]
        return "\n".join(lines)

    roi_lines = "\n".join(
        f"- `{item['name']}`: `{item['path']}` @ ({item['x']}, {item['y']}, {item['width']}, {item['height']})"
        for item in roi_exports
    )
    return "\n".join(
        [
            "# 真实大文件视觉验收报告",
            "",
            f"- 输入文件: `{input_path}`",
            "",
            "## 几何门禁",
            "",
            f"- `layer0/layer1` 预览 IoU: `{geometry_result['iou']:.4f}`",
            f"- `layer0/layer1` 预览 XOR 比例: `{geometry_result['xor_ratio']:.4f}`",
            f"- 自动判定: `{'通过' if geometry_result['passed'] else '失败'}`",
            "",
            "## 参考 ROI",
            "",
            roi_lines or "- 无",
            "",
            _result_block("TIFF 验收", tiff_result),
            _result_block("SVS 验收", svs_result),
        ]
    )
