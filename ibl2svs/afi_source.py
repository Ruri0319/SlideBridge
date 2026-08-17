from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .tiff_source import TiffNativeLevel, TiffSlideSource


def read_afi_paths(path: str | Path) -> list[Path]:
    afi_path = Path(path)
    root = ET.parse(afi_path).getroot()
    if root.tag.rsplit("}", 1)[-1].upper() != "AFI":
        raise RuntimeError("AFI 文件缺少 AFI 根元素")
    references = [
        (element.text or "").strip()
        for element in root
        if element.tag.rsplit("}", 1)[-1] == "Path" and (element.text or "").strip()
    ]
    if not references:
        raise RuntimeError("AFI 文件没有通道 Path")
    paths: list[Path] = []
    for reference in references:
        child = Path(reference)
        paths.append(child if child.is_absolute() else afi_path.parent / child)
    return paths


class AfiSlideSource:
    """AFI channel set backed by one monochrome fluorescence SVS per channel."""

    def __init__(self, path: str | Path, cache_size: int = 64):
        self.path = Path(path)
        self._children: list[TiffSlideSource] = []
        try:
            for child_path in read_afi_paths(self.path):
                self._children.append(TiffSlideSource(child_path, cache_size=cache_size))
            self._configure()
        except Exception:
            self.close()
            raise

    def _configure(self) -> None:
        first = self._children[0]
        for child in self._children:
            if child.modality != "fluorescence" or child.native_channel_count != 1:
                raise RuntimeError("AFI 子文件必须是带 Dye 元数据的单通道荧光 SVS")
            if not child.supports_native_planes:
                raise RuntimeError("AFI 子文件缺少可独立读取的荧光平面")
            if child.level_dimensions != first.level_dimensions:
                raise RuntimeError("AFI 各通道的金字塔层尺寸不一致")
            if child.source_bit_depth != first.source_bit_depth or child.tile_size != first.tile_size:
                raise RuntimeError("AFI 各通道的位深或瓦片网格不一致")

        self.width = first.width
        self.height = first.height
        self.channels = 3
        self.base_info = first.base_info
        self.mpp_x = first.mpp_x
        self.mpp_y = first.mpp_y
        self.modality = "fluorescence"
        self.native_fields = (0,)
        self.native_channel_count = len(self._children)
        self.native_z_count = 1
        self.native_t_count = 1
        self.source_channel_count = self.native_channel_count
        self.source_bit_depth = first.source_bit_depth
        self.source_container = "afi"
        self.source_version = None
        codecs = {child.source_codec for child in self._children}
        self.source_codec = next(iter(codecs)) if len(codecs) == 1 else "mixed"
        self.native_axes = "TZCYX"
        self.compatibility_level = "static_unverified"
        self.tile_size = first.tile_size
        self.level_dimensions = list(first.level_dimensions)
        self.levels = [TiffNativeLevel(dimensions) for dimensions in self.level_dimensions]
        self.supports_native_pyramid = len(self.levels) > 1
        self.supports_native_planes = True
        self.supports_plane_jpeg_passthrough = all(
            child.supports_plane_jpeg_passthrough for child in self._children
        )
        self.channel_metadata = [dict(child.channel_metadata[0]) for child in self._children]
        self.native_resource_dimensions = {}

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.height, self.width, self.channels

    def close(self) -> None:
        for child in self._children:
            child.close()
        self._children.clear()

    def __enter__(self) -> "AfiSlideSource":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def read_level_field_plane_region(
        self,
        level_index: int,
        field_index: int,
        channel_index: int,
        z_index: int,
        t_index: int,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> np.ndarray:
        if field_index != 0:
            raise IndexError("AFI Field 索引越界")
        if channel_index < 0 or channel_index >= self.native_channel_count:
            raise IndexError("AFI 通道索引越界")
        return self._children[channel_index].read_level_field_plane_region(
            level_index,
            0,
            0,
            z_index,
            t_index,
            x,
            y,
            width,
            height,
        )

    def read_level_plane_region(
        self,
        level_index: int,
        channel_index: int,
        z_index: int,
        t_index: int,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> np.ndarray:
        return self.read_level_field_plane_region(
            level_index,
            0,
            channel_index,
            z_index,
            t_index,
            x,
            y,
            width,
            height,
        )

    def iter_native_level_plane_jpegs(
        self,
        level_index: int,
        channel_index: int,
        z_index: int,
        t_index: int,
        field_index: int | None = None,
    ):
        if field_index not in (None, 0):
            raise IndexError("AFI Field 索引越界")
        if channel_index < 0 or channel_index >= self.native_channel_count:
            raise IndexError("AFI 通道索引越界")
        yield from self._children[channel_index].iter_native_level_plane_jpegs(
            level_index,
            0,
            z_index,
            t_index,
            0,
        )

    def read_level_region(
        self,
        level_index: int,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> np.ndarray:
        composite = np.zeros((height, width, 3), dtype=np.uint8)
        for channel_index, metadata in enumerate(self.channel_metadata):
            plane = self.read_level_plane_region(
                level_index,
                channel_index,
                0,
                0,
                x,
                y,
                width,
                height,
            )
            if plane.dtype != np.uint8:
                plane = (plane.astype(np.uint32) * 255 // np.iinfo(plane.dtype).max).astype(np.uint8)
            color = np.asarray(metadata.get("color", (255, 255, 255)), dtype=np.uint16)
            colored = plane[..., None].astype(np.uint16) * color[None, None, :] // 255
            composite = np.maximum(composite, colored.astype(np.uint8))
        return composite

    def get_thumbnail_image(self):
        return self._children[0].get_thumbnail_image()

    def get_macro_image(self):
        return self._children[0].get_macro_image()

    def get_label_image(self):
        return self._children[0].get_label_image()

    def get_scan_metadata(self) -> dict[str, object]:
        return {
            "container": "afi",
            "channelCount": self.native_channel_count,
            "channelDefinitions": self.channel_metadata,
        }
