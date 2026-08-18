from __future__ import annotations

import io
import json
import sqlite3
from bisect import bisect_left, bisect_right
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path

import numpy as np
from PIL import Image

from .models import BaseInfo, BlockRecord


REQUIRED_TABLES = {"tbl_base_info", "tbl_img_info", "tbl_tile_info"}
OPTIONAL_TABLES = {"tbl_shrink_info"}
EXT_IMAGE_TYPES = {"macro": 1, "thumbnail": 2, "label": 3}


class IBLValidationError(RuntimeError):
    pass


class IBLSlide:
    def __init__(self, path: str | Path, cache_size: int | None = None):
        self.path = Path(path)
        self.cache_size = cache_size
        self._con = sqlite3.connect(str(self.path), check_same_thread=False)
        try:
            self._con.row_factory = sqlite3.Row
            self.tables = {
                row["name"]
                for row in self._con.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            self._block_cache: OrderedDict[int, np.ndarray] = OrderedDict()
            self.base_info = self._load_base_info()
            self.source_container = "ibl"
            self.source_version = self.base_info.version
            self.source_codec = "JPEG"
            self._associated_cache: dict[str, Image.Image] = {}
            self._ext_json_cache: dict[int, object] = {}
            self.modality = "brightfield"
            self.native_fields = (0,)
            self.native_channel_count = 1
            self.native_z_count = 1
            self.native_t_count = 1
            self.source_channel_count = 3
            self.source_bit_depth = 8
            self.channel_metadata = []
            self.supports_native_planes = False
            self.blocks = self._load_blocks()
            self.blocks_by_grid = {(block.grid_col, block.grid_row): block for block in self.blocks}
            self.blocks_by_id = {block.block_id: block for block in self.blocks}
            self.blocks_by_row: dict[int, list[BlockRecord]] = {}
            for block in self.blocks:
                self.blocks_by_row.setdefault(block.grid_row, []).append(block)
            for row_blocks in self.blocks_by_row.values():
                row_blocks.sort(key=lambda block: (block.x, block.grid_col))
            self.blocks_by_col: dict[int, list[BlockRecord]] = {}
            for block in self.blocks:
                self.blocks_by_col.setdefault(block.grid_col, []).append(block)
            for col_blocks in self.blocks_by_col.values():
                col_blocks.sort(key=lambda block: (block.y, block.grid_row))
            self._row_spatial_index = self._build_row_spatial_index()
            self._block_ownership = self._build_block_ownership()
            self.max_grid_col = max(block.grid_col for block in self.blocks)
            self.max_grid_row = max(block.grid_row for block in self.blocks)
            if self.cache_size is None:
                blocks_per_row = max(len(row_blocks) for row_blocks in self.blocks_by_row.values())
                self.cache_size = max(64, blocks_per_row * 2)
            self._validate_tile_integrity()
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        con = getattr(self, "_con", None)
        if con is not None:
            con.close()
            self._con = None

    def __enter__(self) -> "IBLSlide":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def width(self) -> int:
        return self.base_info.total_img_width

    @property
    def height(self) -> int:
        return self.base_info.total_img_height

    def _load_base_info(self) -> BaseInfo:
        missing = REQUIRED_TABLES - self.tables
        if missing:
            raise IBLValidationError(f"IBL 缺少必要数据表: {', '.join(sorted(missing))}")

        row = self._con.execute("SELECT * FROM tbl_base_info").fetchone()
        if row is None:
            raise IBLValidationError("tbl_base_info 为空")

        return BaseInfo(
            magic_no=row["magicNo"] or "",
            version=row["version"] or "",
            focus_num=int(row["focus_num"] or 0),
            image_format=int(row["image_format"] or 0),
            layer_size=int(row["layer_size"] or 0),
            img_color=int(row["img_color"] or 0),
            check_sum=int(row["check_sum"] or 0),
            ratio_step=int(row["ratio_step"] or 0),
            max_layer_size=int(row["max_Lay_size"] or 0),
            slide_type=int(row["slide_type"] or 0),
            background_color=int(row["bk_color"] or 255),
            pixel_size_mm=float(row["pixel_size"] or 0.0),
            total_img_num=int(row["total_img_num"] or 0),
            max_zoom_rate=int(row["max_zoom_rate"] or 0),
            img_col=int(row["img_col"] or 0),
            img_row=int(row["img_row"] or 0),
            img_width=int(row["img_width"] or 0),
            img_height=int(row["img_height"] or 0),
            tile_width=int(row["tile_width"] or 0),
            tile_height=int(row["tile_height"] or 0),
            shrink_tile_num=int(row["shrink_tile_num"] or 0),
            total_img_width=int(row["total_img_width"] or 0),
            total_img_height=int(row["total_img_height"] or 0),
        )

    def _load_blocks(self) -> list[BlockRecord]:
        rows = self._con.execute(
            """
            SELECT id, col, row, nX, nY
            FROM tbl_img_info
            ORDER BY row, col
            """
        ).fetchall()
        if not rows:
            raise IBLValidationError("tbl_img_info 为空")
        blocks = [
            BlockRecord(
                block_id=int(row["id"]),
                grid_col=int(row["col"]),
                grid_row=int(row["row"]),
                x=int(row["nX"]),
                y=int(row["nY"]),
                width=min(self.base_info.img_width, self.width - int(row["nX"])),
                height=min(self.base_info.img_height, self.height - int(row["nY"])),
                x1=min(self.width, int(row["nX"]) + self.base_info.img_width),
                y1=min(self.height, int(row["nY"]) + self.base_info.img_height),
            )
            for row in rows
        ]
        return blocks

    def _build_row_spatial_index(self) -> list[dict]:
        index: list[dict] = []
        for grid_row in sorted(self.blocks_by_row):
            blocks = self.blocks_by_row[grid_row]
            index.append(
                {
                    "grid_row": grid_row,
                    "y0": min(block.y for block in blocks),
                    "y1": max(block.y1 for block in blocks),
                    "blocks": blocks,
                    "starts": [block.x for block in blocks],
                    "ends": [block.x1 for block in blocks],
                }
            )
        return index

    @staticmethod
    def _neighbor_split(first_start: int, first_end: int, second_start: int, second_end: int) -> int:
        overlap_start = max(first_start, second_start)
        overlap_end = min(first_end, second_end)
        if overlap_start < overlap_end:
            return (overlap_start + overlap_end) // 2
        if first_start <= second_start:
            return (first_end + second_start) // 2
        return (second_end + first_start) // 2

    def _build_block_ownership(self) -> dict[int, tuple[int, int, int, int]]:
        x_bounds: dict[int, tuple[int, int]] = {}
        y_bounds: dict[int, tuple[int, int]] = {}

        for row_blocks in self.blocks_by_row.values():
            for index, block in enumerate(row_blocks):
                if index == 0:
                    owner_x0 = block.x
                else:
                    prev_block = row_blocks[index - 1]
                    owner_x0 = self._neighbor_split(prev_block.x, prev_block.x1, block.x, block.x1)
                if index == len(row_blocks) - 1:
                    owner_x1 = block.x1
                else:
                    next_block = row_blocks[index + 1]
                    owner_x1 = self._neighbor_split(block.x, block.x1, next_block.x, next_block.x1)
                owner_x0 = max(block.x, min(block.x1, owner_x0))
                owner_x1 = max(owner_x0, min(block.x1, owner_x1))
                x_bounds[block.block_id] = (owner_x0, owner_x1)

        for col_blocks in self.blocks_by_col.values():
            for index, block in enumerate(col_blocks):
                if index == 0:
                    owner_y0 = block.y
                else:
                    prev_block = col_blocks[index - 1]
                    owner_y0 = self._neighbor_split(prev_block.y, prev_block.y1, block.y, block.y1)
                if index == len(col_blocks) - 1:
                    owner_y1 = block.y1
                else:
                    next_block = col_blocks[index + 1]
                    owner_y1 = self._neighbor_split(block.y, block.y1, next_block.y, next_block.y1)
                owner_y0 = max(block.y, min(block.y1, owner_y0))
                owner_y1 = max(owner_y0, min(block.y1, owner_y1))
                y_bounds[block.block_id] = (owner_y0, owner_y1)

        return {
            block.block_id: (
                x_bounds[block.block_id][0],
                y_bounds[block.block_id][0],
                x_bounds[block.block_id][1],
                y_bounds[block.block_id][1],
            )
            for block in self.blocks
        }

    def _expected_full_resolution_tile_count(self) -> int:
        columns = (self.base_info.img_width + self.base_info.tile_width - 1) // self.base_info.tile_width
        rows = (self.base_info.img_height + self.base_info.tile_height - 1) // self.base_info.tile_height
        return columns * rows

    def _validate_tile_integrity(self) -> None:
        counts = self._con.execute(
            """
            SELECT id, COUNT(*) AS n
            FROM tbl_tile_info
            WHERE layer = 0
            GROUP BY id
            """
        ).fetchall()
        if not counts:
            raise IBLValidationError("tbl_tile_info 中没有全分辨率 layer=0 数据")

        expected_tiles = self._expected_full_resolution_tile_count()
        counts_by_id = {int(row["id"]): int(row["n"]) for row in counts}
        bad = [
            block.block_id
            for block in self.blocks
            if counts_by_id.get(block.block_id, 0) != expected_tiles
        ]
        if bad:
            raise IBLValidationError(
                f"存在不完整的全分辨率块，期望每块 {expected_tiles} 个子瓦片，示例 block_id={bad[0]}"
            )

    def _decode_block_from_rows(self, block: BlockRecord, rows) -> np.ndarray:
        expected_tiles = self._expected_full_resolution_tile_count()
        if len(rows) != expected_tiles:
            raise IBLValidationError(
                f"block_id={block.block_id} 的全分辨率子瓦片数量不是 {expected_tiles}"
            )

        canvas = np.full(
            (self.base_info.img_height, self.base_info.img_width, 3),
            fill_value=self.base_info.background_color,
            dtype=np.uint8,
        )
        for row in rows:
            col = int(row["col"])
            grid_row = int(row["row"])
            tile = self._decode_jpeg(row["data"])
            x0 = col * self.base_info.tile_width
            y0 = grid_row * self.base_info.tile_height
            y1 = min(y0 + tile.shape[0], canvas.shape[0])
            x1 = min(x0 + tile.shape[1], canvas.shape[1])
            canvas[y0:y1, x0:x1] = tile[: y1 - y0, : x1 - x0]

        width, height = self._trimmed_block_shape(block)
        return canvas[:height, :width]

    def prefetch_block_row(self, grid_row: int) -> list[tuple[BlockRecord, np.ndarray]]:
        blocks = self.blocks_by_row.get(grid_row, [])
        if not blocks:
            return []
        return self.prefetch_blocks(blocks)

    def _fetch_tile_rows_for_block_ids(self, block_ids: list[int]) -> dict[int, list[sqlite3.Row]]:
        if not block_ids:
            return {}
        placeholders = ",".join("?" for _ in block_ids)
        rows = self._con.execute(
            f"""
            SELECT id, col, row, data
            FROM tbl_tile_info
            WHERE layer = 0 AND id IN ({placeholders})
            ORDER BY id, row, col
            """,
            block_ids,
        ).fetchall()
        grouped: dict[int, list[sqlite3.Row]] = {}
        for row in rows:
            grouped.setdefault(int(row["id"]), []).append(row)
        return grouped

    def _cache_block_array(self, block_id: int, array: np.ndarray) -> None:
        self._block_cache[block_id] = array
        self._block_cache.move_to_end(block_id)
        while len(self._block_cache) > (self.cache_size or 0):
            self._block_cache.popitem(last=False)

    def _resolved_decode_workers(self, requested: int | None = None) -> int:
        if requested is not None:
            return max(1, requested)
        return max(1, min(4, os.cpu_count() or 4))

    def prefetch_blocks(
        self,
        blocks: list[BlockRecord],
        *,
        decode_workers: int | None = None,
    ) -> list[tuple[BlockRecord, np.ndarray]]:
        if not blocks:
            return []

        missing_blocks = [block for block in blocks if block.block_id not in self._block_cache]
        if missing_blocks:
            grouped = self._fetch_tile_rows_for_block_ids([block.block_id for block in missing_blocks])
            workers = self._resolved_decode_workers(decode_workers)
            if workers > 1 and len(missing_blocks) > 1:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    decoded = list(
                        pool.map(
                            lambda block: (block.block_id, self._decode_block_from_rows(block, grouped.get(block.block_id, []))),
                            missing_blocks,
                        )
                    )
                for block_id, array in decoded:
                    self._cache_block_array(block_id, array)
            else:
                for block in missing_blocks:
                    self._cache_block_array(
                        block.block_id,
                        self._decode_block_from_rows(block, grouped.get(block.block_id, [])),
                    )

        result = []
        for block in blocks:
            array = self._block_cache.get(block.block_id)
            if array is None:
                array = self.get_block_array(block)
            else:
                self._block_cache.move_to_end(block.block_id)
            result.append((block, array))
        return result

    def _decode_jpeg(self, blob: bytes) -> np.ndarray:
        with Image.open(io.BytesIO(blob)) as image:
            rgb = image.convert("RGB")
            return np.array(rgb, dtype=np.uint8)

    def _trimmed_block_shape(self, block: BlockRecord) -> tuple[int, int]:
        return block.width, block.height

    def _block_owner_bounds(self, block: BlockRecord) -> tuple[int, int, int, int]:
        return self._block_ownership[block.block_id]

    def blocks_for_region(self, x: int, y: int, width: int, height: int) -> list[BlockRecord]:
        if width <= 0 or height <= 0:
            return []
        x1 = x + width
        y1 = y + height
        blocks: list[BlockRecord] = []
        for row_info in self._row_spatial_index:
            if row_info["y1"] <= y or row_info["y0"] >= y1:
                continue
            starts = row_info["starts"]
            ends = row_info["ends"]
            blocks_in_row = row_info["blocks"]
            start_index = bisect_right(ends, x)
            end_index = bisect_left(starts, x1)
            for block in blocks_in_row[start_index:end_index]:
                if block.x1 <= x or block.x >= x1 or block.y1 <= y or block.y >= y1:
                    continue
                blocks.append(block)
        return blocks

    def read_region(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        *,
        decode_workers: int | None = None,
    ) -> np.ndarray:
        region = np.full((height, width, 3), self.base_info.background_color, dtype=np.uint8)
        blocks = self.blocks_for_region(x, y, width, height)
        for block, block_array in self.prefetch_blocks(blocks, decode_workers=decode_workers):
            bx0, by0, bx1, by1 = self._block_owner_bounds(block)

            ix0 = max(x, bx0)
            iy0 = max(y, by0)
            ix1 = min(x + width, bx1)
            iy1 = min(y + height, by1)
            if ix0 >= ix1 or iy0 >= iy1:
                continue

            src_x0 = ix0 - block.x
            src_y0 = iy0 - block.y
            src_x1 = ix1 - block.x
            src_y1 = iy1 - block.y

            dst_x0 = ix0 - x
            dst_y0 = iy0 - y
            dst_x1 = ix1 - x
            dst_y1 = iy1 - y
            region[dst_y0:dst_y1, dst_x0:dst_x1] = block_array[src_y0:src_y1, src_x0:src_x1]
        return region

    def get_block_array(self, block: BlockRecord | int) -> np.ndarray:
        block_id = block.block_id if isinstance(block, BlockRecord) else int(block)
        cached = self._block_cache.get(block_id)
        if cached is not None:
            self._block_cache.move_to_end(block_id)
            return cached

        rows = self._con.execute(
            """
            SELECT col, row, data
            FROM tbl_tile_info
            WHERE layer = 0 AND id = ?
            ORDER BY row, col
            """,
            (block_id,),
        ).fetchall()
        block_record = self.blocks_by_id[block_id]
        trimmed = self._decode_block_from_rows(block_record, rows)

        self._cache_block_array(block_id, trimmed)
        return trimmed

    def get_preview_image(self) -> Image.Image | None:
        return self.assemble_preview_from_layer1()

    def _available_preview_layer(self) -> int | None:
        row = self._con.execute(
            "SELECT MIN(layer) AS layer FROM tbl_tile_info WHERE layer > 0"
        ).fetchone()
        if row is None or row["layer"] is None:
            return None
        return int(row["layer"])

    def _layer_scale(self, layer: int) -> int:
        return int(self.base_info.ratio_step) ** int(layer)

    def assemble_preview_from_layer1(self, scale: int | None = None) -> Image.Image | None:
        layer = self._available_preview_layer()
        if layer is None:
            return None

        source_scale = self._layer_scale(layer)
        output_scale = source_scale if scale is None else int(scale)
        rows = self._con.execute(
            """
            SELECT i.nX, i.nY, t.data
            FROM tbl_img_info AS i
            JOIN tbl_tile_info AS t
              ON t.id = i.id AND t.layer = ?
            ORDER BY i.row, i.col
            """,
            (layer,),
        ).fetchall()
        if not rows:
            return None

        width = (self.width + output_scale - 1) // output_scale
        height = (self.height + output_scale - 1) // output_scale
        background = (self.base_info.background_color,) * 3
        canvas = Image.new("RGB", (width, height), background)
        for row in rows:
            with Image.open(io.BytesIO(row["data"])) as tile_image:
                tile = tile_image.convert("RGB")
            if output_scale != source_scale:
                target_width = max(1, (tile.width * source_scale + output_scale - 1) // output_scale)
                target_height = max(1, (tile.height * source_scale + output_scale - 1) // output_scale)
                tile = tile.resize((target_width, target_height), resample=Image.Resampling.BILINEAR)
            canvas.paste(tile, (int(row["nX"]) // output_scale, int(row["nY"]) // output_scale))
        return canvas

    def assemble_preview_from_layer0(self, scale: int = 4) -> Image.Image:
        width = (self.width + scale - 1) // scale
        height = (self.height + scale - 1) // scale
        canvas = Image.new("RGB", (width, height), (self.base_info.background_color,) * 3)
        for block in self.blocks:
            owner_x0, owner_y0, owner_x1, owner_y1 = self._block_owner_bounds(block)
            if owner_x1 <= owner_x0 or owner_y1 <= owner_y0:
                continue
            block_array = self.get_block_array(block)
            src_x0 = owner_x0 - block.x
            src_y0 = owner_y0 - block.y
            src_x1 = owner_x1 - block.x
            src_y1 = owner_y1 - block.y
            crop = block_array[src_y0:src_y1, src_x0:src_x1]
            if crop.size == 0:
                continue
            preview_x0 = owner_x0 // scale
            preview_y0 = owner_y0 // scale
            preview_x1 = max(preview_x0 + 1, (owner_x1 + scale - 1) // scale)
            preview_y1 = max(preview_y0 + 1, (owner_y1 + scale - 1) // scale)
            image = Image.fromarray(crop).resize(
                (preview_x1 - preview_x0, preview_y1 - preview_y0),
                resample=Image.Resampling.BILINEAR,
            )
            canvas.paste(image, (preview_x0, preview_y0))
        return canvas

    def debug_export_preview(self, path: str | Path, *, source: str = "layer1", scale: int = 4) -> Path:
        output_path = Path(path)
        if source == "layer1":
            image = self.assemble_preview_from_layer1(scale=scale)
        elif source == "layer0":
            image = self.assemble_preview_from_layer0(scale=scale)
        else:
            raise ValueError(f"unsupported preview source: {source}")
        if image is None:
            raise RuntimeError(f"preview source unavailable: {source}")
        image.save(output_path)
        return output_path

    def debug_export_roi(
        self,
        path: str | Path,
        *,
        x: int,
        y: int,
        width: int,
        height: int,
        decode_workers: int | None = 1,
    ) -> Path:
        output_path = Path(path)
        image = Image.fromarray(
            self.read_region(x, y, width, height, decode_workers=decode_workers),
        )
        image.save(output_path)
        return output_path

    def _get_ext_blob(self, type_id: int) -> bytes | None:
        if "tbl_ext_info" not in self.tables:
            return None
        row = self._con.execute(
            "SELECT data FROM tbl_ext_info WHERE type = ? ORDER BY id LIMIT 1",
            (type_id,),
        ).fetchone()
        if row is None or row["data"] is None:
            return None
        return bytes(row["data"])

    def _get_ext_image(self, type_id: int) -> Image.Image | None:
        image_name = next(
            (name for name, mapped_type in EXT_IMAGE_TYPES.items() if mapped_type == type_id),
            None,
        )
        if image_name is not None:
            cached = self._associated_cache.get(image_name)
            if cached is not None:
                return cached.copy()

        blob = self._get_ext_blob(type_id)
        if blob is None:
            return None
        with Image.open(io.BytesIO(blob)) as image:
            decoded = image.convert("RGB")
        if image_name is not None:
            self._associated_cache[image_name] = decoded
        return decoded.copy()

    def _get_ext_json(self, type_id: int) -> object | None:
        if type_id in self._ext_json_cache:
            return self._ext_json_cache[type_id]
        blob = self._get_ext_blob(type_id)
        if blob is None:
            return None
        value = json.loads(blob.decode("utf-8"))
        self._ext_json_cache[type_id] = value
        return value

    def get_native_associated_image(self, name: str) -> Image.Image | None:
        """Return an IBL image from tbl_ext_info without synthetic resizing."""
        return self._get_ext_image(EXT_IMAGE_TYPES[name])

    def get_macro_image(self) -> Image.Image | None:
        return self._get_ext_image(EXT_IMAGE_TYPES["macro"])

    def get_thumbnail_image(self) -> Image.Image | None:
        native = self._get_ext_image(EXT_IMAGE_TYPES["thumbnail"])
        if native is not None:
            return native
        if "tbl_shrink_info" not in self.tables:
            return None

        layer_row = self._con.execute(
            "SELECT MAX(layerNo) AS layerNo FROM tbl_shrink_info"
        ).fetchone()
        if layer_row is None or layer_row["layerNo"] is None:
            return None
        layer_no = int(layer_row["layerNo"])

        rows = self._con.execute(
            """
            SELECT x, y, data
            FROM tbl_shrink_info
            WHERE layerNo = ?
            ORDER BY y, x
            """,
            (layer_no,),
        ).fetchall()
        if not rows:
            return None

        scale = self._layer_scale(layer_no)
        width = (self.width + scale - 1) // scale
        height = (self.height + scale - 1) // scale
        canvas = Image.new("RGB", (width, height), (self.base_info.background_color,) * 3)
        for row in rows:
            with Image.open(io.BytesIO(row["data"])) as tile_image:
                tile = tile_image.convert("RGB")
            canvas.paste(tile, (int(row["x"]) // scale, int(row["y"]) // scale))
        return canvas

    def get_label_image(self) -> Image.Image | None:
        native = self._get_ext_image(EXT_IMAGE_TYPES["label"])
        if native is not None:
            return native
        return self.get_overview_image()

    def get_overview_image(self) -> Image.Image | None:
        """Return the separate overview image stored in tbl_airimg_info."""
        if "tbl_airimg_info" not in self.tables:
            return None
        row = self._con.execute(
            "SELECT data FROM tbl_airimg_info ORDER BY id LIMIT 1"
        ).fetchone()
        if row is None or row["data"] is None:
            return None
        return Image.open(io.BytesIO(row["data"])).convert("RGB")

    def get_scan_metadata(self) -> dict[str, object]:
        """Return scanner metadata from tbl_user_info and tbl_ext_info type 6."""
        metadata: dict[str, object] = {}
        if "tbl_user_info" in self.tables:
            row = self._con.execute(
                "SELECT * FROM tbl_user_info ORDER BY id LIMIT 1"
            ).fetchone()
            if row is not None:
                metadata.update(dict(row))

        ext_info = self._get_ext_json(6)
        if isinstance(ext_info, dict):
            user_scan_time = metadata.get("scanTime")
            metadata.update(ext_info)
            if user_scan_time is not None:
                metadata["userScanTime"] = user_scan_time
            packet_time = ext_info.get("packetTime")
            if packet_time:
                metadata["scanTime"] = packet_time
            if "scanTime" in ext_info:
                metadata["scanDuration"] = ext_info["scanTime"]
        return metadata
