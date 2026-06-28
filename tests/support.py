from __future__ import annotations

import io
import sqlite3
import struct
import sys
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
VENDOR_DIR = ROOT / "vendor"
if VENDOR_DIR.exists():
    sys.path.insert(0, str(VENDOR_DIR))


def jpeg_bytes(color: tuple[int, int, int], size: tuple[int, int], *, quality: int = 95) -> bytes:
    image = Image.new("RGB", size, color)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality, subsampling=0)
    return buf.getvalue()


def _kfb_image_record(image_type: int, blob: bytes, width: int, height: int) -> bytes:
    record = bytearray(52)
    record[0:4] = bytes([0xF1, image_type, 0xEE, 0xEE])
    struct.pack_into("<I", record, 4, 1)
    struct.pack_into("<I", record, 8, height)
    struct.pack_into("<I", record, 12, width)
    struct.pack_into("<I", record, 16, 3)
    struct.pack_into("<I", record, 20, len(blob))
    struct.pack_into("<I", record, 24, 52)
    record[48:52] = bytes([0xFF, image_type, 0xEE, 0xEE])
    return bytes(record) + blob


def _kfb_tile_record(spec: dict, tile_index_offset: int) -> bytes:
    record = bytearray(64)
    record[0:4] = b"\xf1\x04\xee\xee"
    struct.pack_into("<i", record, 4, spec["x"])
    struct.pack_into("<i", record, 8, spec["y"])
    struct.pack_into("<i", record, 12, spec["width"])
    struct.pack_into("<i", record, 16, spec["height"])
    struct.pack_into("<f", record, 20, spec["scale"])
    struct.pack_into("<i", record, 32, len(spec["blob"]))
    struct.pack_into("<i", record, 36, spec["offset"] - tile_index_offset)
    struct.pack_into("<i", record, 40, -1)
    for offset in (44, 48, 52, 56):
        struct.pack_into("<i", record, offset, 20)
    record[60:64] = b"\xff\x04\xee\xee"
    return bytes(record)


def create_sample_kfb(path: Path, *, include_preview_level: bool = True) -> None:
    width = 24
    height = 18
    tile_size = 16
    header_size = 220
    first_image_offset = header_size

    macro_blob = jpeg_bytes((230, 230, 230), (64, 24), quality=100)
    label_blob = jpeg_bytes((40, 50, 60), (32, 40), quality=100)
    thumb_blob = jpeg_bytes((90, 100, 110), (12, 9), quality=100)
    macro_record = _kfb_image_record(2, macro_blob, 64, 24)
    second_image_offset = first_image_offset + len(macro_record)
    label_record = _kfb_image_record(3, label_blob, 32, 40)

    tile_specs = [
        {"x": 0, "y": 0, "width": 16, "height": 16, "scale": 40.0, "color": (20, 30, 40)},
        {"x": 16, "y": 0, "width": 8, "height": 16, "scale": 40.0, "color": (80, 90, 100)},
        {"x": 0, "y": 16, "width": 16, "height": 2, "scale": 40.0, "color": (140, 150, 160)},
        {"x": 16, "y": 16, "width": 8, "height": 2, "scale": 40.0, "color": (200, 210, 220)},
    ]
    if include_preview_level:
        tile_specs.append({"x": 0, "y": 0, "width": 12, "height": 9, "scale": 20.0, "color": (120, 130, 140)})

    tile_data_offset = second_image_offset + len(label_record)
    tile_data = bytearray()
    current_offset = tile_data_offset
    for spec in tile_specs:
        blob = jpeg_bytes(spec["color"], (spec["width"], spec["height"]), quality=100)
        spec["blob"] = blob
        spec["offset"] = current_offset
        tile_data.extend(blob)
        current_offset += len(blob)

    tile_index_offset = current_offset
    tile_index = b"".join(_kfb_tile_record(spec, tile_index_offset) for spec in tile_specs)
    last_image_offset = tile_index_offset + len(tile_index)
    thumb_record = _kfb_image_record(2, thumb_blob, 12, 9)

    header = bytearray(header_size)
    header[0:4] = b"\xf1\x01\xee\xee"
    header[4:8] = b"KFB\x00"
    struct.pack_into("<f", header, 0x0C, 1.0)
    struct.pack_into("<I", header, 0x10, len(tile_specs))
    struct.pack_into("<I", header, 0x14, height)
    struct.pack_into("<I", header, 0x18, width)
    struct.pack_into("<I", header, 0x1C, 40)
    header[0x20:0x24] = b"JPEG"
    struct.pack_into("<I", header, 0x2C, 1_700_000_000)
    struct.pack_into("<I", header, 0x34, first_image_offset)
    struct.pack_into("<I", header, 0x38, second_image_offset)
    struct.pack_into("<I", header, 0x3C, last_image_offset)
    struct.pack_into("<I", header, 0x44, tile_index_offset)
    struct.pack_into("<f", header, 0x4C, 0.25)
    struct.pack_into("<I", header, 0x58, tile_size)

    device = b"TEST-KFB"
    metadata = bytearray()
    metadata.extend(b"\xff\x01\xee\xee")
    metadata.extend(struct.pack("<I", 1))
    metadata.extend(struct.pack("<II", 29, len(device)))
    metadata.extend(device)
    header[0x5C : 0x5C + len(metadata)] = metadata

    path.write_bytes(bytes(header) + macro_record + label_record + bytes(tile_data) + tile_index + thumb_record)


def create_sample_ibl(
    path: Path,
    *,
    grid_cols: int = 1,
    grid_rows: int = 1,
    img_width: int = 8,
    img_height: int = 8,
    tile_width: int = 2,
    tile_height: int = 2,
    step_x: int | None = None,
    step_y: int | None = None,
    include_preview: bool = True,
    include_shrink: bool = True,
    omit_tile: tuple[int, int] | None = None,
) -> None:
    step_x = img_width if step_x is None else step_x
    step_y = img_height if step_y is None else step_y
    total_img_width = step_x * max(0, grid_cols - 1) + img_width
    total_img_height = step_y * max(0, grid_rows - 1) + img_height
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.executescript(
        """
        CREATE TABLE tbl_img_info (
            id INTEGER NOT NULL UNIQUE,
            layer INTEGER,
            topX INTEGER,
            topY INTEGER,
            leftX INTEGER,
            leftY INTEGER,
            col INTEGER,
            row INTEGER,
            nX INTEGER,
            nY INTEGER,
            PRIMARY KEY(id)
        );
        CREATE TABLE tbl_tile_info (
            id INTEGER NOT NULL,
            layer INTEGER NOT NULL,
            col INTEGER NOT NULL,
            row INTEGER NOT NULL,
            data BLOB NOT NULL,
            PRIMARY KEY(id, layer, col, row)
        );
        CREATE TABLE tbl_base_info (
            magicNo TEXT,
            version TEXT,
            focus_num INTEGER,
            image_format INTEGER,
            layer_size INTEGER,
            img_color INTEGER,
            check_sum INTEGER,
            ratio_step INTEGER,
            max_Lay_size INTEGER,
            slide_type INTEGER,
            bk_color INTEGER,
            pixel_size REAL,
            total_img_num INTEGER,
            max_zoom_rate INTEGER,
            img_col INTEGER,
            img_row INTEGER,
            img_width INTEGER,
            img_height INTEGER,
            tile_width INTEGER,
            tile_height INTEGER,
            shrink_tile_num INTEGER,
            total_img_width INTEGER,
            total_img_height INTEGER
        );
        CREATE TABLE tbl_shrink_info (
            id INTEGER UNIQUE,
            layerNo INTEGER,
            x INTEGER,
            y INTEGER,
            data BLOB,
            PRIMARY KEY(id AUTOINCREMENT)
        );
        """
    )
    cur.execute(
        """
        INSERT INTO tbl_base_info VALUES
        (?, '1.0', 0, 0, 3, 24, 0, 4, 10, 0, 255, 0.00025, ?, 40, ?, ?, ?, ?, ?, ?, 1, ?, ?)
        """,
        (
            "ibl",
            grid_cols * grid_rows,
            grid_cols,
            grid_rows,
            img_width,
            img_height,
            tile_width,
            tile_height,
            total_img_width,
            total_img_height,
        ),
    )
    for grid_row in range(grid_rows):
        for grid_col in range(grid_cols):
            block_id = (grid_col << 20) + (grid_row << 12)
            x = grid_col * step_x
            y = grid_row * step_y
            cur.execute(
                """
                INSERT INTO tbl_img_info VALUES
                (?, 0, 0, 0, 0, 0, ?, ?, ?, ?)
                """,
                (block_id, grid_col, grid_row, x, y),
            )
            for row in range(4):
                for col in range(4):
                    if grid_col == 0 and grid_row == 0 and omit_tile == (col, row):
                        continue
                    color = (
                        min(255, grid_row * 40 + row * 20),
                        min(255, grid_col * 40 + col * 20),
                        min(255, 80 + grid_col * 25 + grid_row * 15),
                    )
                    cur.execute(
                        "INSERT INTO tbl_tile_info VALUES (?, ?, ?, ?, ?)",
                        (block_id, 0, col, row, jpeg_bytes(color, (tile_width, tile_height))),
                    )
            if include_preview:
                cur.execute(
                    "INSERT INTO tbl_tile_info VALUES (?, ?, ?, ?, ?)",
                    (
                        block_id,
                        1,
                        0,
                        0,
                        jpeg_bytes(
                            (
                                min(255, 100 + grid_row * 30),
                                min(255, 110 + grid_col * 30),
                                min(255, 120 + grid_col * 10 + grid_row * 10),
                            ),
                            (max(1, img_width // 4), max(1, img_height // 4)),
                        ),
                    ),
                )
    if include_shrink:
        cur.execute(
            "INSERT INTO tbl_shrink_info(layerNo, x, y, data) VALUES (?, ?, ?, ?)",
            (2, 0, 0, jpeg_bytes((10, 20, 30), (max(1, img_width // 16), max(1, img_height // 16)))),
        )
    con.commit()
    con.close()
