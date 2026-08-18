from __future__ import annotations

import io
import sqlite3
import struct
import sys
import zlib
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


def create_sample_image(
    path: Path,
    *,
    directional_tile: bool = False,
    empty_tiles: set[tuple[int, int, int]] | None = None,
    include_native_resources: bool = False,
) -> None:
    """Create a small file matching the investigated private .image layout."""
    width = 512
    height = 512
    tile_size = 256
    levels = [
        (
            2,
            2,
            [
                (30, 40, 50),
                (80, 90, 100),
                (130, 140, 150),
                (180, 190, 200),
            ],
        ),
        (1, 1, [(100, 110, 120)]),
    ]

    header_size = 0xAC
    header = bytearray(header_size)
    struct.pack_into("<Q", header, 0x10, 0)
    struct.pack_into("<f", header, 0x18, 0.3466053 * width)
    struct.pack_into("<f", header, 0x1C, 0.3466053 * height)
    struct.pack_into("<II", header, 0x20, width, height)
    struct.pack_into("<I", header, 0x38, 40)
    header[0x40 : 0x40 + len("2026-01-01 12:34:56")] = b"2026-01-01 12:34:56"
    device = b"TEST-PUNUOXI"
    struct.pack_into("<I", header, 0x53, len(device))
    header[0x57 : 0x57 + len(device)] = device
    header[0x84 : 0x84 + len("测试医院".encode("gbk"))] = "测试医院".encode("gbk")
    header[0x98 : 0x98 + len("CASE-001")] = b"CASE-001"

    payload = bytearray(header)
    if include_native_resources:
        thumbnail_offset = len(payload)
        thumbnail = Image.new("RGB", (300, 5), (10, 20, 30)).tobytes()
        macro = Image.new("RGB", (1152, 625), (40, 50, 60)).tobytes()
        label = Image.new("RGB", (300, 294), (70, 80, 90)).tobytes()
        payload.extend(thumbnail)
        payload.extend(macro)
        payload.extend(b"\x00" * 24)
        label_offset = len(payload)
        payload.extend(label)
        struct.pack_into("<Q", payload, 0x00, thumbnail_offset)
        struct.pack_into("<Q", payload, 0x08, label_offset)

    entries: list[tuple[int, int, int]] = []
    for level_index, (columns, rows, colors) in enumerate(levels):
        # The private container stores records column-major, while ``colors``
        # is expressed in the more convenient row-major coordinate order.
        for column in range(columns):
            for row in range(rows):
                color = colors[row * columns + column]
                if empty_tiles and (level_index, column, row) in empty_tiles:
                    entries.append((0, len(payload), 0))
                    continue
                if directional_tile and level_index == 0 and column == 0 and row == 0:
                    logical = Image.new("RGB", (tile_size, tile_size), color)
                    logical.paste((220, 30, 40), (0, 0, tile_size // 2, tile_size // 2))
                    logical.paste((30, 210, 50), (tile_size // 2, 0, tile_size, tile_size // 2))
                    logical.paste((40, 60, 220), (0, tile_size // 2, tile_size // 2, tile_size))
                    logical.paste((220, 210, 40), (tile_size // 2, tile_size // 2, tile_size, tile_size))
                    stored = logical.transpose(Image.Transpose.TRANSPOSE)
                    buffer = io.BytesIO()
                    stored.save(buffer, format="JPEG", quality=100, subsampling=0)
                    blob = buffer.getvalue()
                else:
                    blob = jpeg_bytes(color, (tile_size, tile_size), quality=100)
                offset = len(payload)
                payload.extend(blob)
                entries.append((len(blob), offset, 0))

    index_offset = len(payload)
    struct.pack_into("<Q", payload, 0x10, index_offset)
    entry_index = 0
    for columns, rows, _colors in levels:
        payload.extend(struct.pack("<II", columns, rows))
        for _ in range(columns * rows):
            payload.extend(struct.pack("<III", *entries[entry_index]))
            entry_index += 1

    payload.extend(b"0123456789abcdef0123456789abcdef\x00")
    path.write_bytes(bytes(payload))


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


def _kfb_tile_record(
    spec: dict,
    tile_index_offset: int,
    *,
    indirect: bool = False,
    version: float = 1.1,
) -> bytes:
    record_size = 68 if version == 1.0 else 64
    record = bytearray(record_size)
    record[0:4] = b"\xf1\x04\xee\xee"
    struct.pack_into("<i", record, 4, spec["x"])
    struct.pack_into("<i", record, 8, spec["y"])
    struct.pack_into("<i", record, 12, spec["width"])
    struct.pack_into("<i", record, 16, spec["height"])
    struct.pack_into("<f", record, 20, spec["scale"])
    struct.pack_into("<i", record, 32, len(spec["blob"]))
    if indirect and version >= 2.1:
        struct.pack_into("<Q", record, 36, spec["offset_ref"])
        struct.pack_into("<Q", record, 44, spec["size_ref"])
        struct.pack_into("<Q", record, 52, 0)
    elif version >= 2.1:
        struct.pack_into("<Q", record, 36, spec["offset"])
        struct.pack_into("<Q", record, 44, len(spec["blob"]))
        struct.pack_into("<Q", record, 52, 0)
    elif indirect:
        struct.pack_into("<I", record, 36, spec["offset_ref"])
        struct.pack_into("<I", record, 44, spec["size_ref"])
    else:
        struct.pack_into("<i", record, 36, spec["offset"] - tile_index_offset)
    if version < 2.1:
        struct.pack_into("<i", record, 40, -1)
    if indirect and version < 2.1:
        for offset in (48, 52, 56):
            struct.pack_into("<i", record, offset, 0)
    elif not indirect and version < 2.1:
        for offset in (44, 48, 52, 56):
            struct.pack_into("<i", record, offset, 20)
    record[-4:] = b"\xff\x04\xee\xee"
    return bytes(record)


def create_sample_kfb(
    path: Path,
    *,
    include_preview_level: bool = True,
    variant: str = "kfb",
    version: float | None = None,
    header_marker: bytes = b"\xf1\x01\xee\xee",
    compressed_index: bool = False,
    fluorescence_channel: (
        tuple[str, tuple[int, int, int], float]
        | list[tuple[str, tuple[int, int, int], float]]
        | None
    ) = None,
) -> None:
    if variant not in {"kfb", "kfbl", "kfbf"}:
        raise ValueError(f"unsupported KFB fixture variant: {variant}")
    if header_marker not in {b"\xf1\x01\xee\xee", b"\xf1\x02\xee\xee"}:
        raise ValueError("unsupported KFB header marker")
    width = 24
    height = 18
    tile_size = 16
    fluorescence_channels = (
        fluorescence_channel
        if isinstance(fluorescence_channel, list)
        else ([fluorescence_channel] if fluorescence_channel is not None else [])
    )
    header_size = 512 if fluorescence_channels else 220
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
        blob = (
            _gray_jpeg_bytes(int(spec["color"][0]), (spec["width"], spec["height"]))
            if fluorescence_channels
            else jpeg_bytes(spec["color"], (spec["width"], spec["height"]), quality=100)
        )
        spec["blob"] = blob
        spec["offset"] = current_offset
        tile_data.extend(blob)
        current_offset += len(blob)

    indirect = variant == "kfbf"
    version = float(version if version is not None else (2.1 if indirect else 1.1))
    pointer_tables = bytearray()
    if indirect:
        offset_ref_start = current_offset
        size_ref_start = offset_ref_start + len(tile_specs) * 8
        for index, spec in enumerate(tile_specs):
            spec["offset_ref"] = offset_ref_start + index * 8
            spec["size_ref"] = size_ref_start + index * 8
            pointer_tables.extend(struct.pack("<Q", spec["offset"]))
        for spec in tile_specs:
            pointer_tables.extend(struct.pack("<Q", len(spec["blob"])))

    tile_index_offset = current_offset + len(pointer_tables)
    raw_tile_index = b"".join(
        _kfb_tile_record(spec, tile_index_offset, indirect=indirect, version=version)
        for spec in tile_specs
    )
    if compressed_index:
        compressed = zlib.compress(raw_tile_index)
        tile_index = b"\xf1\x04\xee\xee" + struct.pack("<II", len(compressed), len(raw_tile_index)) + compressed
    else:
        tile_index = raw_tile_index
    last_image_offset = tile_index_offset + len(tile_index)
    thumb_record = _kfb_image_record(2, thumb_blob, 12, 9)

    header = bytearray(header_size)
    header[0:4] = header_marker
    header[4:8] = {"kfb": b"KFB\x00", "kfbl": b"KFBL", "kfbf": b"KFBF"}[variant]
    struct.pack_into("<f", header, 0x0C, version)
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
    if not fluorescence_channels:
        metadata.extend(b"\xff\x01\xee\xee")
        metadata.extend(struct.pack("<I", 1))
        metadata.extend(struct.pack("<II", 29, len(device)))
        metadata.extend(device)
    else:
        external_offset = 256
        channel_blobs = {
            76: b"".join(
                str(index + 1).encode("ascii").ljust(20, b"\x00")
                for index in range(len(fluorescence_channels))
            ),
            77: b"".join(name.encode("utf-8")[:39].ljust(40, b"\x00") for name, _color, _exposure in fluorescence_channels),
            78: struct.pack(f"<{len(fluorescence_channels)}i", *range(1, len(fluorescence_channels) + 1)),
            79: b"".join(struct.pack("<iii", *color) for _name, color, _exposure in fluorescence_channels),
            84: struct.pack(f"<{len(fluorescence_channels)}d", *(item[2] for item in fluorescence_channels)),
        }
        offsets: dict[int, int] = {}
        for item_id, blob in channel_blobs.items():
            offsets[item_id] = external_offset
            header[external_offset : external_offset + len(blob)] = blob
            external_offset += len(blob)
        header_items = [(75, struct.pack("<I", len(fluorescence_channels)))] + [
            (item_id, struct.pack("<Q", offsets[item_id]))
            for item_id in (76, 77, 78, 79, 84)
        ]
        metadata.extend(b"\xff\x01\xee\xee")
        metadata.extend(struct.pack("<I", len(header_items)))
        for item_id, blob in header_items:
            metadata.extend(struct.pack("<II", item_id, len(blob)))
            metadata.extend(blob)
    header[0x5C : 0x5C + len(metadata)] = metadata

    path.write_bytes(
        bytes(header)
        + macro_record
        + label_record
        + bytes(tile_data)
        + bytes(pointer_tables)
        + tile_index
        + thumb_record
    )


def _kfba_data_block(values: dict[int, int]) -> bytes:
    block = bytearray(424)
    struct.pack_into("<Q", block, 0, len(values))
    for index, (item_id, value) in enumerate(values.items()):
        struct.pack_into("<IIIQ", block, 8 + index * 20, item_id, 0, 8, value)
    return bytes(block)


def _gray_jpeg_bytes(value: int, size: tuple[int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("L", size, value).save(buffer, format="JPEG", quality=100, subsampling=0)
    return buffer.getvalue()


def create_sample_kfba(
    path: Path,
    *,
    omit_item_id: int | None = None,
    field_count: int = 2,
) -> None:
    width, height, tile_size = 24, 18, 16
    channel_count = 2
    header_size = 512

    macro_blob = jpeg_bytes((230, 230, 230), (64, 24), quality=100)
    label_blob = jpeg_bytes((40, 50, 60), (32, 40), quality=100)
    thumb_blob = jpeg_bytes((90, 100, 110), (12, 9), quality=100)
    macro_record = _kfb_image_record(2, macro_blob, 64, 24)
    label_record = _kfb_image_record(3, label_blob, 32, 40)
    thumb_record = _kfb_image_record(2, thumb_blob, 12, 9)

    block_specs: list[dict[str, object]] = []
    for field_index in range(field_count):
        for level_index, level_width, level_height, tiles in (
            (
                0,
                width,
                height,
                ((0, 0, 16, 16), (16, 0, 8, 16), (0, 16, 16, 2), (16, 16, 8, 2)),
            ),
            (1, 12, 9, ((0, 0, 12, 9),)),
        ):
            del level_width, level_height
            for x, y, tile_width, tile_height in tiles:
                channel_values = (
                    30 + field_index * 40 + level_index * 10,
                    120 + field_index * 40 + level_index * 10,
                )
                block_specs.append(
                    {
                        "field": field_index,
                        "level": level_index,
                        "x": x,
                        "y": y,
                        "width": tile_width,
                        "height": tile_height,
                        "blobs": [
                            _gray_jpeg_bytes(value, (tile_width, tile_height))
                            for value in channel_values
                        ],
                    }
                )

    header = bytearray(header_size)
    header[0:4] = b"\xf1\x01\xee\xee"
    header[4:12] = b"KFBA\x00\x00\x00\x00"
    struct.pack_into("<f", header, 0x0C, 1.4)
    struct.pack_into("<I", header, 0x10, len(block_specs))
    struct.pack_into("<I", header, 0x14, height)
    struct.pack_into("<I", header, 0x18, width)
    struct.pack_into("<I", header, 0x1C, 40)
    header[0x20:0x24] = b"JPEG"
    struct.pack_into("<I", header, 0x2C, 1_700_000_000)
    struct.pack_into("<f", header, 0x4C, 0.25)
    struct.pack_into("<I", header, 0x58, tile_size)

    header_items = [
        (75, struct.pack("<I", channel_count)),
        (76, b"1\x00".ljust(20, b"\x00") + b"2\x00".ljust(20, b"\x00")),
        (77, b"Red\x00".ljust(40, b"\x00") + b"Green\x00".ljust(40, b"\x00")),
        (78, struct.pack("<ii", 1, 2)),
        (79, struct.pack("<iiiiii", 255, 0, 0, 0, 255, 0)),
        (84, struct.pack("<dd", 1.5, 2.5)),
    ]
    metadata = bytearray(b"\xff\x01\xee\xee" + struct.pack("<I", len(header_items)))
    for item_id, blob in header_items:
        metadata.extend(struct.pack("<II", item_id, len(blob)))
        metadata.extend(blob)
    header[0x5C : 0x5C + len(metadata)] = metadata

    payload = bytearray(header)
    first_image_offset = len(payload)
    payload.extend(macro_record)
    second_image_offset = len(payload)
    payload.extend(label_record)
    last_image_offset = len(payload)
    payload.extend(thumb_record)

    for spec in block_specs:
        offsets: list[int] = []
        lengths: list[int] = []
        for blob in spec["blobs"]:
            offsets.append(len(payload))
            lengths.append(len(blob))
            payload.extend(blob)
        spec["offsets"] = offsets
        spec["lengths"] = lengths

    for spec in block_specs:
        spec["offset_table"] = len(payload)
        payload.extend(struct.pack("<QQ", *spec["offsets"]))
        spec["length_table"] = len(payload)
        payload.extend(struct.pack("<QQ", *spec["lengths"]))

    tile_index_offset = len(payload)
    for block_number, spec in enumerate(block_specs):
        values = {
            0: int(spec["x"]),
            1: int(spec["y"]),
            2: int(spec["height"]),
            3: int(spec["width"]),
            4: int(spec["offset_table"]),
            5: int(spec["length_table"]),
            6: int(spec["field"]),
            7: int(spec["level"]),
            10: block_number,
        }
        if omit_item_id is not None:
            values.pop(omit_item_id, None)
        payload.extend(_kfba_data_block(values))

    struct.pack_into("<I", payload, 0x34, first_image_offset)
    struct.pack_into("<I", payload, 0x38, second_image_offset)
    struct.pack_into("<Q", payload, 0x3C, last_image_offset)
    struct.pack_into("<Q", payload, 0x44, tile_index_offset)
    path.write_bytes(payload)


def _kfbx_attribute(attribute_id: int, value_type: int, values) -> bytes:
    if value_type == 0:
        payload = bytes(values)
        count = len(payload)
    else:
        format_char = {1: "B", 2: "i", 3: "Q", 4: "f"}[value_type]
        sequence = tuple(values)
        payload = struct.pack(f"<{len(sequence)}{format_char}", *sequence)
        count = len(sequence)
    return struct.pack("<HHI", attribute_id, value_type, count) + payload


def create_sample_kfbx(path: Path) -> None:
    width, height, tile_size = 24, 18, 16
    level_specs = [
        (
            40.0,
            width,
            height,
            [
                (0, 0, 16, 16, (20, 30, 40)),
                (16, 0, 8, 16, (80, 90, 100)),
                (0, 16, 16, 2, (140, 150, 160)),
                (16, 16, 8, 2, (200, 210, 220)),
            ],
        ),
        (20.0, 12, 9, [(0, 0, 12, 9, (120, 130, 140))]),
    ]

    header = bytearray(76)
    header[:4] = b"KAC\x00"
    payload = bytearray(header)
    encoded_levels: list[tuple[float, int, int, list[int], list[int], list[int]]] = []
    for magnification, level_width, level_height, tiles in level_specs:
        offsets: list[int] = []
        lengths: list[int] = []
        coordinates: list[int] = []
        for x, y, tile_width, tile_height, color in tiles:
            blob = jpeg_bytes(color, (tile_width, tile_height), quality=100)
            offsets.append(len(payload))
            lengths.append(len(blob))
            coordinates.extend((x, y))
            payload.extend(blob)
        encoded_levels.append(
            (magnification, level_width, level_height, offsets, lengths, coordinates)
        )

    first_attribute_offset = len(payload)
    attributes = bytearray()
    attributes.extend(_kfbx_attribute(1, 2, [tile_size]))
    attributes.extend(_kfbx_attribute(2, 2, [tile_size]))
    attributes.extend(_kfbx_attribute(2123, 2, [1]))
    for magnification, level_width, level_height, offsets, lengths, coordinates in encoded_levels:
        nested = bytearray()
        nested.extend(_kfbx_attribute(3, 2, [level_width]))
        nested.extend(_kfbx_attribute(4, 2, [level_height]))
        nested.extend(_kfbx_attribute(5, 4, [magnification]))
        nested.extend(_kfbx_attribute(123, 3, offsets))
        nested.extend(_kfbx_attribute(135, 2, lengths))
        nested.extend(_kfbx_attribute(134, 2, coordinates))
        attributes.extend(_kfbx_attribute(123, 0, nested))

    struct.pack_into("<Q", payload, 68, first_attribute_offset)
    payload.extend(attributes)
    path.write_bytes(payload)


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
            tile_rows = (img_height + tile_height - 1) // tile_height
            tile_cols = (img_width + tile_width - 1) // tile_width
            for row in range(tile_rows):
                for col in range(tile_cols):
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
