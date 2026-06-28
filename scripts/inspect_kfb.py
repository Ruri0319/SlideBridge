from __future__ import annotations

import argparse
import csv
import json
import struct
import sys
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

from PIL import Image


KFB_HEADER_MARKER = b"\xf1\x01\xee\xee"
META_MARKER = b"\xff\x01\xee\xee"
IMAGE_RECORD_SIZE = 52
TILE_RECORD_SIZE = 64


def _u32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def _i32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<i", data, offset)[0]


def _f32(data: bytes, offset: int) -> float:
    return struct.unpack_from("<f", data, offset)[0]


def _parse_metadata(data: bytes, offset: int, limit: int) -> dict[str, object]:
    if data[offset : offset + 4] != META_MARKER:
        return {"marker": data[offset : offset + 4].hex(), "entries": []}

    count = _u32(data, offset + 4)
    pos = offset + 8
    entries: list[dict[str, object]] = []
    for _ in range(count):
        if pos + 8 > limit:
            break
        tag = _u32(data, pos)
        length = _u32(data, pos + 4)
        value = data[pos + 8 : pos + 8 + length]
        entry: dict[str, object] = {"tag": tag, "length": length}
        try:
            text = value.rstrip(b"\x00").decode("utf-8")
        except UnicodeDecodeError:
            text = ""
        if text and all(ch.isprintable() for ch in text):
            entry["text"] = text
        if length == 4:
            entry["uint32"] = struct.unpack("<I", value)[0]
            entry["float32"] = struct.unpack("<f", value)[0]
        else:
            entry["hex"] = value.hex()
        entries.append(entry)
        pos += 8 + length

    return {"marker": META_MARKER.hex(), "count": count, "entries": entries}


def parse_header(path: Path) -> dict[str, object]:
    with path.open("rb") as fh:
        data = fh.read(512)
    if data[:4] != KFB_HEADER_MARKER or data[4:8] != b"KFB\x00":
        raise ValueError("not a recognized KFB header")

    first_image_record_offset = _u32(data, 0x34)
    header = {
        "path": str(path),
        "file_size": path.stat().st_size,
        "marker": data[:4].hex(),
        "magic": data[4:8].rstrip(b"\x00").decode("ascii", errors="replace"),
        "header_float": _f32(data, 0x0C),
        "tile_count": _u32(data, 0x10),
        "height": _u32(data, 0x14),
        "width": _u32(data, 0x18),
        "objective_power": _u32(data, 0x1C),
        "codec": data[0x20:0x24].rstrip(b"\x00").decode("ascii", errors="replace"),
        "raw_timestamp_uint32": _u32(data, 0x2C),
        "first_image_record_offset": first_image_record_offset,
        "second_image_record_offset": _u32(data, 0x38),
        "last_image_record_offset": _u32(data, 0x3C),
        "tile_index_offset": _u32(data, 0x44),
        "mpp": _f32(data, 0x4C),
        "tile_size": _u32(data, 0x58),
        "metadata": _parse_metadata(data, 0x5C, first_image_record_offset),
    }
    return header


def parse_image_record(fh: BinaryIO, offset: int) -> dict[str, object]:
    fh.seek(offset)
    head = fh.read(IMAGE_RECORD_SIZE)
    if len(head) != IMAGE_RECORD_SIZE or head[0] != 0xF1 or head[2:4] != b"\xee\xee":
        raise ValueError(f"invalid image record at {offset}")

    record_type = head[1]
    end_marker = bytes([0xFF, record_type, 0xEE, 0xEE])
    if head[48:52] != end_marker:
        raise ValueError(f"invalid image record end marker at {offset}")

    size = _u32(head, 20)
    jpeg_offset = offset + IMAGE_RECORD_SIZE
    fh.seek(jpeg_offset)
    jpeg = fh.read(size)
    image = Image.open(BytesIO(jpeg))
    image.load()
    return {
        "offset": offset,
        "type": record_type,
        "height": _u32(head, 8),
        "width": _u32(head, 12),
        "channels": _u32(head, 16),
        "jpeg_size": size,
        "record_size": _u32(head, 24),
        "jpeg_offset": jpeg_offset,
        "pil_size": list(image.size),
        "jpeg": jpeg,
    }


def parse_tile_index(path: Path, header: dict[str, object]) -> list[dict[str, object]]:
    index_offset = int(header["tile_index_offset"])
    count = int(header["tile_count"])
    tiles: list[dict[str, object]] = []
    with path.open("rb") as fh:
        fh.seek(index_offset)
        for index in range(count):
            entry = fh.read(TILE_RECORD_SIZE)
            if len(entry) != TILE_RECORD_SIZE:
                raise ValueError("tile index truncated")
            if entry[:4] != b"\xf1\x04\xee\xee" or entry[60:64] != b"\xff\x04\xee\xee":
                raise ValueError(f"invalid tile record marker at index {index}")
            jpeg_size = _i32(entry, 32)
            rel_offset = _i32(entry, 36)
            tiles.append(
                {
                    "index": index,
                    "x": _i32(entry, 4),
                    "y": _i32(entry, 8),
                    "width": _i32(entry, 12),
                    "height": _i32(entry, 16),
                    "scale_value": _f32(entry, 20),
                    "jpeg_size": jpeg_size,
                    "jpeg_offset": index_offset + rel_offset,
                }
            )
    return tiles


def summarize_levels(tiles: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[float, list[dict[str, object]]] = defaultdict(list)
    for tile in tiles:
        groups[round(float(tile["scale_value"]), 6)].append(tile)

    levels: list[dict[str, object]] = []
    for scale, rows in sorted(groups.items(), reverse=True):
        width = max(int(row["x"]) + int(row["width"]) for row in rows)
        height = max(int(row["y"]) + int(row["height"]) for row in rows)
        xs = {int(row["x"]) for row in rows}
        ys = {int(row["y"]) for row in rows}
        levels.append(
            {
                "scale_value": scale,
                "tile_count": len(rows),
                "width": width,
                "height": height,
                "grid_columns": len(xs),
                "grid_rows": len(ys),
            }
        )
    return levels


def export_associated_images(path: Path, header: dict[str, object], output_dir: Path) -> list[dict[str, object]]:
    exported: list[dict[str, object]] = []
    offsets = [
        int(header["first_image_record_offset"]),
        int(header["second_image_record_offset"]),
        int(header["last_image_record_offset"]),
    ]
    with path.open("rb") as fh:
        for offset in offsets:
            record = parse_image_record(fh, offset)
            name = f"associated_type{record['type']}_{record['width']}x{record['height']}.jpg"
            out_path = output_dir / name
            out_path.write_bytes(record.pop("jpeg"))
            record["export_path"] = str(out_path)
            exported.append(record)
    return exported


def stitch_level(path: Path, tiles: list[dict[str, object]], scale_value: float, output_dir: Path) -> Path | None:
    rows = [tile for tile in tiles if round(float(tile["scale_value"]), 6) == round(scale_value, 6)]
    if not rows:
        return None

    width = max(int(tile["x"]) + int(tile["width"]) for tile in rows)
    height = max(int(tile["y"]) + int(tile["height"]) for tile in rows)
    canvas = Image.new("RGB", (width, height), "white")
    with path.open("rb") as fh:
        for tile in rows:
            fh.seek(int(tile["jpeg_offset"]))
            blob = fh.read(int(tile["jpeg_size"]))
            image = Image.open(BytesIO(blob)).convert("RGB")
            canvas.paste(image, (int(tile["x"]), int(tile["y"])))

    out_path = output_dir / f"preview_scale_{scale_value:g}_{width}x{height}.jpg"
    canvas.save(out_path, quality=90)
    return out_path


def write_tile_csv(tiles: list[dict[str, object]], path: Path) -> None:
    fields = ["index", "scale_value", "x", "y", "width", "height", "jpeg_offset", "jpeg_size"]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for tile in tiles:
            writer.writerow({field: tile[field] for field in fields})


def inspect(path: Path, output_dir: Path, preview_scales: list[float]) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    header = parse_header(path)
    tiles = parse_tile_index(path, header)
    levels = summarize_levels(tiles)
    associated = export_associated_images(path, header, output_dir)
    previews = []
    for scale in preview_scales:
        preview = stitch_level(path, tiles, scale, output_dir)
        if preview is not None:
            previews.append(str(preview))

    write_tile_csv(tiles, output_dir / "tile_index.csv")
    summary = {
        "header": header,
        "levels": levels,
        "associated_images": associated,
        "previews": previews,
        "tile_index_csv": str(output_dir / "tile_index.csv"),
    }
    (output_dir / "metadata.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a KFB whole-slide image container.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("kfb_parse_output"))
    parser.add_argument("--preview-scale", type=float, action="append")
    args = parser.parse_args()

    preview_scales = args.preview_scale if args.preview_scale is not None else [2.5, 1.25]
    summary = inspect(args.path, args.output_dir, preview_scales)
    printable = {
        "header": summary["header"],
        "levels": summary["levels"],
        "associated_images": [
            {key: value for key, value in image.items() if key != "jpeg"}
            for image in summary["associated_images"]
        ],
        "previews": summary["previews"],
        "tile_index_csv": summary["tile_index_csv"],
    }
    print(json.dumps(printable, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
