from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from ibl2svs.tiff_source import TiffSlideSource, _recover_shifted_ifd
from ibl2svs.inspection import inspect_file


TIFFFILE_AVAILABLE = importlib.util.find_spec("tifffile") is not None

if TIFFFILE_AVAILABLE:
    import tifffile


def _rgb_fixture(width: int = 40, height: int = 35) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            image[y, x] = (x, y, (x + y) % 256)
    return image


@unittest.skipUnless(TIFFFILE_AVAILABLE, "tifffile is required for TIFF source tests")
class TiffSlideSourceTests(unittest.TestCase):
    def test_rgb_ome_tiff_is_brightfield_without_fluorescence_channels(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "he.ome.tif"
            tifffile.imwrite(
                path,
                _rgb_fixture(16, 16),
                ome=True,
                photometric="rgb",
                metadata={"axes": "YXS"},
            )

            inspection = inspect_file(path)

        self.assertEqual(inspection.source_modality, "brightfield")
        self.assertEqual(inspection.source_container, "ome_tiff")
        self.assertEqual(inspection.channel_count, 0)
        self.assertEqual(inspection.channel_definitions, ())

    def test_tiled_rgb_read_region_uses_segments_not_page_asarray(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "tiled.tif"
            image = _rgb_fixture()
            tifffile.imwrite(path, image, tile=(16, 16), photometric="rgb")

            with TiffSlideSource(path) as source:
                with mock.patch("tifffile.TiffPage.asarray", side_effect=AssertionError("asarray should not be used")):
                    region = source.read_region(10, 7, 20, 14)

            np.testing.assert_array_equal(region, image[7:21, 10:30])

    def test_tiled_rgb_read_region_pads_bottom_right_edge(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "tiled.tif"
            image = _rgb_fixture()
            tifffile.imwrite(path, image, tile=(16, 16), photometric="rgb")

            with TiffSlideSource(path) as source:
                region = source.read_region(36, 32, 8, 6)

            np.testing.assert_array_equal(region[:3, :4], image[32:35, 36:40])
            self.assertTrue(np.all(region[3:] == 255))
            self.assertTrue(np.all(region[:, 4:] == 255))

    def test_stripped_rgb_read_region_crosses_strips(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "stripped.tif"
            image = _rgb_fixture()
            tifffile.imwrite(path, image, rowsperstrip=8, photometric="rgb")

            with TiffSlideSource(path) as source:
                region = source.read_region(5, 6, 12, 10)

            np.testing.assert_array_equal(region, image[6:16, 5:17])

    def test_grayscale_and_rgba_are_converted_to_rgb(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            gray = np.arange(16 * 16, dtype=np.uint8).reshape(16, 16)
            gray_path = root / "gray.tif"
            tifffile.imwrite(gray_path, gray, tile=(16, 16))

            rgba = np.zeros((16, 16, 4), dtype=np.uint8)
            rgba[..., 0] = 100
            rgba[..., 3] = 128
            rgba_path = root / "rgba.tif"
            tifffile.imwrite(rgba_path, rgba, tile=(16, 16), photometric="rgb")

            with TiffSlideSource(gray_path) as source:
                gray_region = source.read_region(0, 0, 4, 4)
            with TiffSlideSource(rgba_path) as source:
                rgba_pixel = source.read_region(0, 0, 1, 1)[0, 0]

            self.assertEqual(gray_region.shape, (4, 4, 3))
            np.testing.assert_array_equal(gray_region[..., 0], gray[:4, :4])
            np.testing.assert_array_equal(gray_region[..., 1], gray[:4, :4])
            self.assertEqual(rgba_pixel.tolist(), [177, 126, 126])

    def test_rejects_unsupported_bit_depth(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "uint16.tif"
            tifffile.imwrite(path, np.zeros((16, 16, 3), dtype=np.uint16), tile=(16, 16), photometric="rgb")

            with self.assertRaisesRegex(RuntimeError, "uint8"):
                TiffSlideSource(path)

    def test_recovers_shifted_ifd_offset(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "shifted.svs"
            stored_ifd = 16
            actual_ifd = 32
            delta = actual_ifd - stored_ifd
            entries: list[tuple[int, int, int, int]] = []
            extra = bytearray()

            def add_extra(payload: bytes) -> int:
                actual = actual_ifd + 2 + 10 * 12 + 4 + len(extra)
                extra.extend(payload)
                return actual - delta

            bits = add_extra(b"\x08\x00\x08\x00\x08\x00")
            description = add_extra(b"Aperio Image Library v12.0.0\x00")
            offsets = add_extra((8).to_bytes(4, "little") + (12).to_bytes(4, "little"))
            counts = add_extra((4).to_bytes(4, "little") + (4).to_bytes(4, "little"))
            entries.extend(
                [
                    (256, 3, 1, 1),
                    (257, 3, 1, 1),
                    (258, 3, 3, bits),
                    (259, 3, 1, 7),
                    (262, 3, 1, 2),
                    (270, 2, 29, description),
                    (277, 3, 1, 3),
                    (284, 3, 1, 1),
                    (324, 4, 2, offsets),
                    (325, 4, 2, counts),
                ]
            )
            ifd = bytearray((len(entries)).to_bytes(2, "little"))
            for tag, value_type, count, value in entries:
                ifd.extend(tag.to_bytes(2, "little"))
                ifd.extend(value_type.to_bytes(2, "little"))
                ifd.extend(count.to_bytes(4, "little"))
                ifd.extend(value.to_bytes(4, "little"))
            ifd.extend((0).to_bytes(4, "little"))

            payload = bytearray(b"II*\x00" + stored_ifd.to_bytes(4, "little"))
            payload.extend(b"\xff\xd8\xff\xd9\xff\xd8\xff\xd9")
            payload.extend(b"\x00" * (actual_ifd - len(payload)))
            payload.extend(ifd)
            payload.extend(extra)
            path.write_bytes(payload)

            self.assertEqual(_recover_shifted_ifd(path), (stored_ifd, delta))


if __name__ == "__main__":
    unittest.main()
