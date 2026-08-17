from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import struct

import numpy as np

from ibl2svs.kfb_source import (
    KfbFormatError,
    KfbSlideSource,
    apply_vendor_lut,
    compose_vendor_channels,
    parse_kfba_data_block,
    parse_kfbx_attributes,
)
from tests.support import create_sample_kfb, create_sample_kfba, create_sample_kfbx


class KfbSlideSourceTests(unittest.TestCase):
    def test_reads_header_levels_and_region(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "sample.kfb"
            create_sample_kfb(path)

            with KfbSlideSource(path) as source:
                self.assertEqual(source.width, 24)
                self.assertEqual(source.height, 18)
                self.assertEqual(source.tile_count, 5)
                self.assertEqual(source.tile_size, 16)
                self.assertEqual(source.source_version, "1.1")
                self.assertEqual(source.level_dimensions, [(24, 18), (12, 9)])
                self.assertEqual(source.compatibility_level, "static_unverified")
                self.assertEqual(source.base_info.max_zoom_rate, 40)
                self.assertAlmostEqual(source.base_info.mpp, 0.25, places=5)

                region = source.read_region(14, 14, 8, 4)

            self.assertEqual(region.shape, (4, 8, 3))
            np.testing.assert_allclose(region[0, 0], [20, 30, 40], atol=3)
            np.testing.assert_allclose(region[0, 4], [80, 90, 100], atol=3)
            np.testing.assert_allclose(region[3, 0], [140, 150, 160], atol=3)
            np.testing.assert_allclose(region[3, 4], [200, 210, 220], atol=3)

    def test_reads_kfbf_indirect_tile_pointers(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "sample.kfbf"
            create_sample_kfb(path, variant="kfbf")

            with KfbSlideSource(path) as source:
                self.assertEqual(source.container_variant, "kfbf")
                self.assertEqual(source.source_version, "2.1")
                self.assertEqual(source.compatibility_level, "sample_verified")
                self.assertEqual(source.width, 24)
                self.assertEqual(source.height, 18)
                region = source.read_region(14, 14, 8, 4)

            np.testing.assert_allclose(region[0, 0], [20, 30, 40], atol=3)
            np.testing.assert_allclose(region[0, 4], [80, 90, 100], atol=3)
            np.testing.assert_allclose(region[3, 0], [140, 150, 160], atol=3)
            np.testing.assert_allclose(region[3, 4], [200, 210, 220], atol=3)

    def test_returns_associated_images(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "sample.kfb"
            create_sample_kfb(path)

            with KfbSlideSource(path) as source:
                macro = source.get_macro_image()
                label = source.get_label_image()
                preview = source.get_preview_image()

            self.assertIsNotNone(macro)
            self.assertIsNotNone(label)
            self.assertIsNotNone(preview)
            self.assertEqual(macro.size, (64, 24))
            self.assertEqual(label.size, (32, 40))
            self.assertEqual(preview.size, (12, 9))

    def test_preview_falls_back_to_macro_without_stitching_main_level(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "sample.kfb"
            create_sample_kfb(path, include_preview_level=False)

            with KfbSlideSource(path) as source:
                source._stitch_level = lambda scale: self.fail("must not stitch the main-resolution level")
                preview = source.get_preview_image()

            self.assertIsNotNone(preview)
            self.assertEqual(preview.size, (64, 24))

    def test_supports_all_classic_versions_markers_and_compressed_index(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            for variant in ("kfb", "kfbl", "kfbf"):
                for version_index in range(14):
                    version = round(1.0 + version_index / 10.0, 1)
                    path = root / f"v{version}.{variant}"
                    create_sample_kfb(
                        path,
                        variant=variant,
                        version=version,
                        header_marker=(
                            b"\xf1\x02\xee\xee" if version_index % 2 else b"\xf1\x01\xee\xee"
                        ),
                        compressed_index=version == 1.2,
                    )
                    with KfbSlideSource(path) as source:
                        self.assertEqual(source.source_container, variant)
                        self.assertEqual(source.source_version, f"{version:.1f}")
                        self.assertEqual(source.record_size, 68 if version == 1.0 else 64)
                        self.assertEqual(source.level_dimensions, [(24, 18), (12, 9)])

    def test_kfbf_uses_full_64_bit_indirect_position(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "sample.kfbf"
            create_sample_kfb(path, variant="kfbf")
            with path.open("r+b") as fh:
                fh.seek(0x44)
                tile_index_offset = struct.unpack("<I", fh.read(4))[0]
                fh.seek(tile_index_offset + 40)
                fh.write(struct.pack("<I", 1))

            with self.assertRaises(KfbFormatError) as raised:
                KfbSlideSource(path)

            self.assertEqual(raised.exception.diagnostic_code, "data_out_of_bounds")
            self.assertEqual(raised.exception.diagnostic_stage, "payload")

    def test_unknown_version_fails_with_structured_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "sample.kfb"
            create_sample_kfb(path)
            with path.open("r+b") as fh:
                fh.seek(0x0C)
                fh.write(struct.pack("<f", 3.0))

            with self.assertRaises(KfbFormatError) as raised:
                KfbSlideSource(path)

            self.assertEqual(raised.exception.diagnostic_code, "unsupported_version")
            self.assertEqual(raised.exception.diagnostic_stage, "header")

    def test_parses_kfba_items_and_kfbx_attributes(self) -> None:
        kfba = struct.pack("<QIIIQ", 1, 7, 3, 8, 99)
        self.assertEqual(parse_kfba_data_block(kfba)[0].value, 99)

        payloads = {
            0: b"ab",
            1: b"cd",
            2: struct.pack("<II", 3, 4),
            3: struct.pack("<QQ", 5, 6),
            4: struct.pack("<ff", 7.0, 8.0),
        }
        kfbx = b"".join(
            struct.pack("<HHI", 100 + value_type, value_type, 2) + payload
            for value_type, payload in payloads.items()
        )
        attributes = parse_kfbx_attributes(kfbx)
        self.assertEqual([attribute.payload for attribute in attributes], list(payloads.values()))

    def test_kfbx_unknown_value_type_fails_explicitly(self) -> None:
        with self.assertRaises(KfbFormatError) as raised:
            parse_kfbx_attributes(struct.pack("<HHI", 5, 9, 1) + b"x")

        self.assertEqual(raised.exception.diagnostic_code, "kfbx_value_type_unsupported")

    def test_reads_kfba_fields_channels_and_associated_images(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "sample.kfba"
            create_sample_kfba(path)

            with KfbSlideSource(path) as source:
                self.assertEqual(source.source_container, "kfba")
                self.assertEqual(source.source_version, "1.4")
                self.assertEqual(source.compatibility_level, "static_unverified")
                self.assertEqual(source.native_fields, (0, 1))
                self.assertEqual(source.native_channel_count, 2)
                self.assertEqual(source.source_axes, "TZCYX")
                self.assertEqual(source.level_dimensions, [(24, 18), (12, 9)])
                self.assertEqual([item["name"] for item in source.channel_metadata], ["Red", "Green"])
                field0_red = source.read_level_field_plane_region(0, 0, 0, 0, 0, 0, 0, 8, 8)
                field1_green = source.read_level_field_plane_region(0, 1, 1, 0, 0, 0, 0, 8, 8)
                composite = source.read_level_region(0, 0, 0, 8, 8)
                macro = source.get_macro_image()
                label = source.get_label_image()
                thumbnail = source.get_thumbnail_image()

            np.testing.assert_allclose(field0_red, 30, atol=2)
            np.testing.assert_allclose(field1_green, 160, atol=2)
            np.testing.assert_allclose(composite[0, 0], [30, 120, 0], atol=2)
            self.assertEqual(macro.size, (64, 24))
            self.assertEqual(label.size, (32, 40))
            self.assertEqual(thumbnail.size, (12, 9))

    def test_kfba_missing_required_item_fails_explicitly(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "missing.kfba"
            create_sample_kfba(path, omit_item_id=5)

            with self.assertRaises(KfbFormatError) as raised:
                KfbSlideSource(path)

            self.assertEqual(raised.exception.diagnostic_code, "kfba_required_item_missing")
            self.assertEqual(raised.exception.diagnostic_stage, "kfba_index")

    def test_kfba_uses_full_64_bit_channel_table_position(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "pointer.kfba"
            create_sample_kfba(path)
            with path.open("r+b") as fh:
                fh.seek(0x44)
                tile_index_offset = struct.unpack("<Q", fh.read(8))[0]
                fh.seek(tile_index_offset + 104)
                fh.write(struct.pack("<I", 1))

            with self.assertRaises(KfbFormatError) as raised:
                KfbSlideSource(path)

            self.assertEqual(raised.exception.diagnostic_code, "data_out_of_bounds")
            self.assertEqual(raised.exception.diagnostic_stage, "payload")

    def test_reads_kfbx_nested_pyramid_attributes(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "sample.kfbx"
            create_sample_kfbx(path)

            with KfbSlideSource(path) as source:
                self.assertEqual(source.source_container, "kfbx")
                self.assertEqual(source.compatibility_level, "static_unverified")
                self.assertEqual(source.level_dimensions, [(24, 18), (12, 9)])
                self.assertEqual(source.tile_size, 16)
                self.assertEqual(source.tile_count, 5)
                region = source.read_region(14, 14, 8, 4)

            np.testing.assert_allclose(region[0, 0], [20, 30, 40], atol=3)
            np.testing.assert_allclose(region[0, 4], [80, 90, 100], atol=3)
            np.testing.assert_allclose(region[3, 0], [140, 150, 160], atol=3)
            np.testing.assert_allclose(region[3, 4], [200, 210, 220], atol=3)

    def test_vendor_lut_and_composition_fixed_vectors(self) -> None:
        samples = np.array([0, 128, 255], dtype=np.uint16)
        np.testing.assert_array_equal(apply_vendor_lut(samples), np.array([0, 128, 255], dtype=np.uint8))
        high_bit = np.array([0, 2048, 4095], dtype=np.uint16)
        np.testing.assert_array_equal(
            apply_vendor_lut(high_bit, black=0, white=4095),
            np.array([0, 128, 255], dtype=np.uint8),
        )

        red = np.array([[100]], dtype=np.uint8)
        green = np.array([[200]], dtype=np.uint8)
        weighted = compose_vendor_channels([red, green], [(255, 0, 0), (0, 255, 0)], mode=1)
        maximum = compose_vendor_channels([red, green], [(255, 255, 255), (255, 255, 255)], mode=2)
        np.testing.assert_array_equal(weighted[0, 0], [100, 200, 0])
        np.testing.assert_array_equal(maximum[0, 0], [200, 200, 200])


if __name__ == "__main__":
    unittest.main()
