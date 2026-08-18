from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import xml.etree.ElementTree as ET
from io import BytesIO

import numpy as np
from PIL import Image
import tifffile

from ibl2svs.converter import convert_file
from ibl2svs.inspection import (
    apply_channel_overrides,
    inspect_file,
    inspect_inputs,
    open_slide,
    output_eligibility,
)
from ibl2svs.kfb_source import KfbSlideSource
from ibl2svs.models import ConvertOptions
from ibl2svs.native_jpeg import select_viewer_compatible_levels
from tests.support import create_sample_kfb, create_sample_kfba


class FluorescenceOutputTests(unittest.TestCase):
    def _sample(self, root: Path) -> Path:
        path = root / "sample.kfbf"
        create_sample_kfb(
            path,
            variant="kfbf",
            fluorescence_channel=("DAPI", (0, 0, 255), 85.0),
        )
        return path

    def test_viewer_compatible_levels_stop_after_useful_overview(self) -> None:
        dimensions = [
            (15750, 26639),
            (7875, 13319),
            (3937, 6659),
            (1968, 3329),
            (992, 1680),
            (496, 840),
            (248, 420),
            (124, 210),
            (62, 105),
            (1, 3),
            (1, 1),
        ]
        levels = [SimpleNamespace(dimensions=value) for value in dimensions]

        selected = select_viewer_compatible_levels(levels, 256)

        self.assertEqual([level.dimensions for level in selected], dimensions[:7])

    def test_inspection_reports_dapi_and_output_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            sample = self._sample(Path(tempdir))

            result = inspect_file(sample)

        self.assertEqual(result.source_modality, "fluorescence")
        self.assertEqual(result.channel_count, 1)
        self.assertEqual(result.channel_definitions[0].name, "DAPI")
        self.assertEqual(result.channel_definitions[0].color, (0, 0, 255))
        self.assertEqual(result.channel_definitions[0].exposure, 85.0)
        self.assertEqual(
            set(result.allowed_output_formats),
            {"ome_tiff", "fluorescence_svs", "afi"},
        )

    def test_unknown_channel_is_numbered_without_dye_inference(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            sample = root / "unknown.kfbf"
            create_sample_kfb(
                sample,
                variant="kfbf",
                fluorescence_channel=("", (255, 255, 255), 10.0),
            )

            inspection = inspect_file(sample)
            result = convert_file(sample, root / "unknown.afi", ConvertOptions(output_format="afi"))

            self.assertEqual(inspection.channel_definitions[0].name, "Channel 1")
            self.assertEqual(inspection.channel_definitions[0].identity_source, "unknown")
            self.assertTrue(result.success, result.error)
            self.assertTrue((root / "unknown_C01.svs").exists())

    def test_channel_override_changes_metadata_not_pixels(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            sample = self._sample(Path(tempdir))
            with KfbSlideSource(sample) as source:
                before = source.read_level_plane_region(0, 0, 0, 0, 0, 0, 16, 16).copy()
                definitions = apply_channel_overrides(
                    source,
                    [{"index": 0, "name": "AF", "fluor": "AF", "color": [255, 255, 255]}],
                )
                after = source.read_level_plane_region(0, 0, 0, 0, 0, 0, 16, 16)

        self.assertEqual(definitions[0].name, "AF")
        self.assertEqual(definitions[0].exposure, 85.0)
        np.testing.assert_array_equal(before, after)

    def test_stale_channel_override_is_rejected_with_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            sample = self._sample(root)

            result = convert_file(
                sample,
                root / "sample.ome.tif",
                ConvertOptions(output_format="ome_tiff", channel_overrides={str(sample): []}),
            )

            self.assertFalse(result.success)
            self.assertEqual(result.diagnostic_code, "channel_override_mismatch")
            self.assertFalse((root / "sample.ome.tif").exists())

    def test_writes_single_channel_pyramidal_ome_tiff(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            sample = self._sample(root)
            result = convert_file(sample, root / "sample.ome.tif", ConvertOptions(output_format="ome_tiff"))

            self.assertTrue(result.success, result.error)
            with tifffile.TiffFile(result.output_path) as tif:
                self.assertEqual(tif.series[0].axes, "YX")
                self.assertEqual(len(tif.series[0].levels), 2)
                self.assertIn("SizeC=\"1\"", tif.ome_metadata)
                self.assertIn("DAPI", tif.ome_metadata)
                self.assertIn("85.0", tif.ome_metadata)
                for level in tif.series[0].levels:
                    page = level.pages[0]
                    for offset, size in zip(page.dataoffsets, page.databytecounts):
                        tif.filehandle.seek(offset)
                        with Image.open(BytesIO(tif.filehandle.read(size))) as tile:
                            self.assertEqual(tile.size, (16, 16))

            reopened = inspect_file(result.output_path)
            self.assertEqual(reopened.source_modality, "fluorescence")
            self.assertEqual(reopened.channel_count, 1)
            self.assertEqual(reopened.channel_definitions[0].name, "DAPI")
            self.assertEqual(reopened.channel_definitions[0].color, (0, 0, 255))
            self.assertEqual(reopened.channel_definitions[0].exposure, 85.0)
            with open_slide(result.output_path) as source:
                plane = source.read_level_field_plane_region(0, 0, 0, 0, 0, 0, 0, 8, 8)
            np.testing.assert_allclose(plane, 20, atol=3)

    def test_writes_fluorescence_svs_page_order(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            sample = self._sample(root)
            result = convert_file(
                sample,
                root / "sample.svs",
                ConvertOptions(output_format="fluorescence_svs"),
            )

            self.assertTrue(result.success, result.error)
            self.assertEqual(result.native_tile_mode, "svs_reencoded")
            with tifffile.TiffFile(result.output_path) as tif:
                self.assertEqual(len(tif.pages), 5)
                self.assertTrue(tif.pages[0].is_tiled)
                self.assertIn("Dye = DAPI", tif.pages[0].description)
                self.assertIn("DisplayColor = 255", tif.pages[0].description)
                self.assertIn("thumbnail", tif.pages[1].description.lower())
                self.assertTrue(tif.pages[2].is_tiled)
                self.assertIn("label", tif.pages[3].description.lower())
                self.assertIn("macro", tif.pages[4].description.lower())

            reopened = inspect_file(result.output_path)
            self.assertEqual(reopened.source_modality, "fluorescence")
            self.assertEqual(reopened.channel_definitions[0].name, "DAPI")
            self.assertEqual(reopened.channel_definitions[0].color, (0, 0, 255))
            self.assertEqual(reopened.channel_definitions[0].exposure, 85.0)

    def test_writes_single_channel_afi_transactionally(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            sample = self._sample(root)
            result = convert_file(sample, root / "sample.afi", ConvertOptions(output_format="afi"))

            self.assertTrue(result.success, result.error)
            self.assertEqual(len(result.output_files), 2)
            paths = [node.text for node in ET.parse(result.output_path).getroot().findall("Path")]
            self.assertEqual(paths, ["sample_C01_DAPI.svs"])
            self.assertTrue((root / paths[0]).exists())

            reopened = inspect_file(result.output_path)
            self.assertEqual(reopened.source_modality, "fluorescence")
            self.assertEqual(reopened.channel_count, 1)
            self.assertEqual(reopened.channel_definitions[0].name, "DAPI")
            with open_slide(result.output_path) as source:
                plane = source.read_level_field_plane_region(0, 0, 0, 0, 0, 0, 0, 8, 8)
            np.testing.assert_allclose(plane, 20, atol=3)

            scanned = inspect_inputs(root, recursive=False)
            scanned_names = {item.input_path.name for item in scanned.files}
            self.assertIn("sample.afi", scanned_names)
            self.assertNotIn("sample_C01_DAPI.svs", scanned_names)

    def test_reopens_multichannel_ome_planes_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            sample = root / "multi.kfba"
            create_sample_kfba(sample, field_count=1)
            result = convert_file(sample, root / "multi.ome.tif", ConvertOptions(output_format="ome_tiff"))

            self.assertTrue(result.success, result.error)
            reopened = inspect_file(result.output_path)
            self.assertEqual(reopened.channel_count, 2)
            self.assertEqual([item.name for item in reopened.channel_definitions], ["Red", "Green"])
            with open_slide(result.output_path) as source:
                red = source.read_level_field_plane_region(0, 0, 0, 0, 0, 0, 0, 8, 8)
                green = source.read_level_field_plane_region(0, 0, 1, 0, 0, 0, 0, 8, 8)
            np.testing.assert_allclose(red, 30, atol=3)
            np.testing.assert_allclose(green, 120, atol=3)

    def test_rewrites_standard_unknown_ome_without_losing_planes(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            sample = root / "standard.ome.tif"
            planes = np.stack(
                [
                    np.full((16, 16), 35, dtype=np.uint8),
                    np.full((16, 16), 145, dtype=np.uint8),
                ]
            )
            tifffile.imwrite(
                sample,
                planes,
                ome=True,
                tile=(16, 16),
                metadata={
                    "axes": "CYX",
                    "Channel": {"Name": ["DAPI", "FITC"], "Color": [255, 16711935]},
                },
            )

            result = convert_file(sample, root / "rewritten.ome.tif", ConvertOptions(output_format="ome_tiff"))

            self.assertTrue(result.success, result.error)
            reopened = inspect_file(result.output_path)
            self.assertEqual(reopened.source_modality, "unknown")
            self.assertEqual(reopened.channel_count, 2)
            with open_slide(result.output_path) as source:
                first = source.read_level_plane_region(0, 0, 0, 0, 0, 0, 8, 8)
                second = source.read_level_plane_region(0, 1, 0, 0, 0, 0, 8, 8)
            np.testing.assert_array_equal(first, planes[0, :8, :8])
            np.testing.assert_array_equal(second, planes[1, :8, :8])

    def test_writes_multichannel_afi_in_source_order(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            sample = root / "multi.kfba"
            create_sample_kfba(sample, field_count=1)

            result = convert_file(sample, root / "multi.afi", ConvertOptions(output_format="afi"))

            self.assertTrue(result.success, result.error)
            paths = [node.text for node in ET.parse(result.output_path).getroot().findall("Path")]
            self.assertEqual(paths, ["multi_C01_Red.svs", "multi_C02_Green.svs"])
            self.assertTrue(all((root / path).exists() for path in paths))

    def test_ome_metadata_preserves_known_channel_when_another_is_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            sample = root / "mixed.kfba"
            create_sample_kfba(sample, field_count=1)
            overrides = {
                str(sample): [
                    {"index": 0, "name": "DAPI", "fluor": "DAPI", "color": [0, 0, 255]},
                    {"index": 1, "name": "", "fluor": "", "color": [255, 255, 255]},
                ]
            }

            result = convert_file(
                sample,
                root / "mixed.ome.tif",
                ConvertOptions(output_format="ome_tiff", channel_overrides=overrides),
            )

            self.assertTrue(result.success, result.error)
            with tifffile.TiffFile(result.output_path) as tif:
                xml = tif.ome_metadata or ""
            self.assertIn('Name="DAPI"', xml)
            self.assertIn('Fluor="DAPI"', xml)
            self.assertIn('Name="Channel 2"', xml)
            reopened = inspect_file(result.output_path)
            self.assertEqual(reopened.source_modality, "fluorescence")
            self.assertEqual(reopened.channel_count, 2)
            self.assertEqual(reopened.channel_definitions[1].identity_source, "unknown")

    def test_afi_failure_removes_all_created_files(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            sample = self._sample(root)
            with mock.patch(
                "ibl2svs.afi_writer.write_fluorescence_svs",
                side_effect=RuntimeError("encode failed"),
            ):
                result = convert_file(sample, root / "sample.afi", ConvertOptions(output_format="afi"))

            self.assertFalse(result.success)
            self.assertFalse(any(root.glob("sample*" + ".afi")))
            self.assertFalse(any(root.glob("sample_C*.svs")))
            self.assertFalse(any(root.glob("*.part")))

    def test_brightfield_is_not_eligible_for_fluorescence_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            sample = root / "brightfield.kfb"
            create_sample_kfb(sample)
            with KfbSlideSource(sample) as source:
                allowed, reasons = output_eligibility(source)

        self.assertEqual(set(allowed), {"ome_tiff", "svs"})
        self.assertIn("fluorescence_svs", reasons)
        self.assertIn("afi", reasons)


if __name__ == "__main__":
    unittest.main()
