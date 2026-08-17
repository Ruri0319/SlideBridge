from __future__ import annotations

import unittest
from pathlib import Path

from ibl2svs.kfb_source import KfbSlideSource


REAL_SAMPLE = Path("/Users/lidongshen/Desktop/test/2026-08-14_19_40_48.kfbf")


@unittest.skipUnless(REAL_SAMPLE.is_file(), "local KFBF 2.1 acceptance sample is unavailable")
class KfbRealSampleTests(unittest.TestCase):
    def test_recovers_verified_header_levels_and_resources(self) -> None:
        with KfbSlideSource(REAL_SAMPLE) as source:
            self.assertEqual(source.source_container, "kfbf")
            self.assertEqual(source.source_version, "2.1")
            self.assertEqual(source.compatibility_level, "sample_verified")
            self.assertEqual((source.width, source.height), (15750, 26639))
            self.assertEqual(source.tile_count, 8654)
            self.assertEqual(source.tile_size, 256)
            self.assertAlmostEqual(source.mpp, 0.251256, places=5)
            self.assertEqual(
                source.level_dimensions,
                [
                    (15750, 26639), (7875, 13319), (3937, 6659), (1968, 3329),
                    (992, 1680), (496, 840), (248, 420), (124, 210), (62, 105),
                    (31, 52), (15, 26), (7, 13), (3, 6), (1, 3), (1, 1), (1, 1),
                ],
            )
            self.assertEqual(
                source.native_resource_dimensions,
                {"thumbnail": (128, 212), "macro": (1868, 880), "label": (1076, 928)},
            )


if __name__ == "__main__":
    unittest.main()
