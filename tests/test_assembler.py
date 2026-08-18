from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image

from ibl2svs.assembler import DensePyramidDrive, PILImageSource, iter_source_tiles


class AssemblerTests(unittest.TestCase):
    def test_dense_pyramid_buffer_respects_memory_headroom(self) -> None:
        slide = SimpleNamespace(width=100_000, height=100_000)
        allocated_shapes: list[tuple[int, ...]] = []

        def fake_zeros(shape, dtype):
            allocated_shapes.append(shape)
            return np.empty((1, 1, 3), dtype=dtype)

        with mock.patch("ibl2svs.assembler.np.zeros", side_effect=fake_zeros):
            drive = DensePyramidDrive(slide, 256, 1024, memory_budget_mb=1024)

        requested_bytes = int(np.prod(allocated_shapes[0]))
        self.assertLessEqual(requested_bytes, int(1024 * 0.55 * 1024 * 1024))
        self.assertGreaterEqual(drive.downsample_factor, 8)

    def test_iter_source_tiles_preserves_global_row_major_order(self) -> None:
        image = np.zeros((16, 32, 3), dtype=np.uint8)
        tile_index = 1
        for tile_y in range(0, 16, 8):
            for tile_x in range(0, 32, 8):
                image[tile_y : tile_y + 8, tile_x : tile_x + 8] = (tile_index, 0, 0)
                tile_index += 1

        source = PILImageSource(Image.fromarray(image))
        tiles = list(iter_source_tiles(source, 8, chunk_size=16))

        self.assertEqual([int(tile[0, 0, 0]) for tile in tiles], [1, 2, 3, 4, 5, 6, 7, 8])


if __name__ == "__main__":
    unittest.main()
