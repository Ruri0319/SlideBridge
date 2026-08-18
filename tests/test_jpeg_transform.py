from __future__ import annotations

from io import BytesIO
import os
import unittest

import numpy as np
from PIL import Image

from ibl2svs.jpeg_transform import transpose_jpeg


class JpegTransformTests(unittest.TestCase):
    def test_lossless_transpose_preserves_quantization_tables(self) -> None:
        source = np.zeros((16, 24, 3), dtype=np.uint8)
        source[:, :12] = (220, 30, 40)
        source[:, 12:] = (30, 200, 50)
        buffer = BytesIO()
        Image.fromarray(source).save(buffer, format="JPEG", quality=87, subsampling=0)
        encoded = buffer.getvalue()

        transformed, mode = transpose_jpeg(encoded)
        if mode != "lossless_transpose":
            if os.environ.get("IBL2SVS_TURBOJPEG"):
                self.fail("configured libjpeg-turbo did not perform a lossless transpose")
            self.skipTest("libjpeg-turbo is not available")

        with Image.open(BytesIO(encoded)) as before, Image.open(BytesIO(transformed)) as after:
            self.assertEqual(after.size, (16, 24))
            self.assertEqual(after.quantization.keys(), before.quantization.keys())
            for table_id, table in before.quantization.items():
                self.assertEqual(sorted(after.quantization[table_id]), sorted(table))
            pixels = np.asarray(after.convert("RGB"))
        self.assertGreater(int(pixels[3, 3, 0]), int(pixels[3, 3, 1]))
        self.assertGreater(int(pixels[20, 3, 1]), int(pixels[20, 3, 0]))


if __name__ == "__main__":
    unittest.main()
