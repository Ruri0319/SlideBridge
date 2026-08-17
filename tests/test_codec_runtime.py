from __future__ import annotations

import unittest

import imagecodecs
import numpy as np


class CodecRuntimeTests(unittest.TestCase):
    def test_runtime_decodes_baseline_jpeg_jxl_and_high_bit_lossless_jpeg(self) -> None:
        rgb = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
        jpeg = imagecodecs.jpeg8_encode(rgb, level=100, colorspace="rgb", outcolorspace="rgb")
        decoded_jpeg = imagecodecs.jpeg_decode(jpeg)
        self.assertEqual(decoded_jpeg.shape, rgb.shape)

        high_bit = np.arange(8 * 8, dtype=np.uint16).reshape(8, 8)
        jxl = imagecodecs.jpegxl_encode(high_bit)
        np.testing.assert_array_equal(imagecodecs.jpegxl_decode(jxl), high_bit)

        lossless_jpeg = imagecodecs.ljpeg_encode(high_bit)
        np.testing.assert_array_equal(imagecodecs.jpeg_decode(lossless_jpeg), high_bit)
        np.testing.assert_array_equal(imagecodecs.jpegsof3_decode(lossless_jpeg), high_bit)


if __name__ == "__main__":
    unittest.main()
