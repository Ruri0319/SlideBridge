from __future__ import annotations

import unittest

from ibl2svs.native_jpeg import jpeg_dimensions


class NativeJpegTests(unittest.TestCase):
    def test_dimensions_ignore_sof_bytes_inside_app_payload(self) -> None:
        app_payload = b"prefix\xff\xc0not-a-marker"
        app = b"\xff\xe0" + (len(app_payload) + 2).to_bytes(2, "big") + app_payload
        sof = (
            b"\xff\xc0\x00\x11\x08"
            + (123).to_bytes(2, "big")
            + (456).to_bytes(2, "big")
            + b"\x03\x01\x11\x00\x02\x11\x00\x03\x11\x00"
        )

        self.assertEqual(jpeg_dimensions(b"\xff\xd8" + app + sof + b"\xff\xd9"), (456, 123))

    def test_dimensions_reject_missing_sof(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "缺少 SOF"):
            jpeg_dimensions(b"\xff\xd8\xff\xd9")


if __name__ == "__main__":
    unittest.main()
