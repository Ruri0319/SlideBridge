from __future__ import annotations

from io import BytesIO
from pathlib import Path
import os
import shutil
import subprocess
from functools import lru_cache

import numpy as np
from PIL import Image


def _valid_jpeg(data: bytes) -> bool:
    return len(data) >= 4 and data[:3] == b"\xff\xd8\xff" and data[-2:] == b"\xff\xd9"


@lru_cache(maxsize=1)
def _find_jpegtran() -> str | None:
    configured = os.environ.get("IBL2SVS_JPEGTRAN", "").strip()
    if configured:
        path = Path(configured)
        if path.is_file():
            return str(path)

    bundled = Path(__file__).with_name("jpegtran.exe")
    if bundled.is_file():
        return str(bundled)

    bundled_unix = Path(__file__).with_name("jpegtran")
    if bundled_unix.is_file():
        return str(bundled_unix)

    return shutil.which("jpegtran")


def _lossless_transpose(data: bytes) -> bytes | None:
    executable = _find_jpegtran()
    if executable is None:
        return None

    try:
        completed = subprocess.run(
            [executable, "-copy", "all", "-perfect", "-transpose"],
            input=data,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0 or not _valid_jpeg(completed.stdout):
        return None
    return bytes(completed.stdout)


def transpose_jpeg(data: bytes, *, quality: int = 100) -> tuple[bytes, str]:
    """Transpose a JPEG, preferring a coefficient-domain lossless transform.

    Punuoxi stores each main-image JPEG with its pixel axes exchanged. The
    bundled/system jpegtran path preserves the original DCT coefficients. A
    Pillow/imagecodecs path is kept as a portable fallback for environments
    where the helper is not available.
    """

    lossless = _lossless_transpose(data)
    if lossless is not None:
        return lossless, "lossless_transpose"

    with Image.open(BytesIO(data)) as image:
        transposed = np.asarray(image.convert("RGB").transpose(Image.Transpose.TRANSPOSE), dtype=np.uint8)

    try:
        from imagecodecs import jpeg8_encode

        encoded = jpeg8_encode(
            np.ascontiguousarray(transposed),
            level=quality,
            colorspace="rgb",
            outcolorspace="rgb",
            optimize=False,
        )
    except ImportError:
        output = BytesIO()
        Image.fromarray(transposed, mode="RGB").save(
            output,
            format="JPEG",
            quality=quality,
            subsampling=0,
            optimize=False,
        )
        encoded = output.getvalue()
    return bytes(encoded), "reencoded"
