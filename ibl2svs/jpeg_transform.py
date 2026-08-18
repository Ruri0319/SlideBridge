from __future__ import annotations

import ctypes
import ctypes.util
from functools import lru_cache
from io import BytesIO
import os
from pathlib import Path
import sys
import threading

import numpy as np
from PIL import Image


TJINIT_TRANSFORM = 2
TJXOP_TRANSPOSE = 3
TJXOPT_PERFECT = 1


class _CroppingRegion(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_int),
        ("y", ctypes.c_int),
        ("w", ctypes.c_int),
        ("h", ctypes.c_int),
    ]


class _Transform(ctypes.Structure):
    _fields_ = [
        ("region", _CroppingRegion),
        ("op", ctypes.c_int),
        ("options", ctypes.c_int),
        ("data", ctypes.c_void_p),
        ("custom_filter", ctypes.c_void_p),
    ]


def _valid_jpeg(data: bytes) -> bool:
    return len(data) >= 4 and data[:3] == b"\xff\xd8\xff" and data[-2:] == b"\xff\xd9"


def _library_candidates() -> list[str]:
    names = ("turbojpeg.dll", "libturbojpeg.dll", "libturbojpeg.dylib", "libturbojpeg.so")
    roots = [Path(__file__).resolve().parent]
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        roots.extend((Path(bundle_root), Path(bundle_root) / "ibl2svs"))
    candidates: list[str] = []
    configured = os.environ.get("IBL2SVS_TURBOJPEG", "").strip()
    if configured:
        candidates.append(configured)
    for root in roots:
        candidates.extend(str(root / name) for name in names)
        candidates.extend(str(path) for path in root.glob("*turbojpeg*"))
    discovered = ctypes.util.find_library("turbojpeg")
    if discovered:
        candidates.append(discovered)
    candidates.extend(
        [
            "/opt/homebrew/lib/libturbojpeg.dylib",
            "/opt/homebrew/opt/jpeg-turbo/lib/libturbojpeg.dylib",
            "/usr/local/lib/libturbojpeg.dylib",
            "/usr/lib/x86_64-linux-gnu/libturbojpeg.so",
            "/usr/local/lib/libturbojpeg.so",
            r"C:\libjpeg-turbo64\bin\turbojpeg.dll",
            r"C:\Program Files\libjpeg-turbo\bin\turbojpeg.dll",
        ]
    )
    return list(dict.fromkeys(candidates))


@lru_cache(maxsize=1)
def _load_turbojpeg():
    for candidate in _library_candidates():
        try:
            library = ctypes.CDLL(candidate)
        except OSError:
            continue
        if all(hasattr(library, name) for name in ("tj3Init", "tj3Destroy", "tj3Transform", "tj3Free")):
            return library
    return None


class _TurboJpegTransformer:
    def __init__(self, library):
        self._library = library
        library.tj3Init.argtypes = [ctypes.c_int]
        library.tj3Init.restype = ctypes.c_void_p
        library.tj3Destroy.argtypes = [ctypes.c_void_p]
        library.tj3Destroy.restype = None
        library.tj3Transform.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(_Transform),
        ]
        library.tj3Transform.restype = ctypes.c_int
        library.tj3Free.argtypes = [ctypes.c_void_p]
        library.tj3Free.restype = None
        self._handle = library.tj3Init(TJINIT_TRANSFORM)
        if not self._handle:
            raise RuntimeError("无法初始化 libjpeg-turbo 转换器")

    def close(self) -> None:
        handle = self._handle
        if handle:
            self._handle = None
            self._library.tj3Destroy(handle)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def transpose(self, data: bytes) -> bytes | None:
        source = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
        destinations = (ctypes.POINTER(ctypes.c_ubyte) * 1)()
        destination_sizes = (ctypes.c_size_t * 1)()
        transform = _Transform(
            region=_CroppingRegion(0, 0, 0, 0),
            op=TJXOP_TRANSPOSE,
            options=TJXOPT_PERFECT,
            data=None,
            custom_filter=None,
        )
        result = self._library.tj3Transform(
            self._handle,
            source,
            len(data),
            1,
            destinations,
            destination_sizes,
            ctypes.byref(transform),
        )
        if result != 0:
            if destinations[0]:
                self._library.tj3Free(destinations[0])
            return None
        if not destinations[0]:
            return None
        try:
            output = ctypes.string_at(destinations[0], destination_sizes[0])
        finally:
            self._library.tj3Free(destinations[0])
        return output if _valid_jpeg(output) else None


_thread_local = threading.local()


def _thread_transformer() -> _TurboJpegTransformer | None:
    transformer = getattr(_thread_local, "transformer", None)
    if transformer is not None:
        return transformer
    library = _load_turbojpeg()
    if library is None:
        return None
    try:
        transformer = _TurboJpegTransformer(library)
    except (OSError, RuntimeError):
        return None
    _thread_local.transformer = transformer
    return transformer


def _lossless_transpose(data: bytes) -> bytes | None:
    transformer = _thread_transformer()
    return transformer.transpose(data) if transformer is not None else None


def transpose_jpeg(data: bytes, *, quality: int = 100) -> tuple[bytes, str]:
    """Transpose a JPEG using DCT coefficients when libjpeg-turbo is available."""
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
