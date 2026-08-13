from __future__ import annotations

import os
from pathlib import Path
import sys


def configure_libvips() -> None:
    candidates: list[Path] = []
    env_dir = os.getenv("SLIDEBRIDGE_LIBVIPS_DIR", os.getenv("IBL2SVS_LIBVIPS_DIR"))
    if env_dir:
        candidates.append(Path(env_dir))

    current = Path(__file__).resolve().parent
    candidates.extend(
        [
            current / "libvips",
            current.parent / "libvips",
        ]
    )

    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        candidates.extend([exe_dir / "libvips", exe_dir])

    for candidate in candidates:
        if not candidate.exists():
            continue

        os.environ.setdefault("VIPS_WARNING", "0")
        path_str = str(candidate)
        if os.name == "nt":
            try:
                os.add_dll_directory(path_str)
            except (AttributeError, FileNotFoundError, OSError):
                pass
        if path_str not in os.environ.get("PATH", ""):
            os.environ["PATH"] = path_str + os.pathsep + os.environ.get("PATH", "")
        break


def load_pyvips():
    configure_libvips()
    try:
        import pyvips
    except Exception as exc:  # pragma: no cover - depends on local runtime
        raise RuntimeError(
            "缺少 pyvips/libvips 运行时，无法使用高性能 TIFF/SVS 写出后端。"
        ) from exc
    return pyvips
