# -*- mode: python ; coding: utf-8 -*-

import ctypes.util
import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules


ROOT = Path.cwd()

hiddenimports = sorted(
    set(
        collect_submodules("PIL")
        + collect_submodules("imagecodecs")
        + [
            "numpy",
            "numpy.core",
            "numpy._core",
            "psutil",
            "tifffile",
        ]
    )
)

binaries = collect_dynamic_libs("imagecodecs")
datas = collect_data_files("tifffile") + [("README.txt", "."), ("THIRD_PARTY_NOTICES.md", ".")]

turbojpeg_candidates = [os.environ.get("IBL2SVS_TURBOJPEG", "")]
discovered_turbojpeg = ctypes.util.find_library("turbojpeg")
if discovered_turbojpeg:
    turbojpeg_candidates.append(discovered_turbojpeg)
turbojpeg_candidates.extend(
    [
        "/opt/homebrew/lib/libturbojpeg.dylib",
        "/opt/homebrew/opt/jpeg-turbo/lib/libturbojpeg.dylib",
        "/usr/local/lib/libturbojpeg.dylib",
        "/usr/lib/x86_64-linux-gnu/libturbojpeg.so",
        r"C:\libjpeg-turbo64\bin\turbojpeg.dll",
        r"C:\Program Files\libjpeg-turbo\bin\turbojpeg.dll",
    ]
)
for turbojpeg_path in turbojpeg_candidates:
    if turbojpeg_path and Path(turbojpeg_path).is_file():
        binaries.append((str(Path(turbojpeg_path).resolve()), "ibl2svs"))
        break

a = Analysis(
    ["worker_main.py"],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "IPython",
        "lxml",
        "matplotlib",
        "pandas",
        "pyvips",
        "pytest",
        "scipy",
        "torch",
        "zarr",
        "tkinter",
        "ttkbootstrap",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="slidebridge-worker",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
)
