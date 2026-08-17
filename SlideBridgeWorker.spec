# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules


ROOT = Path.cwd()

hiddenimports = sorted(
    set(
        collect_submodules("PIL")
        + collect_submodules("tifffile")
        + collect_submodules("imagecodecs")
        + collect_submodules("pyvips")
        + [
            "numpy",
            "numpy.core",
            "numpy._core",
            "psutil",
            "pyvips",
        ]
    )
)

binaries = collect_dynamic_libs("imagecodecs") + collect_dynamic_libs("pyvips")
datas = collect_data_files("tifffile") + [("README.txt", ".")]
for jpegtran_path in (ROOT / "ibl2svs" / "jpegtran.exe", ROOT / "ibl2svs" / "jpegtran"):
    if jpegtran_path.is_file():
        datas.append((str(jpegtran_path), "ibl2svs"))

a = Analysis(
    ["worker_main.py"],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "ttkbootstrap"],
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
