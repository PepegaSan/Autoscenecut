# -*- mode: python ; coding: utf-8 -*-
# Build (from this folder): pyinstaller scenecut_gui.spec
# Requires: pip install pyinstaller pyinstaller-hooks-contrib

import pathlib

# PyInstaller sets SPECPATH to the folder that contains the .spec file (not the file path).
try:
    _sp = pathlib.Path(SPECPATH).resolve()
    spec_dir = _sp if _sp.is_dir() else _sp.parent
except NameError:
    spec_dir = pathlib.Path('.').resolve()

root = str(spec_dir)

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_dynamic_libs

try:
    import tkinterdnd2  # noqa: F401 — nur prüfen, dass Build-Python das Paket hat
except ImportError as e:
    raise SystemExit(
        'tkinterdnd2 fehlt in diesem Python. Gleiche Umgebung wie PyInstaller nutzen:\n'
        '  pip install tkinterdnd2'
    ) from e

datas = [
    (str(spec_dir / 'config_nvidia.example.ini'), '.'),
    (str(spec_dir / 'yamnet.onnx'), '.'),
    (str(spec_dir / 'yamnet_class_map.csv'), '.'),
]
datas += collect_data_files('tkinterdnd2')
binaries = []
binaries += collect_dynamic_libs('tkinterdnd2')
hiddenimports = [
    'autocut_nvidia',
    'analyzer_nvidia',
    'PIL._tkinter_finder',
    'tkinterdnd2',
    'tkinterdnd2.TkinterDnD',
]

for pkg in ('customtkinter',):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

block_cipher = None

a = Analysis(
    [str(spec_dir / 'gui_nvidia.py')],
    pathex=[root],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='ScenecutNVIDIA',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
