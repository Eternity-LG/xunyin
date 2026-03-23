# -*- mode: python ; coding: utf-8 -*-

import whisper
import os

# 获取 whisper 模型路径
whisper_path = os.path.dirname(whisper.__file__)

a = Analysis(
    ['xunyin.py'],
    pathex=[],
    binaries=[],
    datas=[
        # 包含 whisper 的 assets
        (whisper_path, 'whisper'),
    ],
    hiddenimports=[
        'whisper',
        'whisper.tokenizer',
        'whisper.decoding',
        'whisper.audio',
        'scipy.signal',
        'pyperclip',
        'PyQt6',
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        'PyQt6.QtWidgets',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='迅音',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # 无控制台窗口
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # 可以添加图标文件路径
)
