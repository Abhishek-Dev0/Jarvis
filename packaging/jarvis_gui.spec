# PyInstaller spec for the JARVIS desktop app.
#
# --onedir (not --onefile): torch + faster-whisper + insightface/opencv make
# for a genuinely large bundle. --onefile self-extracts to a temp dir on
# EVERY launch, which at this size means a real multi-second-to-minutes
# delay every single time the app opens. --onedir starts instantly; Inno
# Setup (installer.iss) wraps this folder into one Jarvis-Setup.exe, which
# is what actually delivers "one downloadable file you install."
#
# Build with:  pyinstaller packaging/jarvis_gui.spec --noconfirm
# (run from the repo root, or pass --distpath/--workpath if not)

import os

from PyInstaller.utils.hooks import collect_data_files

REPO_ROOT = os.path.dirname(os.path.abspath(SPECPATH))

datas = [
    (os.path.join(REPO_ROOT, "jarvis", "data", "tokenizer.json"), "jarvis/data"),
    (os.path.join(REPO_ROOT, "jarvis", "checkpoints", "best.pt"), "jarvis/checkpoints"),
    (os.path.join(REPO_ROOT, "assets", "jarvis_cat.ico"), "assets"),
]

# Real bug found in the built app: faster_whisper's bundled VAD model
# (assets/silero_vad_v6.onnx, shipped INSIDE the pip package) was silently
# missing at runtime -- ONNXRuntimeError: NO_SUCHFILE. None of these four
# packages have a PyInstaller hook (confirmed: zero "hook-<name>" lines in
# a full build log), so hiddenimports alone was never enough -- it collects
# Python modules, not a package's bundled non-Python data files. Same root
# cause almost certainly affects the others; collect_data_files() is the
# correct general fix, not a per-file patch.
for _pkg in ("faster_whisper", "speechbrain", "kokoro_onnx", "insightface"):
    try:
        datas += collect_data_files(_pkg)
    except Exception:
        pass

# jarvis/data/models/ (Piper/Kokoro TTS weights) — bundled whole so voice
# output works out of the box with no first-run download. Large (several
# hundred MB); skip by deleting this block if you'd rather ship without
# voice and let users fetch weights on first use instead.
models_dir = os.path.join(REPO_ROOT, "jarvis", "data", "models")
if os.path.isdir(models_dir):
    for root, _dirs, files in os.walk(models_dir):
        for fname in files:
            src = os.path.join(root, fname)
            rel = os.path.relpath(root, REPO_ROOT)
            datas.append((src, rel.replace(os.sep, "/")))

hiddenimports = [
    # These pull in dynamic/plugin-style imports PyInstaller's static
    # analysis can miss. Expect to extend this list on the first real build
    # — that's normal for a stack this size, not a sign something's wrong.
    "faster_whisper",
    "kokoro_onnx",
    "speechbrain",
    "insightface",
    "onnxruntime",
    "cv2",
    "yfinance",
    "skimage",
    "pypdf",
    "docx",
    # ambient vision (modules/scene_watch.py) + market_analysis.py's
    # ml_signal + the GUI's Markets tab, added after the first real build
    "transformers",
    "timm",
    "einops",
    "sklearn",
    "matplotlib",
    "matplotlib.backends.backend_qtagg",
]

block_cipher = None

a = Analysis(
    [os.path.join(REPO_ROOT, "packaging", "jarvis_gui_launcher.py")],
    pathex=[REPO_ROOT],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Jarvis",
    debug=False,
    strip=False,
    upx=False,
    console=False,  # windowed app — no console popup
    icon=os.path.join(REPO_ROOT, "assets", "jarvis_cat.ico"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="Jarvis",
)
