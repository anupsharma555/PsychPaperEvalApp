from __future__ import annotations

import os
from pathlib import Path
import sys

from setuptools import setup

ROOT = Path(__file__).resolve().parents[1]
APP_NAME = os.environ.get("PAPER_EVAL_APP_NAME", "PaperEval")
APP_SCRIPT = str(ROOT / "scripts" / "paper_eval_gui.py")
ICON_PATH = str(ROOT / "PaperEval.icns")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OPTIONS = {
    "argv_emulation": False,
    "iconfile": ICON_PATH,
    "packages": ["desktop_legacy"],
    "includes": ["tkinter"],
    "plist": {
        "CFBundleIdentifier": "com.papereval.app",
        "CFBundleDisplayName": APP_NAME,
        "CFBundleName": APP_NAME,
        "CFBundleVersion": "0.3",
        "CFBundleShortVersionString": "0.3",
        "LSMinimumSystemVersion": "12.0",
        "LSEnvironment": {
            "PAPER_EVAL_ROOT": str(ROOT),
        },
    },
}

setup(
    name=APP_NAME,
    app=[APP_SCRIPT],
    options={"py2app": OPTIONS},
    setup_requires=["py2app"],
)
