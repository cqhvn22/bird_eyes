"""Portable source-checkout launcher; installed command calls perch.cli directly."""
from pathlib import Path
import sys

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    from perch.cli import camera_main
    raise SystemExit(camera_main())
