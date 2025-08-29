# scripts/run_crawling_instagram.py
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODS = ROOT / "src" / "modules"
if str(MODS) not in sys.path:
    sys.path.insert(0, str(MODS))

WF = ROOT / "src" / "utils" / "crawling_instagram.py"
runpy.run_path(WF.as_posix(), run_name="__main__")
