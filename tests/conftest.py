"""
conftest.py — project-wide pytest configuration.

Adds both the project root and the audio_features package directory to
sys.path, and pre-registers the audio_features package in sys.modules so
that relative imports inside the package (e.g. `from .utils import EPS`)
resolve correctly when modules are imported from tests.
"""
import sys
import importlib.util as ilu
from pathlib import Path

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent   # .../Song Analysis/
PACKAGE_DIR  = PROJECT_ROOT / "audio_features"          # .../Song Analysis/audio_features/

# Add both to sys.path so both import styles work:
#   import audio_features.X      (package form)
#   import X                     (flat form, for modules outside the package)
for p in (str(PROJECT_ROOT), str(PACKAGE_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

# ---------------------------------------------------------------------------
# Pre-register the audio_features package so relative imports resolve.
# This is necessary when pytest collects modules that live inside the
# package but are imported without a proper package install (no pip install -e .).
# ---------------------------------------------------------------------------
if "audio_features" not in sys.modules and PACKAGE_DIR.exists():
    init_path = PACKAGE_DIR / "__init__.py"

    if init_path.exists():
        # Normal package with __init__.py
        spec = ilu.spec_from_file_location(
            "audio_features",
            str(init_path),
            submodule_search_locations=[str(PACKAGE_DIR)],
        )
    else:
        # Namespace package (no __init__.py) — create a minimal package entry
        spec = ilu.spec_from_file_location(
            "audio_features",
            str(PACKAGE_DIR),
            submodule_search_locations=[str(PACKAGE_DIR)],
        )

    if spec is not None:
        pkg = ilu.module_from_spec(spec)
        pkg.__path__ = [str(PACKAGE_DIR)]
        pkg.__package__ = "audio_features"
        sys.modules["audio_features"] = pkg
        try:
            spec.loader.exec_module(pkg)
        except (AttributeError, TypeError):
            pass   # namespace package — no loader to exec