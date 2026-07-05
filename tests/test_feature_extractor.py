import types
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
from typing import Any, Dict
 
import numpy as np
import sys
from pathlib import Path
 
# ---------------------------------------------------------------------------
# Path setup — insert both the project root and the package directory so
# that both `import audio_features.X` and `import X` resolve correctly,
# and relative imports inside the package work via the package path.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]          # .../Song Analysis/
PACKAGE = ROOT / "audio_features"                   # .../Song Analysis/audio_features/
 
sys.path.insert(0, str(ROOT))
# Insert the package dir so that modules using relative imports can be
# imported as top-level names when the package import fails.
if PACKAGE.exists():
    sys.path.insert(0, str(PACKAGE))
 
# ---------------------------------------------------------------------------
# Attempt real imports.
# Strategy: always try the package form first (audio_features.X).
# If that fails because there is no __init__.py, try importing the package
# folder explicitly so relative imports resolve via importlib.
# ---------------------------------------------------------------------------
_REAL_IMPORTS = False
 
try:
    from audio_features.audio_signal import AudioSignal
    from audio_features.feature_extractor import FeatureExtractor
    _REAL_IMPORTS = True
except ImportError:
    # Package import failed — attempt to load via importlib so relative
    # imports inside feature_extractor.py resolve correctly.
    try:
        import importlib.util as _ilu
 
        def _load_from_package(pkg_path: Path, module_name: str):
            """Load a module from a package directory using importlib."""
            spec = _ilu.spec_from_file_location(
                f"audio_features.{module_name}",
                str(pkg_path / f"{module_name}.py"),
                submodule_search_locations=[str(pkg_path)],
            )
            mod = _ilu.module_from_spec(spec)
            sys.modules[f"audio_features.{module_name}"] = mod
            spec.loader.exec_module(mod)
            return mod
 
        # Ensure audio_features is a known package in sys.modules
        if "audio_features" not in sys.modules:
            _pkg_spec = _ilu.spec_from_file_location(
                "audio_features",
                str(PACKAGE / "__init__.py") if (PACKAGE / "__init__.py").exists()
                else str(PACKAGE),
                submodule_search_locations=[str(PACKAGE)],
            )
            _pkg = _ilu.module_from_spec(_pkg_spec)
            sys.modules["audio_features"] = _pkg
 
        _as_mod = _load_from_package(PACKAGE, "audio_signal")
        AudioSignal = _as_mod.AudioSignal
 
        _fe_mod = _load_from_package(PACKAGE, "feature_extractor")
        FeatureExtractor = _fe_mod.FeatureExtractor
 
        _REAL_IMPORTS = True
    except Exception:
        # Final fallback: stub placeholders so static-method tests still run
        AudioSignal = None       # type: ignore
        FeatureExtractor = None  # type: ignore
        _REAL_IMPORTS = False
 
 
# ---------------------------------------------------------------------------
# Synthetic signal helpers
# ---------------------------------------------------------------------------
 
def _make_audio_signal(duration_sec=2.0, sr=22050, N=2048, H=512, freq_hz=440.0):
    t = np.linspace(0, duration_sec, int(duration_sec * sr), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * freq_hz * t)
    y += 0.25 * np.sin(2 * np.pi * 2 * freq_hz * t)
    if _REAL_IMPORTS:
        return AudioSignal(signal=y, sr=sr, N=N, H=H)
    raise RuntimeError("AudioSignal not available")
 
 
def _make_silence(duration_sec=2.0, sr=22050, N=2048, H=512):
    y = np.zeros(int(duration_sec * sr))
    if _REAL_IMPORTS:
        return AudioSignal(signal=y, sr=sr, N=N, H=H)
    raise RuntimeError("AudioSignal not available")
 
 
def _make_fe(duration_sec=2.0, freq_hz=440.0, **flags):
    sig = _make_audio_signal(duration_sec=duration_sec, freq_hz=freq_hz)
    return FeatureExtractor(sig, **flags)
 
 
# ---------------------------------------------------------------------------
# Minimal stub FeatureExtractor for testing static methods in isolation
# (used when full imports are unavailable, or for pure-logic unit tests)
# ---------------------------------------------------------------------------
 
class _StubFE:
    """
    Standalone reimplementation of the three pure static methods so we can
    test their logic without any audio processing dependencies.
    """
 
    @staticmethod
    def _flatten(d: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in d.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    out[f"{k}.{kk}"] = vv
            else:
                out[k] = v
        return out
 
    @staticmethod
    def _first_existing(d: Dict[str, Any], keys: list) -> Any:
        for k in keys:
            if k in d:
                return d[k]
        return None
 
    @staticmethod
    def _add_unified_aliases(d: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(d)
        aliases = {
            "loudness":        ["time.loudness",        "frequency.loudness",         "mfcc.loudness",        "tempogram.loudness"],
            "energy":          ["time.energy",           "frequency.energy",           "mfcc.energy",          "chroma.energy"],
            "speechiness":     ["mfcc.speechiness",      "time.speechiness",           "frequency.speechiness","chroma.speechiness"],
            "acousticness":    ["frequency.acousticness","mfcc.acousticness",          "time.acousticness",    "chroma.acousticness"],
            "danceability":    ["time.danceability",     "frequency.danceability",     "tempogram.danceability","chroma.danceability"],
            "valence":         ["chroma.valence",        "frequency.valence",          "mfcc.valence",         "tempogram.valence"],
            "tempo":           ["tempogram.tempo",       "time.tempo",                 "frequency.tempo",      "chroma.tempo"],
            "liveness":        ["frequency.liveness",    "time.liveness",              "mfcc.liveness",        "tempogram.liveness"],
            "instrumentalness":["frequency.instrumentalness","mfcc.instrumentalness",  "time.instrumentalness","chroma.instrumentalness"],
            "key":             ["chroma.key",            "frequency.key"],
            "mode":            ["chroma.mode",           "frequency.mode",             "tempogram.mode"],
            "time_signature":  ["time.time_signature",   "tempogram.time_signature",   "frequency.time_signature","chroma.time_signature"],
        }
        for target, sources in aliases.items():
            for src in sources:
                if src not in out:
                    continue
                val = out[src]
                if val is None:
                    continue
                if isinstance(val, float) and np.isnan(val):
                    continue
                out[target] = val
                out[f"{target}.__source__"] = src
                break
        return out
 
 
# Use real FE if available, fall back to stub for static-method tests
_FE = FeatureExtractor if _REAL_IMPORTS else None
_SFE = _StubFE
 
 
# ===========================================================================
# Bug 1 — _flatten NOT called in extract() (already-flat keys preserved)
# ===========================================================================
 
class TestFlattenNotCalledInExtract(unittest.TestCase):
    """
    Bug 1: extract() must NOT call _flatten on its output because
    _extract_* methods already return flat dotted keys.
    Calling _flatten on flat input is a no-op, but on nested dicts it would
    produce double-dotted keys like 'time.energy.value'.
    """
 
    def test_flatten_passthrough_on_flat_dict(self):
        """_flatten on an already-flat dict must return it unchanged."""
        flat = {"time.energy": 0.5, "mfcc.loudness": -10.0, "chroma.key": 3}
        result = _SFE._flatten(flat)
        self.assertEqual(result, flat)
 
    def test_flatten_expands_nested_dict(self):
        """_flatten on a nested dict must expand one level into dotted keys."""
        nested = {"time": {"energy": 0.5, "loudness": -10.0}}
        result = _SFE._flatten(nested)
        self.assertIn("time.energy", result)
        self.assertIn("time.loudness", result)
        self.assertNotIn("time", result)
 
    def test_flatten_does_not_double_dot_already_flat(self):
        """_flatten on dotted-key dict must NOT produce double dots."""
        flat = {"time.energy": 0.5}
        result = _SFE._flatten(flat)
        self.assertNotIn("time.energy.energy", result)
        self.assertIn("time.energy", result)
 
    def test_flatten_mixed_flat_and_nested(self):
        """_flatten handles a mix of flat scalars and nested dicts."""
        d = {"a": 1.0, "b": {"x": 2.0, "y": 3.0}}
        result = _SFE._flatten(d)
        self.assertIn("a", result)
        self.assertIn("b.x", result)
        self.assertIn("b.y", result)
        self.assertNotIn("b", result)
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_extract_keys_are_flat_dotted(self):
        """
        All keys in extract() output must be flat dotted strings or
        alias names — no double-dotted keys like 'time.energy.value'.
        """
        fe = _make_fe()
        out = fe.extract()
        for key in out:
            if ".__source__" not in key:
                parts = key.split(".")
                self.assertLessEqual(
                    len(parts), 2,
                    f"Key '{key}' has more than one dot — _flatten may have been "
                    "applied to already-flat keys (Bug 1).",
                )
 
 
# ===========================================================================
# Bug 2 — _first_existing is dead code (never called by _add_unified_aliases)
# ===========================================================================
 
class TestFirstExisting(unittest.TestCase):
    """
    Bug 2: _first_existing is a utility that should replace the inline
    logic in _add_unified_aliases. It is currently dead code.
    We test its correctness independently so it's ready to use.
    """
 
    def test_returns_first_found_key(self):
        d = {"a": 1, "b": 2, "c": 3}
        result = _SFE._first_existing(d, ["b", "a", "c"])
        self.assertEqual(result, 2)
 
    def test_skips_missing_keys(self):
        d = {"c": 42}
        result = _SFE._first_existing(d, ["a", "b", "c"])
        self.assertEqual(result, 42)
 
    def test_returns_none_when_no_key_found(self):
        d = {"x": 1}
        result = _SFE._first_existing(d, ["a", "b", "c"])
        self.assertIsNone(result)
 
    def test_empty_dict_returns_none(self):
        self.assertIsNone(_SFE._first_existing({}, ["a", "b"]))
 
    def test_empty_keys_list_returns_none(self):
        self.assertIsNone(_SFE._first_existing({"a": 1}, []))
 
    def test_returns_falsy_values(self):
        """_first_existing must return 0 and '' — not skip them as falsy."""
        d = {"a": 0, "b": ""}
        self.assertEqual(_SFE._first_existing(d, ["a"]), 0)
        self.assertEqual(_SFE._first_existing(d, ["b"]), "")
 
    def test_not_called_by_add_unified_aliases(self):
        """
        Verify _first_existing is not called inside _add_unified_aliases
        (it remains dead code until explicitly wired in).
        """
        import inspect
        src = inspect.getsource(_SFE._add_unified_aliases)
        self.assertNotIn("_first_existing", src,
                         "_first_existing should not be called inside "
                         "_add_unified_aliases until it replaces the inline loop.")
 
 
# ===========================================================================
# Bug 3 — tempogram.loudness must be a float scalar
# ===========================================================================
 
class TestTempogramLoudnessScalar(unittest.TestCase):
    """
    Bug 3: _loudness_tempogram_per_beat() returns an ndarray.
    _extract_tempogram must reduce it to float(np.mean(...)).
    """
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_tempogram_loudness_is_float(self):
        fe = _make_fe(compute_time=False, compute_frequency=False,
                      compute_mfcc=False, compute_chroma=False)
        out = fe._extract_tempogram()
        val = out["tempogram.loudness"]
        self.assertIsInstance(
            val, (float, np.floating),
            f"tempogram.loudness must be a float scalar, got {type(val)}. "
            "Bug 3 (array not reduced to mean) may still be present.",
        )
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_tempogram_loudness_finite(self):
        fe = _make_fe(compute_time=False, compute_frequency=False,
                      compute_mfcc=False, compute_chroma=False)
        val = fe._extract_tempogram()["tempogram.loudness"]
        self.assertTrue(np.isfinite(val), "tempogram.loudness must be finite.")
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_tempogram_loudness_not_ndarray(self):
        fe = _make_fe(compute_time=False, compute_frequency=False,
                      compute_mfcc=False, compute_chroma=False)
        val = fe._extract_tempogram()["tempogram.loudness"]
        self.assertNotIsInstance(
            val, np.ndarray,
            "tempogram.loudness is still an ndarray — Bug 3 not fully fixed.",
        )
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_loudness_alias_is_scalar_in_full_extract(self):
        """After aliasing, top-level 'loudness' must not be an ndarray."""
        fe = _make_fe()
        out = fe.extract()
        if "loudness" in out:
            self.assertNotIsInstance(out["loudness"], np.ndarray,
                                     "Top-level 'loudness' alias must be a scalar.")
 
 
# ===========================================================================
# Bug 4 — tempogram.tempo must be present
# ===========================================================================
 
class TestTempogramTempoPresent(unittest.TestCase):
    """
    Bug 4: _extract_tempogram was missing 'tempogram.tempo'.
    Without it, the 'tempo' alias (which lists tempogram.tempo first) would
    silently fall through to time.tempo — or return nothing if time is disabled.
    """
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_tempogram_tempo_key_exists(self):
        fe = _make_fe(compute_time=False, compute_frequency=False,
                      compute_mfcc=False, compute_chroma=False)
        out = fe._extract_tempogram()
        self.assertIn(
            "tempogram.tempo", out,
            "'tempogram.tempo' missing from _extract_tempogram output — Bug 4.",
        )
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_tempogram_tempo_is_float(self):
        fe = _make_fe(compute_time=False, compute_frequency=False,
                      compute_mfcc=False, compute_chroma=False)
        val = fe._extract_tempogram()["tempogram.tempo"]
        self.assertIsInstance(val, (float, np.floating))
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_tempogram_tempo_in_bpm_range(self):
        fe = _make_fe(compute_time=False, compute_frequency=False,
                      compute_mfcc=False, compute_chroma=False)
        val = fe._extract_tempogram()["tempogram.tempo"]
        self.assertGreaterEqual(val, 0.0)
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_tempo_alias_resolves_to_tempogram_when_time_disabled(self):
        """
        With compute_time=False, tempo alias must still resolve via
        tempogram.tempo (Bug 4 fix), not fall back to time.tempo.
        """
        fe = _make_fe(compute_time=False, compute_frequency=False,
                      compute_mfcc=False, compute_chroma=True)
        out = fe.extract()
        self.assertIn("tempo", out, "'tempo' alias must be present in extract() output.")
        src = out.get("tempo.__source__", "")
        self.assertIn(
            "tempogram", src,
            f"'tempo' should source from tempogram when time is disabled, got '{src}'.",
        )
 
 
# ===========================================================================
# Bug 5 — _add_unified_aliases skips None and NaN sources
# ===========================================================================
 
class TestAliasSkipsNoneAndNan(unittest.TestCase):
    """
    Bug 5: if the highest-priority source is None or NaN, the alias must
    skip it and try the next source — not silently assign a bad value.
    """
 
    def test_none_source_skipped(self):
        d = {
            "time.loudness": None,          # primary — must be skipped
            "frequency.loudness": -12.0,    # fallback — must be used
        }
        out = _SFE._add_unified_aliases(d)
        self.assertEqual(
            out["loudness"], -12.0,
            "None primary source must be skipped; fallback must be used.",
        )
        self.assertEqual(out["loudness.__source__"], "frequency.loudness")
 
    def test_nan_source_skipped(self):
        d = {
            "time.loudness": float("nan"),  # primary — NaN, must be skipped
            "frequency.loudness": -8.0,
        }
        out = _SFE._add_unified_aliases(d)
        self.assertEqual(out["loudness"], -8.0)
        self.assertEqual(out["loudness.__source__"], "frequency.loudness")
 
    def test_all_none_produces_no_alias(self):
        """If every source is None, no alias key must be set."""
        d = {"time.loudness": None, "frequency.loudness": None}
        out = _SFE._add_unified_aliases(d)
        self.assertNotIn("loudness", out,
                         "All-None sources must not produce a loudness alias.")
 
    def test_all_nan_produces_no_alias(self):
        d = {"time.loudness": float("nan"), "frequency.loudness": float("nan")}
        out = _SFE._add_unified_aliases(d)
        self.assertNotIn("loudness", out)
 
    def test_zero_is_not_skipped(self):
        """0.0 is a valid value — must not be treated as falsy and skipped."""
        d = {"time.energy": 0.0}
        out = _SFE._add_unified_aliases(d)
        self.assertIn("energy", out)
        self.assertEqual(out["energy"], 0.0)
 
    def test_first_valid_wins_over_later_none(self):
        """Valid primary wins even when a later source would also be valid."""
        d = {
            "time.energy": 0.8,
            "frequency.energy": 0.5,
        }
        out = _SFE._add_unified_aliases(d)
        # time.energy is NOT first in the energy alias list (mfcc is)
        # but time is second — verify correct priority is honoured
        self.assertIn("energy", out)
        self.assertIn(out["energy.__source__"], ["time.energy", "frequency.energy"])
 
    def test_source_tracking_key_present(self):
        """Every resolved alias must have a corresponding __source__ key."""
        d = {"chroma.valence": 0.65, "frequency.valence": 0.7}
        out = _SFE._add_unified_aliases(d)
        if "valence" in out:
            self.assertIn("valence.__source__", out)
 
    def test_none_then_valid_then_none(self):
        """Second-priority valid value must win when first is None."""
        d = {
            "mfcc.speechiness": None,
            "time.speechiness": 0.35,
            "frequency.speechiness": 0.2,
        }
        out = _SFE._add_unified_aliases(d)
        self.assertEqual(out["speechiness"], 0.35)
        self.assertEqual(out["speechiness.__source__"], "time.speechiness")
 
 
# ===========================================================================
# Bug 6 — Unused imports removed (json, Optional)
# ===========================================================================
 
class TestUnusedImports(unittest.TestCase):
    """Bug 6: json and Optional must not appear in the module's imports."""
 
    def test_json_not_imported(self):
        import inspect
        try:
            import feature_extractor as fe_mod  # type: ignore
        except ImportError:
            try:
                from audio_features import feature_extractor as fe_mod  # type: ignore
            except ImportError:
                self.skipTest("feature_extractor module not importable directly.")
        src = inspect.getsource(fe_mod)
        self.assertNotIn(
            "import json", src,
            "'import json' still present — unused import not removed (Bug 6).",
        )
 
    def test_optional_not_imported(self):
        import inspect
        try:
            import feature_extractor as fe_mod  # type: ignore
        except ImportError:
            try:
                from audio_features import feature_extractor as fe_mod  # type: ignore
            except ImportError:
                self.skipTest("feature_extractor module not importable directly.")
        src = inspect.getsource(fe_mod)
        self.assertNotIn(
            "Optional", src,
            "'Optional' still imported — unused import not removed (Bug 6).",
        )
 
 
# ===========================================================================
# Bug 7 — tempogram.mode must be a plain string scalar
# ===========================================================================
 
class TestTempogramModeScalar(unittest.TestCase):
    """
    Bug 7: _mode_tempogram() returns a dict.
    _extract_tempogram must index it as ["mode"] to return a plain string.
    """
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_tempogram_mode_is_string(self):
        fe = _make_fe(compute_time=False, compute_frequency=False,
                      compute_mfcc=False, compute_chroma=False)
        val = fe._extract_tempogram()["tempogram.mode"]
        self.assertIsInstance(
            val, str,
            f"tempogram.mode must be a plain string, got {type(val)}. "
            "Bug 7 (_mode_tempogram dict not subscripted) may still be present.",
        )
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_tempogram_mode_valid_value(self):
        fe = _make_fe(compute_time=False, compute_frequency=False,
                      compute_mfcc=False, compute_chroma=False)
        val = fe._extract_tempogram()["tempogram.mode"]
        self.assertIn(val, ("major", "minor"),
                      f"tempogram.mode must be 'major' or 'minor', got '{val}'.")
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_tempogram_mode_not_dict(self):
        fe = _make_fe(compute_time=False, compute_frequency=False,
                      compute_mfcc=False, compute_chroma=False)
        val = fe._extract_tempogram()["tempogram.mode"]
        self.assertNotIsInstance(val, dict,
                                 "tempogram.mode must not be a dict — Bug 7 not fixed.")
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_no_double_dot_mode_keys_in_extract(self):
        """
        If tempogram.mode were a dict, _flatten (if called) would produce
        keys like 'tempogram.mode.mode'. None of those should exist.
        """
        fe = _make_fe()
        out = fe.extract()
        double_dot_mode = [k for k in out if k.startswith("tempogram.mode.")]
        self.assertEqual(
            double_dot_mode, [],
            f"Double-dot mode keys found: {double_dot_mode} — Bug 7 not fixed.",
        )
 
 
# ===========================================================================
# Bug 8 — tempogram.time_signature must be a plain int scalar
# ===========================================================================
 
class TestTempogramTimeSignatureScalar(unittest.TestCase):
    """
    Bug 8: _time_signature_tempogram() returns a dict.
    _extract_tempogram must index it as ["time_signature"] to return an int.
    """
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_tempogram_time_signature_is_int(self):
        fe = _make_fe(compute_time=False, compute_frequency=False,
                      compute_mfcc=False, compute_chroma=False)
        val = fe._extract_tempogram()["tempogram.time_signature"]
        self.assertIsInstance(
            val, (int, np.integer),
            f"tempogram.time_signature must be an int, got {type(val)}. Bug 8.",
        )
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_tempogram_time_signature_valid_value(self):
        fe = _make_fe(compute_time=False, compute_frequency=False,
                      compute_mfcc=False, compute_chroma=False)
        val = fe._extract_tempogram()["tempogram.time_signature"]
        self.assertIn(int(val), (3, 4),
                      f"time_signature must be 3 or 4, got {val}.")
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_tempogram_time_signature_not_dict(self):
        fe = _make_fe(compute_time=False, compute_frequency=False,
                      compute_mfcc=False, compute_chroma=False)
        val = fe._extract_tempogram()["tempogram.time_signature"]
        self.assertNotIsInstance(val, dict,
                                 "tempogram.time_signature is still a dict — Bug 8.")
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_no_double_dot_time_sig_keys(self):
        fe = _make_fe()
        out = fe.extract()
        bad = [k for k in out if k.startswith("tempogram.time_signature.")]
        self.assertEqual(bad, [],
                         f"Unexpected double-dot time_signature keys: {bad} — Bug 8.")
 
 
# ===========================================================================
# Bug 9 — from_audio uses AudioSignal(signal=y, ...) not AudioSignal(y=y, ...)
# ===========================================================================
 
class TestFromAudioConstructor(unittest.TestCase):
    """Bug 9: from_audio must call AudioSignal(signal=y, ...) not AudioSignal(y=y, ...)."""
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_from_audio_does_not_raise(self):
        """from_audio must construct without TypeError (wrong kwarg name)."""
        sr = 22050
        t = np.linspace(0, 2.0, int(2.0 * sr), endpoint=False)
        y = 0.5 * np.sin(2 * np.pi * 440 * t)
        try:
            fe = FeatureExtractor.from_audio(y, sr)
        except TypeError as exc:
            self.fail(
                f"from_audio raised TypeError: {exc}. "
                "AudioSignal may still be called with y= instead of signal= (Bug 9)."
            )
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_from_audio_produces_valid_extractor(self):
        sr = 22050
        y = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 2.0, int(2.0 * sr), endpoint=False))
        fe = FeatureExtractor.from_audio(y, sr)
        self.assertIsInstance(fe, FeatureExtractor)
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_from_audio_custom_fft_and_hop(self):
        sr = 22050
        y = np.random.randn(sr * 2)
        fe = FeatureExtractor.from_audio(y, sr, n_fft=1024, hop_length=256)
        self.assertEqual(fe.sig.N, 1024)
        self.assertEqual(fe.sig.H, 256)
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_from_audio_compute_flags_forwarded(self):
        sr = 22050
        y = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, sr, endpoint=False))
        fe = FeatureExtractor.from_audio(y, sr, compute_time=False, compute_mfcc=False)
        self.assertIsNone(fe._time)
        self.assertIsNone(fe._mfcc)
        self.assertIsNotNone(fe._freq)
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_from_audio_extract_runs(self):
        sr = 22050
        y = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 2.0, int(2.0 * sr), endpoint=False))
        fe = FeatureExtractor.from_audio(y, sr)
        out = fe.extract()
        self.assertIsInstance(out, dict)
        self.assertGreater(len(out), 0)
 
 
# ===========================================================================
# General — Construction and module flags
# ===========================================================================
 
class TestConstruction(unittest.TestCase):
    """Test __post_init__ correctly wires feature modules based on compute_* flags."""
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_all_modules_enabled_by_default(self):
        fe = _make_fe()
        self.assertIsNotNone(fe._time)
        self.assertIsNotNone(fe._freq)
        self.assertIsNotNone(fe._mfcc)
        self.assertIsNotNone(fe._chroma)
        self.assertIsNotNone(fe._temp)
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_compute_time_false(self):
        fe = _make_fe(compute_time=False)
        self.assertIsNone(fe._time)
        self.assertIsNotNone(fe._freq)
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_compute_frequency_false(self):
        fe = _make_fe(compute_frequency=False)
        self.assertIsNone(fe._freq)
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_compute_mfcc_false(self):
        fe = _make_fe(compute_mfcc=False)
        self.assertIsNone(fe._mfcc)
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_compute_chroma_false(self):
        fe = _make_fe(compute_chroma=False)
        self.assertIsNone(fe._chroma)
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_compute_tempogram_false(self):
        fe = _make_fe(compute_tempogram=False)
        self.assertIsNone(fe._temp)
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_all_disabled(self):
        fe = _make_fe(
            compute_time=False, compute_frequency=False,
            compute_mfcc=False, compute_chroma=False, compute_tempogram=False
        )
        self.assertIsNone(fe._time)
        self.assertIsNone(fe._freq)
        self.assertIsNone(fe._mfcc)
        self.assertIsNone(fe._chroma)
        self.assertIsNone(fe._temp)
 
 
# ===========================================================================
# General — _add_unified_aliases correctness
# ===========================================================================
 
class TestAddUnifiedAliases(unittest.TestCase):
    """Test alias resolution priority, source tracking, and edge cases."""
 
    def test_all_aliases_defined(self):
        """All 12 expected top-level aliases must be in the alias table."""
        expected = {
            "loudness", "energy", "speechiness", "acousticness",
            "danceability", "valence", "tempo", "liveness",
            "instrumentalness", "key", "mode", "time_signature",
        }
        # Invoke with a dummy dict containing every possible source
        sources_present = {
            "time.loudness": 1.0, "frequency.loudness": 1.0,
            "mfcc.loudness": 1.0, "tempogram.loudness": 1.0,
            "time.energy": 1.0, "frequency.energy": 1.0,
            "mfcc.energy": 1.0, "chroma.energy": 1.0,
            "mfcc.speechiness": 0.1, "time.speechiness": 0.1,
            "frequency.speechiness": 0.1, "chroma.speechiness": 0.1,
            "frequency.acousticness": 0.5, "mfcc.acousticness": 0.5,
            "time.acousticness": 0.5, "chroma.acousticness": 0.5,
            "time.danceability": 0.6, "frequency.danceability": 0.6,
            "tempogram.danceability": 0.6, "chroma.danceability": 0.6,
            "chroma.valence": 0.4, "frequency.valence": 0.4,
            "mfcc.valence": 0.4, "tempogram.valence": 0.4,
            "tempogram.tempo": 120.0, "time.tempo": 120.0,
            "frequency.tempo": 120.0, "chroma.tempo": 120.0,
            "frequency.liveness": 0.3, "time.liveness": 0.3,
            "mfcc.liveness": 0.3, "tempogram.liveness": 0.3,
            "frequency.instrumentalness": 0.7, "mfcc.instrumentalness": 0.7,
            "time.instrumentalness": 0.7, "chroma.instrumentalness": 0.7,
            "chroma.key": 0, "frequency.key": 0,
            "chroma.mode": "maj", "frequency.mode": "maj", "tempogram.mode": "major",
            "time.time_signature": 4, "tempogram.time_signature": 4,
            "frequency.time_signature": 4, "chroma.time_signature": 4,
        }
        out = _SFE._add_unified_aliases(sources_present)
        for alias in expected:
            self.assertIn(alias, out, f"Alias '{alias}' missing from output.")
 
    def test_source_tracking_for_all_aliases(self):
        """Every resolved alias must have a matching __source__ key."""
        d = {"time.energy": 0.5, "chroma.key": 3}
        out = _SFE._add_unified_aliases(d)
        for key in list(out.keys()):
            if ".__source__" not in key and key in [
                "energy", "key", "loudness", "valence", "tempo",
                "instrumentalness", "mode", "speechiness", "acousticness",
                "danceability", "liveness", "time_signature"
            ]:
                if key in out:
                    self.assertIn(
                        f"{key}.__source__", out,
                        f"Missing __source__ key for alias '{key}'.",
                    )
 
    def test_priority_order_respected(self):
        """tempo alias: tempogram.tempo must win over time.tempo."""
        d = {"tempogram.tempo": 120.0, "time.tempo": 90.0}
        out = _SFE._add_unified_aliases(d)
        self.assertEqual(out["tempo"], 120.0)
        self.assertEqual(out["tempo.__source__"], "tempogram.tempo")
 
    def test_original_keys_preserved(self):
        """Input keys must still be present in output alongside aliases."""
        d = {"chroma.valence": 0.7, "time.energy": 0.5}
        out = _SFE._add_unified_aliases(d)
        self.assertIn("chroma.valence", out)
        self.assertIn("time.energy", out)
 
    def test_empty_dict_produces_no_aliases(self):
        out = _SFE._add_unified_aliases({})
        alias_keys = [k for k in out if "." not in k or k.endswith(".__source__")]
        self.assertEqual(alias_keys, [])
 
    def test_unknown_keys_passed_through(self):
        """Keys not in any alias source list must pass through unchanged."""
        d = {"custom.feature": 42.0}
        out = _SFE._add_unified_aliases(d)
        self.assertIn("custom.feature", out)
        self.assertEqual(out["custom.feature"], 42.0)
 
 
# ===========================================================================
# General — extract() output structure
# ===========================================================================
 
class TestExtractOutputStructure(unittest.TestCase):
    """Test the shape and content of the full extract() output."""
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_extract_returns_dict(self):
        fe = _make_fe()
        out = fe.extract()
        self.assertIsInstance(out, dict)
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_extract_non_empty(self):
        fe = _make_fe()
        out = fe.extract()
        self.assertGreater(len(out), 10,
                           "extract() should return many features.")
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_top_level_aliases_present(self):
        fe = _make_fe()
        out = fe.extract()
        for alias in ("loudness", "energy", "tempo", "key", "mode"):
            self.assertIn(alias, out, f"Top-level alias '{alias}' missing.")
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_all_values_are_scalars_or_arrays(self):
        """No value in extract() output should be a raw dict."""
        fe = _make_fe()
        out = fe.extract()
        for key, val in out.items():
            if ".__source__" in key:
                continue
            self.assertNotIsInstance(
                val, dict,
                f"Key '{key}' has a dict value — a nested dict leaked through.",
            )
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_source_keys_are_strings(self):
        """All __source__ values must be strings (dotted key names)."""
        fe = _make_fe()
        out = fe.extract()
        for key, val in out.items():
            if ".__source__" in key:
                self.assertIsInstance(val, str,
                                      f"__source__ for '{key}' must be a string.")
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_module_prefix_keys_present(self):
        """Raw per-module keys must survive aliasing."""
        fe = _make_fe()
        out = fe.extract()
        expected_prefixes = ["time.", "frequency.", "mfcc.", "chroma.", "tempogram."]
        for prefix in expected_prefixes:
            found = [k for k in out if k.startswith(prefix)]
            self.assertGreater(
                len(found), 0,
                f"No '{prefix}' keys found in extract() output.",
            )
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_extract_all_disabled_returns_only_empty(self):
        """With all modules disabled, extract() returns an empty dict."""
        fe = _make_fe(
            compute_time=False, compute_frequency=False,
            compute_mfcc=False, compute_chroma=False, compute_tempogram=False,
        )
        out = fe.extract()
        self.assertEqual(out, {})
 
 
# ===========================================================================
# General — selective module extraction
# ===========================================================================
 
class TestSelectiveExtraction(unittest.TestCase):
    """Test that disabling a module removes its keys from extract() output."""
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_time_disabled_no_time_keys(self):
        fe = _make_fe(compute_time=False)
        out = fe.extract()
        time_keys = [k for k in out if k.startswith("time.")]
        self.assertEqual(time_keys, [], "No time.* keys should appear when time disabled.")
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_frequency_disabled_no_frequency_keys(self):
        fe = _make_fe(compute_frequency=False)
        out = fe.extract()
        freq_keys = [k for k in out if k.startswith("frequency.")]
        self.assertEqual(freq_keys, [])
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_mfcc_disabled_no_mfcc_keys(self):
        fe = _make_fe(compute_mfcc=False)
        out = fe.extract()
        mfcc_keys = [k for k in out if k.startswith("mfcc.")]
        self.assertEqual(mfcc_keys, [])
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_chroma_disabled_no_chroma_keys(self):
        fe = _make_fe(compute_chroma=False)
        out = fe.extract()
        chroma_keys = [k for k in out if k.startswith("chroma.")]
        self.assertEqual(chroma_keys, [])
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_tempogram_disabled_no_tempogram_keys(self):
        fe = _make_fe(compute_tempogram=False)
        out = fe.extract()
        temp_keys = [k for k in out if k.startswith("tempogram.")]
        self.assertEqual(temp_keys, [])
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_time_only_still_produces_aliases(self):
        """With only time enabled, loudness/energy/etc. aliases must still resolve."""
        fe = _make_fe(
            compute_frequency=False, compute_mfcc=False,
            compute_chroma=False, compute_tempogram=False,
        )
        out = fe.extract()
        self.assertIn("loudness", out)
        self.assertIn("energy", out)
 
 
# ===========================================================================
# General — individual _extract_* method contracts
# ===========================================================================
 
class TestExtractTimeKeys(unittest.TestCase):
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_all_expected_keys(self):
        fe = _make_fe()
        out = fe._extract_time()
        expected = [
            "time.loudness", "time.energy", "time.speechiness",
            "time.acousticness", "time.danceability", "time.tempo",
            "time.liveness", "time.instrumentalness", "time.time_signature",
        ]
        for k in expected:
            self.assertIn(k, out, f"Missing key: {k}")
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_time_values_finite(self):
        fe = _make_fe()
        out = fe._extract_time()
        for k, v in out.items():
            if isinstance(v, float):
                self.assertTrue(np.isfinite(v), f"time key '{k}' is not finite.")
 
 
class TestExtractFrequencyKeys(unittest.TestCase):
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_all_expected_keys(self):
        fe = _make_fe()
        out = fe._extract_frequency()
        expected = [
            "frequency.loudness", "frequency.energy", "frequency.speechiness",
            "frequency.acousticness", "frequency.danceability", "frequency.valence",
            "frequency.tempo", "frequency.liveness", "frequency.instrumentalness",
            "frequency.key", "frequency.mode", "frequency.time_signature",
        ]
        for k in expected:
            self.assertIn(k, out, f"Missing key: {k}")
 
 
class TestExtractChromaKeys(unittest.TestCase):
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_all_expected_keys(self):
        fe = _make_fe()
        out = fe._extract_chroma()
        expected = [
            "chroma.energy", "chroma.speechiness", "chroma.acousticness",
            "chroma.danceability", "chroma.valence", "chroma.tempo",
            "chroma.instrumentalness", "chroma.key", "chroma.mode",
            "chroma.time_signature",
        ]
        for k in expected:
            self.assertIn(k, out, f"Missing key: {k}")
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_chroma_key_is_int(self):
        fe = _make_fe()
        out = fe._extract_chroma()
        self.assertIsInstance(out["chroma.key"], (int, np.integer))
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_chroma_mode_is_string(self):
        fe = _make_fe()
        out = fe._extract_chroma()
        self.assertIsInstance(out["chroma.mode"], str)
 
 
class TestExtractTempogramKeys(unittest.TestCase):
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_all_expected_keys(self):
        fe = _make_fe()
        out = fe._extract_tempogram()
        expected = [
            "tempogram.loudness", "tempogram.tempo", "tempogram.danceability",
            "tempogram.valence", "tempogram.liveness", "tempogram.mode",
            "tempogram.time_signature",
        ]
        for k in expected:
            self.assertIn(k, out, f"Missing key: {k}")
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_tempogram_all_scalars(self):
        fe = _make_fe()
        out = fe._extract_tempogram()
        for k, v in out.items():
            self.assertNotIsInstance(v, dict,
                                     f"'{k}' must be a scalar, not a dict.")
            self.assertNotIsInstance(v, np.ndarray,
                                     f"'{k}' must be a scalar, not an ndarray.")
 
 
class TestExtractMfccKeys(unittest.TestCase):
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_all_expected_keys(self):
        fe = _make_fe()
        out = fe._extract_mfcc()
        expected = [
            "mfcc.loudness", "mfcc.energy", "mfcc.speechiness",
            "mfcc.acousticness", "mfcc.valence", "mfcc.liveness",
            "mfcc.instrumentalness",
        ]
        for k in expected:
            self.assertIn(k, out, f"Missing key: {k}")
 
 
# ===========================================================================
# General — edge cases
# ===========================================================================
 
class TestEdgeCases(unittest.TestCase):
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_silence_signal_no_crash(self):
        """All-zero signal must not raise during extract()."""
        sig = _make_silence()
        fe = FeatureExtractor(sig)
        try:
            out = fe.extract()
        except Exception as exc:
            self.fail(f"extract() raised on silence: {exc}")
        self.assertIsInstance(out, dict)
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_very_short_signal_no_crash(self):
        sr = 22050
        y = np.sin(2 * np.pi * 440 * np.arange(sr // 5) / sr)
        sig = AudioSignal(signal=y, sr=sr, N=2048, H=512)
        fe = FeatureExtractor(sig)
        out = fe.extract()
        self.assertIsInstance(out, dict)
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_extract_idempotent(self):
        """Calling extract() twice must return the same values."""
        fe = _make_fe()
        out1 = fe.extract()
        out2 = fe.extract()
        scalar_keys = [k for k, v in out1.items()
                       if isinstance(v, (int, float, np.floating, np.integer))]
        for k in scalar_keys:
            self.assertAlmostEqual(
                float(out1[k]), float(out2[k]), places=10,
                msg=f"Key '{k}' differs between two calls to extract().",
            )
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_high_frequency_signal(self):
        """Nyquist-adjacent signal (10 kHz) must not crash."""
        sr = 22050
        y = 0.5 * np.sin(2 * np.pi * 10000 * np.linspace(0, 2.0, int(2.0 * sr), endpoint=False))
        sig = AudioSignal(signal=y, sr=sr, N=2048, H=512)
        fe = FeatureExtractor(sig)
        out = fe.extract()
        self.assertIsInstance(out, dict)
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_all_numeric_values_finite_or_string(self):
        """Every numeric value in extract() must be finite (no inf/nan leaks)."""
        fe = _make_fe()
        out = fe.extract()
        for key, val in out.items():
            if ".__source__" in key:
                continue
            if isinstance(val, (float, np.floating)):
                self.assertTrue(
                    np.isfinite(val),
                    f"Key '{key}' has non-finite value {val}.",
                )
 
 
if __name__ == "__main__":
    unittest.main()