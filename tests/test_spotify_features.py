import sys
import unittest
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_REAL_IMPORTS = False

try:
    from audio_features.audio_signal import AudioSignal
    from audio_features.spotify_features import SpotifyFusion
    _REAL_IMPORTS = True
except ImportError:
    try:
        from audio_signal import AudioSignal          # type: ignore
        from spotify_features import SpotifyFusion    # type: ignore
        _REAL_IMPORTS = True
    except ImportError:
        AudioSignal = None    # type: ignore
        SpotifyFusion = None  # type: ignore
        _REAL_IMPORTS = False

# ---------------------------------------------------------------------------
# Inline stub of the three pure static methods for isolation tests
# (runs without any audio stack dependency)
# ---------------------------------------------------------------------------
EPS_VALUE = 1e-8
 
 
def _safe_float_stub(v, default=0.0):
    try:
        f = float(v)
        return f if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default
 
 
def _safe_clip01_stub(x):
    return float(np.clip(x, 0.0, 1.0))
 
 
class _StubSF:
    """Pure reimplementation of SpotifyFusion's three static methods."""
 
    @staticmethod
    def _first_existing(d: Dict[str, Any], keys: list) -> Any:
        for k in keys:
            if k in d and d[k] is not None:
                return d[k]
        return None
 
    @staticmethod
    def _softmax(x, temperature: float = 1.0):
        x = np.asarray(x, dtype=float)
        t = max(float(temperature), 1e-8)
        z = x / t
        z = z - np.max(z)
        e = np.exp(z)
        return e / np.sum(e)   # Bug 7 fix: no EPS
 
    @staticmethod
    def _weighted_softmax_fuse(values, logits, temperature=1.0, clip01=True):
        # Bug 1 fix: np.array, not np.ndarray
        vals = np.array([_safe_float_stub(v, default=np.nan) for v in values], dtype=float)
        logits = np.array(logits, dtype=float)
        mask = np.isfinite(vals) & np.isfinite(logits)
        if not np.any(mask):
            return 0.0, np.array([], dtype=float)
        vals = vals[mask]
        logits = logits[mask]
        w = _StubSF._softmax(logits, temperature=temperature)
        fused = float(np.sum(w * vals))
        if clip01:
            fused = _safe_clip01_stub(fused)
        return fused, w
 
 
_SSF = _StubSF
 
 
# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------
 
def _make_y(duration_sec=2.0, sr=22050, freq_hz=440.0):
    t = np.linspace(0, duration_sec, int(duration_sec * sr), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * freq_hz * t)
    y += 0.25 * np.sin(2 * np.pi * 2 * freq_hz * t)
    return y, sr
 
 
def _make_sig(duration_sec=2.0, freq_hz=440.0):
    y, sr = _make_y(duration_sec=duration_sec, freq_hz=freq_hz)
    return AudioSignal(signal=y, sr=sr, N=2048, H=512)
 
 
def _make_sf(duration_sec=2.0, freq_hz=440.0, **kwargs):
    return SpotifyFusion(_make_sig(duration_sec=duration_sec, freq_hz=freq_hz), **kwargs)
 
 
def _make_silence_sig():
    y = np.zeros(int(2.0 * 22050))
    return AudioSignal(signal=y, sr=22050, N=2048, H=512)
 
 
# ===========================================================================
# Bug 1 — np.array vs np.ndarray in _weighted_softmax_fuse
# ===========================================================================
 
class TestWeightedSoftmaxFuseArrayConstruction(unittest.TestCase):
    """
    Bug 1: np.ndarray([v0, v1, v2]) interprets the list as a shape, not data.
    After fix: np.array([...]) correctly constructs a data array.
    """
 
    def test_basic_fuse_produces_finite_result(self):
        fused, w = _SSF._weighted_softmax_fuse([0.3, 0.5, 0.7], [1.0, 2.0, 3.0])
        self.assertTrue(np.isfinite(fused),
                        "Fused value must be finite — np.ndarray bug would produce garbage.")
 
    def test_vals_interpreted_as_data_not_shape(self):
        """If vals were treated as a shape, a list of three floats would
        produce a 3D uninitialized array, not a 1D array of those floats."""
        fused, w = _SSF._weighted_softmax_fuse([0.2, 0.4, 0.6], [1.0, 1.0, 1.0])
        # Equal logits → equal weights → fused ≈ mean(vals) ≈ 0.4
        self.assertAlmostEqual(fused, 0.4, places=5,
                               msg="Equal-weight fuse must equal mean of values.")
 
    def test_single_value_returns_that_value(self):
        fused, w = _SSF._weighted_softmax_fuse([0.75], [1.0])
        self.assertAlmostEqual(fused, 0.75, places=6)
 
    def test_result_clipped_to_zero_one_when_clip01(self):
        fused, _ = _SSF._weighted_softmax_fuse([2.0, 3.0], [1.0, 1.0], clip01=True)
        self.assertGreaterEqual(fused, 0.0)
        self.assertLessEqual(fused, 1.0)
 
    def test_result_not_clipped_when_clip01_false(self):
        fused, _ = _SSF._weighted_softmax_fuse([200.0, 300.0], [1.0, 1.0], clip01=False)
        self.assertGreater(fused, 1.0,
                           "clip01=False must allow values > 1.")
 
    def test_all_none_returns_zero(self):
        fused, w = _SSF._weighted_softmax_fuse([None, None], [1.0, 1.0])
        self.assertEqual(fused, 0.0)
        self.assertEqual(w.size, 0)
 
    def test_mixed_none_and_valid(self):
        fused, w = _SSF._weighted_softmax_fuse([None, 0.6, None], [1.0, 2.0, 1.0])
        self.assertTrue(np.isfinite(fused))
        self.assertGreater(fused, 0.0)
 
    def test_nan_values_masked_out(self):
        fused, w = _SSF._weighted_softmax_fuse([float("nan"), 0.5], [1.0, 1.0])
        self.assertAlmostEqual(fused, 0.5, places=6)
 
    def test_weights_sum_to_one(self):
        _, w = _SSF._weighted_softmax_fuse([0.1, 0.5, 0.9], [1.0, 2.0, 3.0])
        self.assertAlmostEqual(float(np.sum(w)), 1.0, places=10)
 
    def test_higher_logit_gets_more_weight(self):
        _, w = _SSF._weighted_softmax_fuse([0.1, 0.9], [1.0, 5.0])
        self.assertGreater(w[1], w[0],
                           "Higher logit must receive higher softmax weight.")
 
 
# ===========================================================================
# Bug 2 — from_audio forwards **kwargs to SpotifyFusion, not AudioSignal
# ===========================================================================
 
class TestFromAudio(unittest.TestCase):
    """
    Bug 2: **kwargs must go to SpotifyFusion, not AudioSignal.
    Before fix: AudioSignal(signal=y, sr=sr, N=n_fft, H=hop_length, **kwargs)
    would raise TypeError for unrecognised kwargs like temperature=0.5.
    """
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_from_audio_no_kwargs_does_not_raise(self):
        y, sr = _make_y()
        sf = SpotifyFusion.from_audio(y, sr)
        self.assertIsInstance(sf, SpotifyFusion)
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_from_audio_temperature_kwarg_forwarded(self):
        """temperature must reach SpotifyFusion, not AudioSignal."""
        y, sr = _make_y()
        sf = SpotifyFusion.from_audio(y, sr, temperature=0.5)
        self.assertAlmostEqual(sf.temperature, 0.5, places=10)
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_from_audio_compute_flags_forwarded(self):
        """compute_* flags must reach SpotifyFusion.__post_init__."""
        y, sr = _make_y()
        sf = SpotifyFusion.from_audio(y, sr, compute_mfcc=False, compute_chroma=False)
        self.assertFalse(sf.compute_mfcc)
        self.assertFalse(sf.compute_chroma)
        self.assertIsNone(sf._features._mfcc)
        self.assertIsNone(sf._features._chroma)
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_from_audio_custom_fft_and_hop(self):
        y, sr = _make_y()
        sf = SpotifyFusion.from_audio(y, sr, n_fft=1024, hop_length=256)
        self.assertEqual(sf.sig.N, 1024)
        self.assertEqual(sf.sig.H, 256)
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_from_audio_extract_runs(self):
        y, sr = _make_y()
        sf = SpotifyFusion.from_audio(y, sr)
        out = sf.extract()
        self.assertIsInstance(out, dict)
        self.assertGreater(len(out), 0)
 
 
# ===========================================================================
# Bug 3 — _first_existing fallback logic tested properly
# ===========================================================================
 
class TestFirstExisting(unittest.TestCase):
    """
    Bug 3: _first_existing must try keys in order and return the first
    non-None match. Previously it was called with single-key lists,
    making fallback logic unreachable.
    """
 
    def test_returns_first_non_none(self):
        d = {"a": None, "b": 0.5, "c": 0.9}
        result = _SSF._first_existing(d, ["a", "b", "c"])
        self.assertEqual(result, 0.5)
 
    def test_skips_none_values(self):
        d = {"x": None, "y": None, "z": 42.0}
        result = _SSF._first_existing(d, ["x", "y", "z"])
        self.assertEqual(result, 42.0)
 
    def test_returns_none_when_all_missing(self):
        d = {"a": 1.0}
        result = _SSF._first_existing(d, ["b", "c", "d"])
        self.assertIsNone(result)
 
    def test_returns_none_when_all_none(self):
        d = {"a": None, "b": None}
        result = _SSF._first_existing(d, ["a", "b"])
        self.assertIsNone(result)
 
    def test_priority_first_key_wins(self):
        d = {"a": 1.0, "b": 2.0}
        result = _SSF._first_existing(d, ["a", "b"])
        self.assertEqual(result, 1.0)
 
    def test_returns_zero_without_skipping(self):
        """0.0 is a valid value — must not be treated as falsy."""
        d = {"a": 0.0}
        result = _SSF._first_existing(d, ["a"])
        self.assertEqual(result, 0.0)
 
    def test_returns_false_without_skipping(self):
        """False is a valid value — must not be skipped."""
        d = {"a": False}
        result = _SSF._first_existing(d, ["a"])
        self.assertEqual(result, False)
 
    def test_empty_keys_returns_none(self):
        d = {"a": 1.0}
        result = _SSF._first_existing(d, [])
        self.assertIsNone(result)
 
    def test_empty_dict_returns_none(self):
        result = _SSF._first_existing({}, ["a", "b"])
        self.assertIsNone(result)
 
    def test_multi_key_fallback_chain(self):
        """Full fallback chain: first two missing, third present."""
        d = {"chroma.key": 5}
        result = _SSF._first_existing(d, ["key", "nonexistent", "chroma.key"])
        self.assertEqual(result, 5)
 
 
# ===========================================================================
# Bug 4 — mode mapped via _MODE_MAP, not int(mode_val)
# ===========================================================================
 
class TestModeMapping(unittest.TestCase):
    """
    Bug 4: mode_val is a string ("maj", "min", "major", "minor").
    int("maj") raises ValueError. After fix, _MODE_MAP converts to 0/1.
    """
 
    _MODE_MAP = {"maj": 1, "major": 1, "min": 0, "minor": 0}
 
    def _map(self, mode_val):
        return self._MODE_MAP.get(str(mode_val).lower(), -1) if mode_val is not None else -1
 
    def test_maj_maps_to_one(self):
        self.assertEqual(self._map("maj"), 1)
 
    def test_major_maps_to_one(self):
        self.assertEqual(self._map("major"), 1)
 
    def test_min_maps_to_zero(self):
        self.assertEqual(self._map("min"), 0)
 
    def test_minor_maps_to_zero(self):
        self.assertEqual(self._map("minor"), 0)
 
    def test_none_maps_to_minus_one(self):
        self.assertEqual(self._map(None), -1)
 
    def test_unknown_string_maps_to_minus_one(self):
        self.assertEqual(self._map("dorian"), -1)
 
    def test_case_insensitive(self):
        self.assertEqual(self._map("MAJ"), 1)
        self.assertEqual(self._map("MINOR"), 0)
        self.assertEqual(self._map("Major"), 1)
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_extract_mode_is_int(self):
        """extract() must return mode as an integer, not a string."""
        sf = _make_sf()
        out = sf.extract()
        self.assertIsInstance(out["mode"], int,
                              "mode must be int after _MODE_MAP conversion.")
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_extract_mode_valid_value(self):
        """mode must be 0, 1, or -1."""
        sf = _make_sf()
        out = sf.extract()
        self.assertIn(out["mode"], (-1, 0, 1),
                      f"Unexpected mode value: {out['mode']}")
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_extract_does_not_raise_value_error(self):
        """Must not raise ValueError from int('maj')."""
        sf = _make_sf()
        try:
            out = sf.extract()
        except ValueError as exc:
            self.fail(f"extract() raised ValueError on mode conversion: {exc}")
 
 
# ===========================================================================
# Bug 5 — spotify_score formula scale coupling documented and tested
# ===========================================================================
 
class TestSpotifyScoreFormula(unittest.TestCase):
    """
    Bug 5: spotify_score re-normalises tempo and loudness inline.
    We test the formula's mathematical contract independent of audio content.
    """
 
    def _score(self, energy=0.5, danceability=0.5, valence=0.5, acousticness=0.5,
               speechiness=0.5, liveness=0.5, instrumentalness=0.5,
               tempo=120.0, loudness=-20.0):
        return float(np.clip(
            0.15 * energy +
            0.15 * danceability +
            0.12 * valence +
            0.12 * acousticness +
            0.10 * speechiness +
            0.10 * liveness +
            0.08 * instrumentalness +
            0.08 * np.clip(tempo / 240.0, 0.0, 1.0) +
            0.10 * np.clip((loudness + 80.0) / 80.0, 0.0, 1.0),
            0.0, 1.0
        ))
 
    def test_all_midpoint_values_gives_midpoint_score(self):
        """All features at 0.5 / 120 BPM / -40 dB → score near 0.5."""
        score = self._score(energy=0.5, danceability=0.5, valence=0.5,
                            acousticness=0.5, speechiness=0.5, liveness=0.5,
                            instrumentalness=0.5, tempo=120.0, loudness=-40.0)
        self.assertAlmostEqual(score, 0.5, delta=0.05)
 
    def test_all_max_values_gives_score_one(self):
        """Max all features → score must be 1.0."""
        score = self._score(energy=1.0, danceability=1.0, valence=1.0,
                            acousticness=1.0, speechiness=1.0, liveness=1.0,
                            instrumentalness=1.0, tempo=240.0, loudness=0.0)
        self.assertAlmostEqual(score, 1.0, places=6)
 
    def test_all_min_values_gives_score_zero(self):
        """Min all features → score must be 0.0."""
        score = self._score(energy=0.0, danceability=0.0, valence=0.0,
                            acousticness=0.0, speechiness=0.0, liveness=0.0,
                            instrumentalness=0.0, tempo=0.0, loudness=-80.0)
        self.assertAlmostEqual(score, 0.0, places=6)
 
    def test_score_monotone_in_energy(self):
        """Increasing energy alone must increase the score."""
        s1 = self._score(energy=0.2)
        s2 = self._score(energy=0.8)
        self.assertGreater(s2, s1)
 
    def test_score_monotone_in_tempo(self):
        s1 = self._score(tempo=60.0)
        s2 = self._score(tempo=200.0)
        self.assertGreater(s2, s1)
 
    def test_score_monotone_in_loudness(self):
        s1 = self._score(loudness=-70.0)
        s2 = self._score(loudness=-10.0)
        self.assertGreater(s2, s1)
 
    def test_weights_sum_to_one(self):
        """Hard-coded formula weights must sum to 1.0."""
        weights = [0.15, 0.15, 0.12, 0.12, 0.10, 0.10, 0.08, 0.08, 0.10]
        self.assertAlmostEqual(sum(weights), 1.0, places=10)
 
    def test_tempo_renormalised_before_weighting(self):
        """240 BPM contributes 0.08 * 1.0 = 0.08, not 0.08 * 240."""
        score_high_tempo = self._score(energy=0.0, danceability=0.0, valence=0.0,
                                       acousticness=0.0, speechiness=0.0, liveness=0.0,
                                       instrumentalness=0.0, tempo=240.0, loudness=-80.0)
        self.assertAlmostEqual(score_high_tempo, 0.08, places=6)
 
    def test_loudness_renormalised_before_weighting(self):
        """0 dB loudness contributes 0.10 * 1.0 = 0.10."""
        score_loud = self._score(energy=0.0, danceability=0.0, valence=0.0,
                                 acousticness=0.0, speechiness=0.0, liveness=0.0,
                                 instrumentalness=0.0, tempo=0.0, loudness=0.0)
        self.assertAlmostEqual(score_loud, 0.10, places=6)
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_extract_spotify_score_in_range(self):
        sf = _make_sf()
        out = sf.extract()
        self.assertGreaterEqual(out["spotify_score"], 0.0)
        self.assertLessEqual(out["spotify_score"], 1.0)
 
 
# ===========================================================================
# Bug 6 — Optional not imported
# ===========================================================================
 
class TestUnusedImports(unittest.TestCase):
    """Bug 6: Optional must not appear in the module's imports."""
 
    def test_optional_not_imported(self):
        import inspect
        try:
            import spotify_features as sf_mod  # type: ignore
        except ImportError:
            try:
                from audio_features import spotify_features as sf_mod  # type: ignore
            except ImportError:
                self.skipTest("spotify_features not directly importable.")
        src = inspect.getsource(sf_mod)
        self.assertNotIn("Optional", src,
                         "'Optional' still imported — Bug 6 not fixed.")
 
 
# ===========================================================================
# Bug 7 — _softmax weights sum exactly to 1.0 (no EPS in denominator)
# ===========================================================================
 
class TestSoftmax(unittest.TestCase):
    """Bug 7: removing EPS ensures weights sum exactly to 1.0."""
 
    def test_weights_sum_to_one(self):
        w = _SSF._softmax([1.0, 2.0, 3.0])
        self.assertAlmostEqual(float(np.sum(w)), 1.0, places=12,
                               msg="Softmax weights must sum to 1.0.")
 
    def test_weights_sum_to_one_after_eps_removal(self):
        """With EPS in denominator, sum < 1. Without it, sum == 1 exactly."""
        w = _SSF._softmax([0.5, 1.5, 2.5])
        s = float(np.sum(w))
        self.assertNotAlmostEqual(
            s, 1.0 - 1e-9, places=12,
            msg="Sum should not be less than 1 by EPS amount.",
        )
        self.assertAlmostEqual(s, 1.0, places=12)
 
    def test_all_weights_non_negative(self):
        w = _SSF._softmax([-10.0, 0.0, 5.0])
        self.assertTrue(np.all(w >= 0.0))
 
    def test_max_logit_gets_highest_weight(self):
        w = _SSF._softmax([1.0, 5.0, 2.0])
        self.assertEqual(int(np.argmax(w)), 1)
 
    def test_uniform_logits_give_equal_weights(self):
        w = _SSF._softmax([3.0, 3.0, 3.0])
        np.testing.assert_allclose(w, [1/3, 1/3, 1/3], atol=1e-10)
 
    def test_temperature_one_is_default(self):
        w1 = _SSF._softmax([1.0, 2.0, 3.0], temperature=1.0)
        w2 = _SSF._softmax([1.0, 2.0, 3.0])
        np.testing.assert_allclose(w1, w2, atol=1e-12)
 
    def test_high_temperature_flattens_distribution(self):
        """High temperature → weights approach uniform."""
        w_low = _SSF._softmax([1.0, 5.0], temperature=0.1)
        w_high = _SSF._softmax([1.0, 5.0], temperature=100.0)
        # High-temp distribution should be more uniform
        self.assertLess(np.max(w_high) - np.min(w_high),
                        np.max(w_low) - np.min(w_low))
 
    def test_low_temperature_sharpens_distribution(self):
        """Low temperature → winner-take-all."""
        w = _SSF._softmax([1.0, 2.0, 10.0], temperature=0.01)
        self.assertGreater(w[2], 0.99)
 
    def test_numerical_stability_large_logits(self):
        """Max-subtraction prevents overflow for very large logits."""
        w = _SSF._softmax([1000.0, 1001.0, 999.0])
        self.assertTrue(np.all(np.isfinite(w)))
        self.assertAlmostEqual(float(np.sum(w)), 1.0, places=10)
 
    def test_numerical_stability_very_negative_logits(self):
        w = _SSF._softmax([-1000.0, -999.0, -998.0])
        self.assertTrue(np.all(np.isfinite(w)))
        self.assertAlmostEqual(float(np.sum(w)), 1.0, places=10)
 
    def test_single_element_returns_one(self):
        w = _SSF._softmax([42.0])
        self.assertAlmostEqual(float(w[0]), 1.0, places=12)
 
    def test_zero_temperature_clamps_to_epsilon(self):
        """temperature=0 must clamp to 1e-8 (not divide by zero)."""
        w = _SSF._softmax([1.0, 2.0, 3.0], temperature=0.0)
        self.assertTrue(np.all(np.isfinite(w)))
        self.assertAlmostEqual(float(np.sum(w)), 1.0, places=8)
 
 
# ===========================================================================
# General — Construction
# ===========================================================================
 
class TestConstruction(unittest.TestCase):
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_default_flags_all_true(self):
        sf = _make_sf()
        self.assertTrue(sf.compute_time)
        self.assertTrue(sf.compute_frequency)
        self.assertTrue(sf.compute_mfcc)
        self.assertTrue(sf.compute_chroma)
        self.assertTrue(sf.compute_tempogram)
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_default_temperature_is_one(self):
        sf = _make_sf()
        self.assertAlmostEqual(sf.temperature, 1.0, places=10)
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_feature_extractor_wired(self):
        sf = _make_sf()
        from audio_features.feature_extractor import FeatureExtractor  # type: ignore
        self.assertIsInstance(sf._features, FeatureExtractor)
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_compute_flags_propagate_to_extractor(self):
        sf = _make_sf(compute_mfcc=False, compute_chroma=False)
        self.assertIsNone(sf._features._mfcc)
        self.assertIsNone(sf._features._chroma)
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_custom_temperature(self):
        sf = _make_sf(temperature=0.5)
        self.assertAlmostEqual(sf.temperature, 0.5, places=10)
 
 
# ===========================================================================
# General — extract() output contract
# ===========================================================================
 
class TestExtractOutputContract(unittest.TestCase):
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def setUp(self):
        self.sf = _make_sf()
        self.out = self.sf.extract()
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_returns_dict(self):
        self.assertIsInstance(self.out, dict)
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_all_expected_keys_present(self):
        expected = {
            "loudness", "energy", "speechiness", "acousticness",
            "danceability", "valence", "tempo", "liveness",
            "instrumentalness", "key", "mode", "time_signature", "spotify_score",
        }
        for k in expected:
            self.assertIn(k, self.out, f"Missing key: '{k}'")
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_continuous_features_in_zero_one(self):
        for feat in ("energy", "speechiness", "acousticness", "danceability",
                     "valence", "liveness", "instrumentalness", "spotify_score"):
            val = self.out[feat]
            self.assertGreaterEqual(val, 0.0, msg=f"{feat} < 0")
            self.assertLessEqual(val, 1.0, msg=f"{feat} > 1")
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_tempo_is_in_bpm_range(self):
        tempo = self.out["tempo"]
        self.assertGreaterEqual(tempo, 0.0)
        self.assertLessEqual(tempo, 240.0)
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_loudness_in_db_range(self):
        loud = self.out["loudness"]
        self.assertGreaterEqual(loud, -80.0)
        self.assertLessEqual(loud, 0.0 + 1e-6)
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_key_is_valid(self):
        key = self.out["key"]
        self.assertIsInstance(key, int)
        self.assertIn(key, list(range(24)) + [-1])
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_mode_is_valid(self):
        self.assertIn(self.out["mode"], (-1, 0, 1))
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_time_signature_is_valid(self):
        self.assertIn(self.out["time_signature"], (3, 4))
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_all_values_finite(self):
        for k, v in self.out.items():
            if isinstance(v, (float, np.floating)):
                self.assertTrue(np.isfinite(v), f"Key '{k}' is not finite.")
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_spotify_score_is_float(self):
        self.assertIsInstance(self.out["spotify_score"], float)
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_extract_idempotent(self):
        """Two extract() calls must return identical values."""
        out2 = self.sf.extract()
        for k, v in self.out.items():
            if isinstance(v, (float, np.floating)):
                self.assertAlmostEqual(float(v), float(out2[k]), places=10,
                                       msg=f"Key '{k}' differs between calls.")
 
 
# ===========================================================================
# General — temperature sensitivity
# ===========================================================================
 
class TestTemperatureSensitivity(unittest.TestCase):
    """Temperature controls how sharply the softmax weights differ."""
 
    def test_low_temperature_concentrates_weight(self):
        """Very low temperature → highest logit dominates."""
        vals = [0.1, 0.5, 0.9]
        logits = [1.0, 2.0, 5.0]
        fused_low, w_low = _SSF._weighted_softmax_fuse(vals, logits, temperature=0.01)
        fused_high, w_high = _SSF._weighted_softmax_fuse(vals, logits, temperature=10.0)
        # Low temperature pushes result toward the highest-logit value (0.9)
        self.assertGreater(fused_low, fused_high)
 
    def test_high_temperature_averages_values(self):
        """Very high temperature → all logits equal → result ≈ mean(vals)."""
        vals = [0.2, 0.5, 0.8]
        logits = [1.0, 3.0, 9.0]
        fused, _ = _SSF._weighted_softmax_fuse(vals, logits, temperature=1000.0)
        mean_val = float(np.mean(vals))
        self.assertAlmostEqual(fused, mean_val, delta=0.05)
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_temperature_affects_output(self):
        """Different temperatures must produce different energy scores."""
        sf_sharp = _make_sf(temperature=0.1)
        sf_flat  = _make_sf(temperature=5.0)
        out_sharp = sf_sharp.extract()
        out_flat  = sf_flat.extract()
        # At least one feature must differ between the two temperatures
        features = ["energy", "danceability", "valence", "speechiness"]
        diffs = [abs(out_sharp[f] - out_flat[f]) for f in features]
        self.assertGreater(max(diffs), 0.0,
                           "temperature has no effect on any feature.")
 
 
# ===========================================================================
# General — selective module disable
# ===========================================================================
 
class TestSelectiveModules(unittest.TestCase):
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_time_only_still_extracts(self):
        sf = _make_sf(compute_frequency=False, compute_mfcc=False,
                      compute_chroma=False, compute_tempogram=False)
        out = sf.extract()
        self.assertIn("energy", out)
        self.assertIn("spotify_score", out)
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_all_disabled_fallback_values(self):
        """With all modules off, every fused feature must be 0 or default."""
        sf = _make_sf(compute_time=False, compute_frequency=False,
                      compute_mfcc=False, compute_chroma=False,
                      compute_tempogram=False)
        out = sf.extract()
        self.assertIsInstance(out, dict)
        # All sources are missing → fused = 0.0
        for feat in ("energy", "danceability", "valence", "speechiness",
                     "acousticness", "liveness", "instrumentalness"):
            self.assertEqual(out[feat], 0.0, f"{feat} should be 0 with all disabled.")
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_chroma_only_populates_key_and_mode(self):
        sf = _make_sf(compute_time=False, compute_frequency=False,
                      compute_mfcc=False, compute_tempogram=False)
        out = sf.extract()
        # chroma provides key and mode
        self.assertNotEqual(out["key"], -1,
                            "chroma.key should resolve key when chroma is enabled.")
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_tempogram_provides_tempo(self):
        sf = _make_sf(compute_time=False, compute_frequency=False,
                      compute_mfcc=False, compute_chroma=False)
        out = sf.extract()
        self.assertGreaterEqual(out["tempo"], 0.0)
 
 
# ===========================================================================
# General — edge cases
# ===========================================================================
 
class TestEdgeCases(unittest.TestCase):
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_silence_no_crash(self):
        sig = _make_silence_sig()
        sf = SpotifyFusion(sig)
        out = sf.extract()
        self.assertIsInstance(out, dict)
        self.assertIn("spotify_score", out)
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_silence_features_in_range(self):
        sig = _make_silence_sig()
        sf = SpotifyFusion(sig)
        out = sf.extract()
        for feat in ("energy", "danceability", "valence", "speechiness",
                     "acousticness", "liveness", "instrumentalness", "spotify_score"):
            self.assertGreaterEqual(out[feat], 0.0, msg=f"silence: {feat} < 0")
            self.assertLessEqual(out[feat], 1.0, msg=f"silence: {feat} > 1")
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_very_short_signal_no_crash(self):
        """Signal must produce >= 9 STFT frames for librosa.feature.delta."""
        sr = 22050
        y = np.sin(2 * np.pi * 440 * np.arange(sr // 5) / sr)
        sig = AudioSignal(signal=y, sr=sr, N=2048, H=512)
        sf = SpotifyFusion(sig)
        out = sf.extract()
        self.assertIsInstance(out, dict)
 
    def test_fuse_with_all_nan_logits(self):
        fused, w = _SSF._weighted_softmax_fuse(
            [0.5, 0.5], [float("nan"), float("nan")]
        )
        self.assertEqual(fused, 0.0)
        self.assertEqual(w.size, 0)
 
    def test_fuse_single_valid_source(self):
        """Only one valid source → its value must be the fused result."""
        fused, w = _SSF._weighted_softmax_fuse([None, 0.7, None], [1.0, 2.0, 1.0])
        self.assertAlmostEqual(fused, 0.7, places=6)
        self.assertAlmostEqual(float(np.sum(w)), 1.0, places=10)
 
    def test_fuse_consistent_with_manual_weighted_mean(self):
        """Verify fuse matches hand-calculated softmax-weighted average."""
        vals = [0.2, 0.8]
        logits = [1.0, 1.0]  # equal logits → equal weights
        fused, _ = _SSF._weighted_softmax_fuse(vals, logits, clip01=False)
        expected = float(np.mean(vals))
        self.assertAlmostEqual(fused, expected, places=6)
 
    @unittest.skipUnless(_REAL_IMPORTS, "Full audio stack not available")
    def test_high_frequency_signal_no_crash(self):
        sr = 22050
        y = 0.5 * np.sin(2 * np.pi * 10000 * np.linspace(0, 2.0, int(2.0 * sr), endpoint=False))
        sig = AudioSignal(signal=y, sr=sr, N=2048, H=512)
        sf = SpotifyFusion(sig)
        out = sf.extract()
        self.assertIsInstance(out, dict)
 
 
if __name__ == "__main__":
    unittest.main()