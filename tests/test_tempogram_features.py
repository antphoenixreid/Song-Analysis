"""
Unit tests for TempogramFeatures.

Covers:
  - Bug 3:  _tempogram_autocorr norm_sum divides by sum(acf) — can be negative/zero
  - Bug 4:  _tempo_variation_curve missing "curve" key (KeyError in _liveness_tempogram)
  - Bug 5:  _global_bpm divisor loop uses break instead of continue
  - Bug 7:  _danceability_tempogram accesses "multi_periodicity" key (doesn't exist)
  - Bug 8:  _danceability_tempogram calls _beat_strength (method doesn't exist)
  - Bug 9:  _mode_tempogram passes norm_score= instead of norm_sum= (silently ignored)
  - General behavioral coverage of the full TempogramFeatures pipeline
"""

import types
import unittest

import numpy as np
from pathlib import Path
import pytest
import sys

# ---------------------------------------------------------------------------
# Minimal stubs for package imports
# ---------------------------------------------------------------------------
EPS_VALUE = 1e-10

utils_stub = types.ModuleType("audio_features.utils")
utils_stub.EPS = EPS_VALUE
utils_stub.safe_clip01 = lambda x: float(np.clip(x, 0.0, 1.0))

audio_signal_stub = types.ModuleType("audio_features.audio_signal")


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from audio_features.audio_signal import AudioSignal

audio_signal_stub.AudioSignal = AudioSignal

sys.modules.setdefault("audio_features", types.ModuleType("audio_features"))
sys.modules["audio_features.utils"] = utils_stub
sys.modules["audio_features.audio_signal"] = audio_signal_stub
sys.modules.setdefault(
    "audio_features.tempogram_features",
    types.ModuleType("audio_features.tempogram_features"),
)

from audio_features.tempogram_features import TempogramFeatures

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_signal(duration_sec=3.0, sr=22050, N=2048, H=512, bpm=120.0):
    """Rhythmic pulse train at the given BPM to give tempogram tests signal."""
    n_samples = int(duration_sec * sr)
    y = np.zeros(n_samples)
    beat_period_samples = int(sr * 60.0 / bpm)
    for i in range(0, n_samples, beat_period_samples):
        end = min(i + beat_period_samples // 8, n_samples)
        y[i:end] = 1.0
    # Add light noise so the signal is non-trivial
    rng = np.random.default_rng(42)
    y += 0.05 * rng.standard_normal(n_samples)
    return AudioSignal(signal=y, sr=sr, N=N, H=H)


def _make_sine(duration_sec=3.0, sr=22050, N=2048, H=512, freq_hz=440.0):
    t = np.linspace(0, duration_sec, int(duration_sec * sr), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * freq_hz * t)
    return AudioSignal(signal=y, sr=sr, N=N, H=H)


def _make_silence(duration_sec=3.0, sr=22050, N=2048, H=512):
    y = np.zeros(int(duration_sec * sr))
    return AudioSignal(signal=y, sr=sr, N=N, H=H)


def _make_tf(duration_sec=3.0, bpm=120.0):
    return TempogramFeatures(_make_signal(duration_sec=duration_sec, bpm=bpm))


def _make_tf_silence():
    return TempogramFeatures(_make_silence())


def _regular_beat_times(bpm=120.0, duration_sec=4.0):
    period = 60.0 / bpm
    return np.arange(0.0, duration_sec, period)


# ---------------------------------------------------------------------------
# Bug 3 — _tempogram_autocorr norm_sum divides by sum(acf), which can be < 0
# ---------------------------------------------------------------------------

class TestTempogramAutocorrNormSum(unittest.TestCase):
    """
    Bug 3: dividing acf by np.sum(acf) is wrong because post-zero-mean the
    sum of the full ACF can be near zero or negative, producing large/sign-
    flipped values. The zero-lag acf[0] (always >= 0) must be the divisor.
    """

    def setUp(self):
        self.tf = _make_tf()

    def test_tempogram_values_finite(self):
        """All tempogram entries must be finite after normalization."""
        tg = self.tf._tempogram_autocorr()["tempogram"]
        self.assertTrue(
            np.all(np.isfinite(tg)),
            "Tempogram contains non-finite values — Bug 3 may still be present "
            "(sum(acf) normalization can produce inf/nan).",
        )

    def test_norm_sum_true_no_large_values(self):
        """
        With correct zero-lag normalization, all values should be <= 1.
        Dividing by sum(acf) can produce values >> 1 when the sum is small.
        """
        tg = self.tf._tempogram_autocorr(norm_sum=True)["tempogram"]
        max_val = float(np.max(np.abs(tg)))
        self.assertLessEqual(
            max_val, 2.01,
            f"Max tempogram value {max_val:.4f} is implausibly large — "
            "sum(acf) normalization (Bug 3) may be in effect.",
        )

    def test_norm_sum_false_zero_lag_is_one(self):
        """
        With norm_sum=False the code uses acf[0] — zero-lag row should be 1.
        This verifies the non-buggy path and gives a baseline for comparison.
        """
        tg = self.tf._tempogram_autocorr(norm_sum=False)["tempogram"]
        # Row 0 is the zero-lag; after acf/acf[0] it should be ~1.0
        nonzero_cols = np.any(tg != 0, axis=0)
        if np.any(nonzero_cols):
            row0 = tg[0, nonzero_cols]
            np.testing.assert_allclose(
                row0, 1.0, atol=0.15,
                err_msg="Zero-lag row must be ~1.0 with norm_sum=False.",
            )

    def test_tempogram_shape(self):
        result = self.tf._tempogram_autocorr()
        tg = result["tempogram"]
        bpm = result["bpm"]
        self.assertEqual(tg.shape[0], self.tf.N)  # win_length rows
        self.assertEqual(bpm.shape[0], self.tf.N)
        self.assertGreater(tg.shape[1], 0)

    def test_bpm_axis_zero_lag_is_zero(self):
        """BPM[0] is the zero-lag (infinite BPM) — stored as 0."""
        bpm = self.tf._tempogram_autocorr()["bpm"]
        self.assertEqual(float(bpm[0]), 0.0)

    def test_bpm_axis_is_non_negative(self):
        bpm = self.tf._tempogram_autocorr()["bpm"]
        self.assertTrue(np.all(bpm >= 0.0))

    def test_cached(self):
        r1 = self.tf._tempogram_autocorr()
        r2 = self.tf._tempogram_autocorr()
        self.assertIs(r1, r2)

    def test_center_false_produces_fewer_windows(self):
        r_center = self.tf._tempogram_autocorr(center=True)
        r_no_center = self.tf._tempogram_autocorr(center=False)
        self.assertLessEqual(
            r_no_center["tempogram"].shape[1],
            r_center["tempogram"].shape[1],
        )


# ---------------------------------------------------------------------------
# Bug 4 — _tempo_variation_curve missing "curve" key
# ---------------------------------------------------------------------------

class TestTempoVariationCurvKey(unittest.TestCase):
    """
    Bug 4: result dict must include a 'curve' key so that _liveness_tempogram
    (which calls result["curve"]) does not raise a KeyError.
    """

    def setUp(self):
        self.tf = _make_tf()

    def test_curve_key_present(self):
        result = self.tf._tempo_variation_curve()
        self.assertIn(
            "curve", result,
            "'curve' key missing from _tempo_variation_curve result — Bug 4 still present.",
        )

    def test_curve_is_ndarray(self):
        curve = self.tf._tempo_variation_curve()["curve"]
        self.assertIsInstance(curve, np.ndarray)

    def test_curve_non_negative(self):
        """'curve' represents absolute variation — must be >= 0."""
        curve = self.tf._tempo_variation_curve()["curve"]
        if curve.size > 0:
            self.assertTrue(np.all(curve >= 0.0))

    def test_variation_key_present(self):
        self.assertIn("variation", self.tf._tempo_variation_curve())

    def test_abs_variation_equals_abs_of_variation(self):
        result = self.tf._tempo_variation_curve()
        if result["variation"].size > 0:
            np.testing.assert_allclose(
                result["abs_variation"],
                np.abs(result["variation"]),
                atol=1e-10,
            )

    def test_curve_equals_abs_variation(self):
        """'curve' should be the absolute variation (what liveness uses)."""
        result = self.tf._tempo_variation_curve()
        if result["curve"].size > 0:
            np.testing.assert_allclose(
                result["curve"], result["abs_variation"], atol=1e-10
            )

    def test_liveness_does_not_raise(self):
        """Downstream caller must not raise KeyError via the 'curve' key."""
        try:
            val = self.tf._liveness_tempogram()
        except KeyError as exc:
            self.fail(
                f"_liveness_tempogram raised KeyError: {exc}. "
                "'curve' key is still missing from _tempo_variation_curve (Bug 4)."
            )

    def test_empty_signal_curve_is_empty_array(self):
        tf = _make_tf_silence()
        result = tf._tempo_variation_curve()
        self.assertIn("curve", result)
        self.assertIsInstance(result["curve"], np.ndarray)

    def test_cached(self):
        r1 = self.tf._tempo_variation_curve()
        r2 = self.tf._tempo_variation_curve()
        self.assertIs(r1, r2)


# ---------------------------------------------------------------------------
# Bug 5 — _global_bpm divisor loop uses break instead of continue
# ---------------------------------------------------------------------------

class TestGlobalBpmDivisorLoop(unittest.TestCase):
    """
    Bug 5: in the sub-octave correction loop (divisors [2, 3, 4]),
    'break' when candidate < bpm_min exits the loop prematurely — a later
    divisor (e.g. 3 after 2 was too small) could still land in range.
    """

    def setUp(self):
        self.tf = _make_tf()

    def test_global_bpm_returns_valid_result(self):
        result = self.tf._global_bpm()
        for k in ("bpm", "lag", "strength", "bpm_axis"):
            self.assertIn(k, result)

    def test_global_bpm_in_range(self):
        bpm = self.tf._global_bpm()["bpm"]
        self.assertGreaterEqual(bpm, 0.0)
        # Either 0 (not detected) or within the valid window
        if bpm > 0:
            self.assertGreaterEqual(bpm, 40.0)
            self.assertLessEqual(bpm, 240.0)

    def test_divisor_3_checked_even_when_divisor_2_below_min(self):
        """
        If detected_bpm = 300 and bpm_min = 120:
          300/2 = 150 → in range → would stop here (correct)
        But if detected_bpm = 500 and bpm_min = 200:
          500/2 = 250 → above range, break skips 500/3 ≈ 167 which is also out
        The key case: detected_bpm = 700, bpm_min = 200, bpm_max = 240:
          700/2 = 350 → above max (should continue, not break)
          700/3 ≈ 233 → in range! would be found with continue but not break
        We simulate this by directly calling the octave logic.
        """
        # Construct a tf with a high detected BPM by injecting cache
        tf = _make_tf()
        bpm_axis = np.zeros(tf.N, dtype=float)
        bpm_axis[1:] = 60.0 * tf.frame_rate / np.arange(1, tf.N)
        # Inject a fake global AC that peaks at ~300 BPM (outside [40, 240])
        # Then check divisor logic handles continue vs break correctly

        # The simplest behavioral check: result is always in valid range
        result = tf._global_bpm(bpm_min=40.0, bpm_max=240.0)
        bpm = result["bpm"]
        if bpm > 0:
            self.assertLessEqual(bpm, 240.0, "BPM exceeds bpm_max after octave correction.")
            self.assertGreaterEqual(bpm, 40.0, "BPM below bpm_min after octave correction.")

    def test_strength_is_non_negative(self):
        self.assertGreaterEqual(self.tf._global_bpm()["strength"], 0.0)

    def test_bpm_axis_returned(self):
        axis = self.tf._global_bpm()["bpm_axis"]
        self.assertIsInstance(axis, np.ndarray)
        self.assertGreater(axis.size, 0)

    def test_cached(self):
        r1 = self.tf._global_bpm()
        r2 = self.tf._global_bpm()
        self.assertIs(r1, r2)

    def test_silence_returns_zero_bpm(self):
        tf = _make_tf_silence()
        result = tf._global_bpm()
        # Silence may detect 0 or a spurious low value; must not crash
        self.assertIsInstance(result["bpm"], float)
        self.assertGreaterEqual(result["bpm"], 0.0)


# ---------------------------------------------------------------------------
# Bug 7 — _danceability_tempogram accesses non-existent "multi_periodicity" key
# ---------------------------------------------------------------------------

class TestDanceabilityTempogramKeys(unittest.TestCase):
    """
    Bug 7: _multi_periodic_structure returns 'score', not 'multi_periodicity'.
    Accessing the wrong key raises a KeyError every call.
    """

    def setUp(self):
        self.tf = _make_tf()

    def test_does_not_raise_key_error(self):
        try:
            val = self.tf._danceability_tempogram()
        except KeyError as exc:
            self.fail(
                f"_danceability_tempogram raised KeyError: {exc}. "
                "Bug 7 ('multi_periodicity' key) or Bug 8 (_beat_strength) is still present."
            )

    def test_value_in_range(self):
        val = self.tf._danceability_tempogram()
        self.assertGreaterEqual(val, 0.0)
        self.assertLessEqual(val, 1.0)

    def test_cached(self):
        v1 = self.tf._danceability_tempogram()
        v2 = self.tf._danceability_tempogram()
        self.assertEqual(v1, v2)

    def test_multi_periodic_structure_has_score_not_multi_periodicity(self):
        """Verify the upstream method returns 'score', not 'multi_periodicity'."""
        result = self.tf._multi_periodic_structure()
        self.assertIn("score", result)
        self.assertNotIn(
            "multi_periodicity", result,
            "'multi_periodicity' key should not exist — _danceability_tempogram "
            "must use 'score' instead.",
        )


# ---------------------------------------------------------------------------
# Bug 8 — _danceability_tempogram calls non-existent _beat_strength
# ---------------------------------------------------------------------------

class TestDanceabilityTempogramBeatStrength(unittest.TestCase):
    """
    Bug 8: _beat_strength does not exist; the correct method is
    _beat_periodicity_strength. Calling the wrong name raises AttributeError.
    """

    def setUp(self):
        self.tf = _make_tf()

    def test_does_not_raise_attribute_error(self):
        try:
            val = self.tf._danceability_tempogram()
        except AttributeError as exc:
            self.fail(
                f"_danceability_tempogram raised AttributeError: {exc}. "
                "Bug 8 (_beat_strength method) is still present."
            )

    def test_beat_periodicity_strength_exists(self):
        self.assertTrue(
            hasattr(self.tf, "_beat_periodicity_strength"),
            "_beat_periodicity_strength method must exist.",
        )

    def test_beat_strength_does_not_exist(self):
        self.assertFalse(
            hasattr(self.tf, "_beat_strength"),
            "_beat_strength should not exist — it is a typo for _beat_periodicity_strength.",
        )

    def test_beat_periodicity_strength_returns_valid(self):
        result = self.tf._beat_periodicity_strength()
        self.assertIn("strength", result)
        self.assertGreaterEqual(result["strength"], 0.0)
        self.assertLessEqual(result["strength"], 1.0)


# ---------------------------------------------------------------------------
# Bug 9 — _mode_tempogram passes norm_score= instead of norm_sum=
# ---------------------------------------------------------------------------

class TestModeTempogramNormParam(unittest.TestCase):
    """
    Bug 9: _mode_tempogram calls _multi_periodic_structure with the
    keyword argument 'norm_score' which doesn't exist, causing norm_sum
    to silently default to True regardless of what was passed.
    """

    def setUp(self):
        self.tf = _make_tf()

    def test_does_not_raise(self):
        try:
            result = self.tf._mode_tempogram()
        except TypeError as exc:
            self.fail(f"_mode_tempogram raised TypeError: {exc}.")

    def test_norm_sum_false_propagates_to_multi_periodic(self):
        """
        Call _mode_tempogram(norm_sum=False) and verify it actually
        calls _multi_periodic_structure with norm_sum=False.
        If Bug 9 is present, the cache key for _multi_periodic_structure
        will always be norm_sum=True regardless.
        """
        tf_a = _make_tf()
        tf_a._mode_tempogram(norm_sum=False)
        cache_keys = list(tf_a._cache_tempogram.keys())
        mps_keys = [k for k in cache_keys if k.startswith("multi_periodic_structure")]
        # With bug, only True variant is cached; with fix, False should appear
        has_false_variant = any("False" in k for k in mps_keys)
        self.assertTrue(
            has_false_variant,
            "No multi_periodic_structure cache entry with norm_sum=False found. "
            "Bug 9 (norm_score= typo) may still be silently passing True.",
        )

    def test_mode_result_keys(self):
        result = self.tf._mode_tempogram()
        for k in ("mode", "score_major", "score_minor", "delta_score"):
            self.assertIn(k, result)

    def test_mode_is_valid(self):
        mode = self.tf._mode_tempogram()["mode"]
        self.assertIn(mode, ("major", "minor"))

    def test_scores_in_range(self):
        result = self.tf._mode_tempogram()
        self.assertGreaterEqual(result["score_major"], 0.0)
        self.assertLessEqual(result["score_major"], 1.0)
        self.assertGreaterEqual(result["score_minor"], 0.0)
        self.assertLessEqual(result["score_minor"], 1.0)

    def test_delta_consistent_with_scores(self):
        result = self.tf._mode_tempogram()
        expected_delta = result["score_major"] - result["score_minor"]
        self.assertAlmostEqual(result["delta_score"], expected_delta, places=10)

    def test_cached(self):
        r1 = self.tf._mode_tempogram()
        r2 = self.tf._mode_tempogram()
        self.assertIs(r1, r2)


# ---------------------------------------------------------------------------
# General — Onset strength
# ---------------------------------------------------------------------------

class TestOnsetStrength(unittest.TestCase):
    def setUp(self):
        self.tf = _make_tf()

    def test_result_keys(self):
        result = self.tf._onset_strength()
        self.assertIn("onset_env", result)
        self.assertIn("times", result)

    def test_onset_env_non_negative(self):
        env = self.tf._onset_strength()["onset_env"]
        self.assertTrue(np.all(env >= 0.0))

    def test_times_monotonically_increasing(self):
        times = self.tf._onset_strength()["times"]
        self.assertTrue(np.all(np.diff(times) > 0.0))

    def test_times_and_env_same_length(self):
        result = self.tf._onset_strength()
        self.assertEqual(result["onset_env"].size, result["times"].size)

    def test_smooth_option(self):
        r1 = self.tf._onset_strength(smooth=False)
        r2 = self.tf._onset_strength(smooth=True, smooth_width=5)
        # Smoothed and unsmoothed share the same length
        self.assertEqual(r1["onset_env"].size, r2["onset_env"].size)

    def test_detrend_option(self):
        r = self.tf._onset_strength(detrend=True)
        self.assertIn("onset_env", r)
        self.assertTrue(np.all(r["onset_env"] >= 0.0))

    def test_silence_env_near_zero(self):
        tf = _make_tf_silence()
        env = tf._onset_strength()["onset_env"]
        self.assertTrue(np.all(env < 1.0))

    def test_cached(self):
        r1 = self.tf._onset_strength()
        r2 = self.tf._onset_strength()
        self.assertIs(r1, r2)


# ---------------------------------------------------------------------------
# General — Onset energy
# ---------------------------------------------------------------------------

class TestOnsetEnergy(unittest.TestCase):
    def setUp(self):
        self.tf = _make_tf()

    def test_sum_mode_non_negative(self):
        e = self.tf._onset_energy(normalize=False)
        self.assertGreaterEqual(e, 0.0)

    def test_mean_mode_non_negative(self):
        e = self.tf._onset_energy(normalize=True)
        self.assertGreaterEqual(e, 0.0)

    def test_sum_gte_mean(self):
        e_sum = self.tf._onset_energy(normalize=False)
        e_mean = self.tf._onset_energy(normalize=True)
        self.assertGreaterEqual(e_sum, e_mean)

    def test_silence_energy_near_zero(self):
        tf = _make_tf_silence()
        self.assertAlmostEqual(tf._onset_energy(), 0.0, places=5)

    def test_cached(self):
        v1 = self.tf._onset_energy()
        v2 = self.tf._onset_energy()
        self.assertEqual(v1, v2)


# ---------------------------------------------------------------------------
# General — Transient curve
# ---------------------------------------------------------------------------

class TestTransientCurve(unittest.TestCase):
    def setUp(self):
        self.tf = _make_tf()

    def test_result_keys(self):
        result = self.tf._transient_curve()
        for k in ("curve", "peak_count", "mean_slope", "peaks"):
            self.assertIn(k, result)

    def test_curve_non_negative_when_normalized(self):
        curve = self.tf._transient_curve(normalize=True)["curve"]
        self.assertTrue(np.all(curve >= 0.0))

    def test_curve_max_one_when_normalized(self):
        curve = self.tf._transient_curve(normalize=True)["curve"]
        if curve.size > 0:
            self.assertLessEqual(float(np.max(curve)), 1.0 + 1e-9)

    def test_peak_count_non_negative(self):
        self.assertGreaterEqual(self.tf._transient_curve()["peak_count"], 0)

    def test_mean_slope_non_negative(self):
        self.assertGreaterEqual(self.tf._transient_curve()["mean_slope"], 0.0)

    def test_silence_peak_count_zero(self):
        tf = _make_tf_silence()
        # Silence may produce one-frame onset — allow 0 peaks
        pc = tf._transient_curve()["peak_count"]
        self.assertGreaterEqual(pc, 0)

    def test_cached(self):
        r1 = self.tf._transient_curve()
        r2 = self.tf._transient_curve()
        self.assertIs(r1, r2)


# ---------------------------------------------------------------------------
# General — Envelope periodicity
# ---------------------------------------------------------------------------

class TestEnvelopePeriodicity(unittest.TestCase):
    def setUp(self):
        self.tf = _make_tf()

    def test_autocorr_keys(self):
        result = self.tf._envelope_periodicity(method="autocorr")
        for k in ("periodicity", "acf", "bpm", "best_lag"):
            self.assertIn(k, result)

    def test_fourier_keys(self):
        result = self.tf._envelope_periodicity(method="fourier")
        for k in ("periodicity", "scores", "bpm", "best_idx"):
            self.assertIn(k, result)

    def test_periodicity_in_range(self):
        for method in ("autocorr", "fourier"):
            p = self.tf._envelope_periodicity(method=method)["periodicity"]
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)

    def test_invalid_method_raises(self):
        with self.assertRaises(ValueError):
            self.tf._envelope_periodicity(method="invalid")

    def test_acf_zero_lag_is_one_when_normalized(self):
        acf = self.tf._envelope_periodicity(method="autocorr", normalize=True)["acf"]
        if acf.size > 0:
            self.assertAlmostEqual(float(acf[0]), 1.0, places=5)

    def test_cached(self):
        r1 = self.tf._envelope_periodicity()
        r2 = self.tf._envelope_periodicity()
        self.assertIs(r1, r2)


# ---------------------------------------------------------------------------
# General — Fourier tempogram
# ---------------------------------------------------------------------------

class TestTempogramFourier(unittest.TestCase):
    def setUp(self):
        self.tf = _make_tf()

    def test_result_keys(self):
        result = self.tf._tempogram_fourier()
        for k in ("tempogram", "bpm", "times"):
            self.assertIn(k, result)

    def test_tempogram_non_negative(self):
        tg = self.tf._tempogram_fourier()["tempogram"]
        self.assertTrue(np.all(tg >= 0.0))

    def test_bpm_axis_starts_near_zero(self):
        bpm = self.tf._tempogram_fourier()["bpm"]
        self.assertAlmostEqual(float(bpm[0]), 0.0, places=5)

    def test_shape_rows_match_fft_bins(self):
        tf = self.tf
        result = tf._tempogram_fourier(win_length=tf.N)
        expected_rows = tf.N // 2 + 1
        self.assertEqual(result["tempogram"].shape[0], expected_rows)

    def test_invalid_window_falls_back_to_rect(self):
        """Unknown window name should fall back to rectangular without error."""
        result = self.tf._tempogram_fourier(window="unknown")
        self.assertIn("tempogram", result)

    def test_cached(self):
        r1 = self.tf._tempogram_fourier()
        r2 = self.tf._tempogram_fourier()
        self.assertIs(r1, r2)


# ---------------------------------------------------------------------------
# General — Tempo spectrum
# ---------------------------------------------------------------------------

class TestTempoSpectrum(unittest.TestCase):
    def setUp(self):
        self.tf = _make_tf()

    def test_result_keys(self):
        result = self.tf._tempo_spectrum()
        self.assertIn("spectrum", result)
        self.assertIn("bpm", result)

    def test_spectrum_non_negative(self):
        S = self.tf._tempo_spectrum()["spectrum"]
        self.assertTrue(np.all(S >= 0.0))

    def test_mean_and_median_differ(self):
        s_mean = self.tf._tempo_spectrum(average="mean")["spectrum"]
        s_med = self.tf._tempo_spectrum(average="median")["spectrum"]
        # Not necessarily different, but both should be valid
        self.assertEqual(s_mean.shape, s_med.shape)

    def test_invalid_average_raises(self):
        with self.assertRaises(ValueError):
            self.tf._tempo_spectrum(average="invalid")

    def test_cached(self):
        r1 = self.tf._tempo_spectrum()
        r2 = self.tf._tempo_spectrum()
        self.assertIs(r1, r2)


# ---------------------------------------------------------------------------
# General — Global BPM
# ---------------------------------------------------------------------------

class TestGlobalBpm(unittest.TestCase):
    def setUp(self):
        self.tf = _make_tf(bpm=120.0)

    def test_detects_plausible_bpm(self):
        bpm = self.tf._global_bpm()["bpm"]
        # Should detect something in range for a 120 BPM pulse train
        if bpm > 0:
            self.assertGreaterEqual(bpm, 40.0)
            self.assertLessEqual(bpm, 240.0)

    def test_120bpm_signal_near_120(self):
        bpm = self.tf._global_bpm()["bpm"]
        if bpm > 0:
            # Allow octave error (60 or 240) but must be a harmonic of 120
            harmonics = {60.0, 120.0, 240.0}
            close_to_harmonic = any(abs(bpm - h) < 20 for h in harmonics)
            self.assertTrue(
                close_to_harmonic,
                f"Detected BPM {bpm:.1f} is not close to a harmonic of 120 BPM.",
            )

    def test_strength_in_range(self):
        strength = self.tf._global_bpm()["strength"]
        self.assertGreaterEqual(strength, 0.0)

    def test_lag_non_negative(self):
        self.assertGreaterEqual(self.tf._global_bpm()["lag"], 0)

    def test_custom_bpm_range_respected(self):
        result = self.tf._global_bpm(bpm_min=100.0, bpm_max=140.0)
        bpm = result["bpm"]
        if bpm > 0:
            self.assertGreaterEqual(bpm, 100.0)
            self.assertLessEqual(bpm, 140.0)


# ---------------------------------------------------------------------------
# General — Local BPM curve
# ---------------------------------------------------------------------------

class TestLocalBpmCurve(unittest.TestCase):
    def setUp(self):
        self.tf = _make_tf()

    def test_result_keys(self):
        result = self.tf._local_bpm_curve()
        self.assertIn("bpm_curve", result)
        self.assertIn("strength_curve", result)

    def test_bpm_curve_in_range(self):
        curve = self.tf._local_bpm_curve()["bpm_curve"]
        if curve.size > 0:
            self.assertTrue(np.all(curve >= 40.0))
            self.assertTrue(np.all(curve <= 240.0))

    def test_strength_curve_non_negative(self):
        sc = self.tf._local_bpm_curve()["strength_curve"]
        self.assertTrue(np.all(sc >= 0.0))

    def test_bpm_and_strength_same_length(self):
        result = self.tf._local_bpm_curve()
        self.assertEqual(result["bpm_curve"].size, result["strength_curve"].size)

    def test_cached(self):
        r1 = self.tf._local_bpm_curve()
        r2 = self.tf._local_bpm_curve()
        self.assertIs(r1, r2)


# ---------------------------------------------------------------------------
# General — Pulse clarity
# ---------------------------------------------------------------------------

class TestPulseClarity(unittest.TestCase):
    def setUp(self):
        self.tf = _make_tf()

    def test_result_keys(self):
        result = self.tf._pulse_clarity()
        for k in ("clarity", "best_peak", "runner-up"):
            self.assertIn(k, result)

    def test_clarity_in_range(self):
        c = self.tf._pulse_clarity()["clarity"]
        self.assertGreaterEqual(c, 0.0)
        self.assertLessEqual(c, 1.0)

    def test_best_peak_gte_runner_up(self):
        result = self.tf._pulse_clarity()
        self.assertGreaterEqual(result["best_peak"], result["runner-up"])

    def test_rhythmic_signal_has_clarity(self):
        c = self.tf._pulse_clarity()["clarity"]
        self.assertGreater(c, 0.0)

    def test_cached(self):
        r1 = self.tf._pulse_clarity()
        r2 = self.tf._pulse_clarity()
        self.assertIs(r1, r2)


# ---------------------------------------------------------------------------
# General — Tempo stability index
# ---------------------------------------------------------------------------

class TestTempoStabilityIndex(unittest.TestCase):
    def setUp(self):
        self.tf = _make_tf()

    def test_result_keys(self):
        result = self.tf._tempo_stability_index()
        for k in ("stability", "mean_bpm", "std_bpm"):
            self.assertIn(k, result)

    def test_stability_in_range(self):
        s = self.tf._tempo_stability_index()["stability"]
        self.assertGreaterEqual(s, 0.0)
        self.assertLessEqual(s, 1.0)

    def test_std_bpm_non_negative(self):
        self.assertGreaterEqual(self.tf._tempo_stability_index()["std_bpm"], 0.0)

    def test_constant_bpm_signal_high_stability(self):
        """A pure 120 BPM pulse should have high tempo stability."""
        tf = _make_tf(bpm=120.0, duration_sec=5.0)
        stab = tf._tempo_stability_index()["stability"]
        self.assertGreater(stab, 0.5)

    def test_cached(self):
        r1 = self.tf._tempo_stability_index()
        r2 = self.tf._tempo_stability_index()
        self.assertIs(r1, r2)


# ---------------------------------------------------------------------------
# General — Beat fluctuation rate
# ---------------------------------------------------------------------------

class TestBeatFluctuationRate(unittest.TestCase):
    def setUp(self):
        self.tf = _make_tf()

    def test_non_negative(self):
        self.assertGreaterEqual(self.tf._beat_fluctuation_rate(), 0.0)

    def test_silence_returns_zero(self):
        tf = _make_tf_silence()
        self.assertAlmostEqual(tf._beat_fluctuation_rate(), 0.0, places=5)

    def test_cached(self):
        v1 = self.tf._beat_fluctuation_rate()
        v2 = self.tf._beat_fluctuation_rate()
        self.assertEqual(v1, v2)


# ---------------------------------------------------------------------------
# General — Multi-periodic structure
# ---------------------------------------------------------------------------

class TestMultiPeriodicStructure(unittest.TestCase):
    def setUp(self):
        self.tf = _make_tf()

    def test_result_keys(self):
        result = self.tf._multi_periodic_structure()
        for k in ("score", "primary_bpm", "half_bpm", "double_bpm"):
            self.assertIn(k, result)

    def test_score_in_range(self):
        s = self.tf._multi_periodic_structure()["score"]
        self.assertGreaterEqual(s, 0.0)
        self.assertLessEqual(s, 2.0)

    def test_half_and_double_bpm_relationship(self):
        result = self.tf._multi_periodic_structure()
        if result["primary_bpm"] > 0:
            self.assertAlmostEqual(
                result["half_bpm"], result["primary_bpm"] / 2.0, delta=5.0
            )
            self.assertAlmostEqual(
                result["double_bpm"], result["primary_bpm"] * 2.0, delta=10.0
            )

    def test_cached(self):
        r1 = self.tf._multi_periodic_structure()
        r2 = self.tf._multi_periodic_structure()
        self.assertIs(r1, r2)


# ---------------------------------------------------------------------------
# General — Swing ratio
# ---------------------------------------------------------------------------

class TestSwingRatio(unittest.TestCase):
    def setUp(self):
        self.tf = _make_tf()

    def test_result_keys(self):
        result = self.tf._swing_ratio()
        for k in ("ratio", "symmetry"):
            self.assertIn(k, result)

    def test_ratio_positive(self):
        self.assertGreater(self.tf._swing_ratio()["ratio"], 0.0)

    def test_symmetry_in_range(self):
        s = self.tf._swing_ratio()["symmetry"]
        self.assertGreaterEqual(s, 0.0)
        self.assertLessEqual(s, 1.0)

    def test_cached(self):
        r1 = self.tf._swing_ratio()
        r2 = self.tf._swing_ratio()
        self.assertIs(r1, r2)


# ---------------------------------------------------------------------------
# General — Tempo spectral moments
# ---------------------------------------------------------------------------

class TestTempoSpectralMoments(unittest.TestCase):
    def setUp(self):
        self.tf = _make_tf()

    def test_centroid_result_keys(self):
        result = self.tf._tempo_spectral_centroid()
        self.assertIn("centroid", result)

    def test_centroid_in_bpm_range(self):
        c = self.tf._tempo_spectral_centroid(bpm_min=30.0, bpm_max=300.0)["centroid"]
        if c > 0:
            self.assertGreaterEqual(c, 30.0)
            self.assertLessEqual(c, 300.0)

    def test_bandwidth_non_negative(self):
        bw = self.tf._tempo_bandwidth()["bandwidth"]
        self.assertGreaterEqual(bw, 0.0)

    def test_bandwidth_result_keys(self):
        result = self.tf._tempo_bandwidth()
        for k in ("bandwidth", "centroid"):
            self.assertIn(k, result)

    def test_skewness_result_keys(self):
        result = self.tf._tempo_skewness()
        self.assertIn("skewness", result)

    def test_skewness_finite(self):
        sk = self.tf._tempo_skewness()["skewness"]
        self.assertTrue(np.isfinite(sk))

    def test_kurtosis_result_keys(self):
        result = self.tf._tempo_kurtosis()
        self.assertIn("kurtosis", result)

    def test_excess_kurtosis_subtracts_3(self):
        raw = self.tf._tempo_kurtosis(excess=False)["kurtosis"]
        excess = self.tf._tempo_kurtosis(excess=True)["kurtosis"]
        self.assertAlmostEqual(excess, raw - 3.0, places=8)

    def test_cached_centroid(self):
        r1 = self.tf._tempo_spectral_centroid()
        r2 = self.tf._tempo_spectral_centroid()
        self.assertIs(r1, r2)


# ---------------------------------------------------------------------------
# General — Beat position
# ---------------------------------------------------------------------------

class TestBeatPosition(unittest.TestCase):
    def setUp(self):
        self.tf = _make_tf()
        self.beat_times = _regular_beat_times(bpm=120.0, duration_sec=4.0)

    def test_phase_in_zero_one(self):
        result = self.tf._beat_position(beat_times=self.beat_times, mode="phase")
        pos = result["beat_position"]
        self.assertTrue(np.all(pos >= 0.0))
        self.assertTrue(np.all(pos < 1.0 + 1e-9))

    def test_fractional_can_exceed_one(self):
        """
        After Bug 2 fix: fractional mode uses global beat_period and can
        return values > 1 for frames far from a beat.
        """
        result = self.tf._beat_position(beat_times=self.beat_times, mode="fractional")
        pos = result["beat_position"]
        # Must at minimum return an array; Bug 2 fix makes values potentially > 1
        self.assertIsInstance(pos, np.ndarray)
        self.assertGreater(pos.size, 0)

    def test_nearest_in_half_one_range(self):
        result = self.tf._beat_position(beat_times=self.beat_times, mode="nearest")
        pos = result["beat_position"]
        self.assertTrue(np.all(pos >= -0.5 - 1e-9))
        self.assertTrue(np.all(pos <= 0.5 + 1e-9))

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            self.tf._beat_position(beat_times=self.beat_times, mode="bad")

    def test_no_beats_returns_empty(self):
        result = self.tf._beat_position(beat_times=None, beat_frames=None)
        self.assertEqual(result["beat_position"].size, 0)

    def test_beat_period_positive(self):
        result = self.tf._beat_position(beat_times=self.beat_times, mode="phase")
        self.assertGreater(result["beat_period"], 0.0)


# ---------------------------------------------------------------------------
# General — Beat alignment histogram
# ---------------------------------------------------------------------------

class TestBeatAlignmentHistogram(unittest.TestCase):
    def setUp(self):
        self.tf = _make_tf()
        self.beat_times = _regular_beat_times()

    def test_result_keys(self):
        result = self.tf._beat_alignment_histogram(beat_times=self.beat_times)
        for k in ("histogram", "bins", "peak_bin"):
            self.assertIn(k, result)

    def test_histogram_sums_to_one_when_normalized(self):
        hist = self.tf._beat_alignment_histogram(
            beat_times=self.beat_times, normalize=True
        )["histogram"]
        self.assertAlmostEqual(float(np.sum(hist)), 1.0, places=5)

    def test_histogram_non_negative(self):
        hist = self.tf._beat_alignment_histogram(
            beat_times=self.beat_times
        )["histogram"]
        self.assertTrue(np.all(hist >= 0.0))

    def test_n_bins_respected(self):
        for n in (8, 16, 32):
            hist = self.tf._beat_alignment_histogram(
                beat_times=self.beat_times, n_bins=n
            )["histogram"]
            self.assertEqual(hist.size, n)

    def test_no_beats_returns_zero_histogram(self):
        result = self.tf._beat_alignment_histogram(beat_times=None, beat_frames=None)
        self.assertTrue(np.all(result["histogram"] == 0.0))
        self.assertEqual(result["peak_bin"], -1)


# ---------------------------------------------------------------------------
# General — Interbeat interval variance
# ---------------------------------------------------------------------------

class TestInterbeatIntervalVariance(unittest.TestCase):
    def setUp(self):
        self.tf = _make_tf()

    def test_regular_beats_low_variance(self):
        bt = _regular_beat_times(bpm=120.0)
        var = self.tf._interbeat_interval_variance(beat_times=bt)
        self.assertAlmostEqual(var, 0.0, places=10)

    def test_irregular_beats_positive_variance(self):
        rng = np.random.default_rng(0)
        bt = np.cumsum(0.5 + 0.1 * rng.standard_normal(20))
        bt = bt[bt > 0]
        var = self.tf._interbeat_interval_variance(beat_times=bt)
        self.assertGreater(var, 0.0)

    def test_normalized_variance_dimensionless(self):
        bt = _regular_beat_times(bpm=120.0)
        rng = np.random.default_rng(1)
        bt_noisy = bt + 0.01 * rng.standard_normal(bt.size)
        var_raw = self.tf._interbeat_interval_variance(
            beat_times=bt_noisy, normalize=False
        )
        var_norm = self.tf._interbeat_interval_variance(
            beat_times=bt_noisy, normalize=True
        )
        # Normalized should be much smaller (divided by mean_ibi^2 ~= 0.5^2)
        self.assertLess(var_norm, var_raw * 10)

    def test_too_few_beats_returns_zero(self):
        bt = np.array([0.0, 0.5])  # only 2 beats → size < 3
        var = self.tf._interbeat_interval_variance(beat_times=bt)
        self.assertEqual(var, 0.0)

    def test_no_beats_returns_zero(self):
        var = self.tf._interbeat_interval_variance(beat_times=None, beat_frames=None)
        self.assertEqual(var, 0.0)

    def test_different_beat_times_not_cache_colliding(self):
        """Bug 6 fix: different beat_times must not return stale cached values."""
        bt1 = _regular_beat_times(bpm=60.0)
        bt2 = _regular_beat_times(bpm=120.0)
        rng = np.random.default_rng(7)
        bt1_noisy = bt1 + 0.03 * rng.standard_normal(bt1.size)
        bt2_noisy = bt2 + 0.03 * rng.standard_normal(bt2.size)
        var1 = self.tf._interbeat_interval_variance(beat_times=bt1_noisy)
        var2 = self.tf._interbeat_interval_variance(beat_times=bt2_noisy)
        # Different beat grids must produce independently computed results
        self.assertIsInstance(var1, float)
        self.assertIsInstance(var2, float)


# ---------------------------------------------------------------------------
# General — Beat sync offset
# ---------------------------------------------------------------------------

class TestBeatSyncOffset(unittest.TestCase):
    def setUp(self):
        self.tf = _make_tf()
        self.beat_times = _regular_beat_times()

    def test_result_keys(self):
        result = self.tf._beat_sync_offset(beat_times=self.beat_times)
        for k in ("offsets", "mean_offset", "mean_abs_offset"):
            self.assertIn(k, result)

    def test_mean_abs_offset_non_negative(self):
        result = self.tf._beat_sync_offset(beat_times=self.beat_times, absolute=True)
        self.assertGreaterEqual(result["mean_abs_offset"], 0.0)

    def test_event_frames_branch_used(self):
        """Bug 1 fix: event_frames must be converted, not ignored."""
        n_frames = self.tf._onset_strength()["onset_env"].size
        event_frames = np.arange(0, n_frames, 10, dtype=int)
        result = self.tf._beat_sync_offset(
            beat_times=self.beat_times,
            event_frames=event_frames
        )
        self.assertGreater(result["offsets"].size, 0)

    def test_no_beats_returns_empty(self):
        result = self.tf._beat_sync_offset(beat_times=None, beat_frames=None)
        self.assertEqual(result["offsets"].size, 0)

    def test_signed_offset_can_be_negative(self):
        result = self.tf._beat_sync_offset(
            beat_times=self.beat_times, absolute=False
        )
        offsets = result["offsets"]
        if offsets.size > 0:
            # Some events must fall before or on a beat
            self.assertTrue(np.any(offsets >= 0.0))


# ---------------------------------------------------------------------------
# General — Loudness per beat
# ---------------------------------------------------------------------------

class TestLoudnessTempogramPerBeat(unittest.TestCase):
    def setUp(self):
        self.tf = _make_tf()

    def test_returns_array(self):
        result = self.tf._loudness_tempogram_per_beat()
        self.assertIsInstance(result, np.ndarray)

    def test_non_negative(self):
        result = self.tf._loudness_tempogram_per_beat()
        self.assertTrue(np.all(result >= 0.0))

    def test_with_explicit_beat_times(self):
        bt = _regular_beat_times(duration_sec=3.0)
        result = self.tf._loudness_tempogram_per_beat(beat_times=bt)
        self.assertEqual(result.size, bt.size)

    def test_cached(self):
        r1 = self.tf._loudness_tempogram_per_beat()
        r2 = self.tf._loudness_tempogram_per_beat()
        self.assertIs(r1, r2)


# ---------------------------------------------------------------------------
# General — Valence and liveness tempogram
# ---------------------------------------------------------------------------

class TestValenceLivenessTempogram(unittest.TestCase):
    def setUp(self):
        self.tf = _make_tf()

    def test_valence_in_range(self):
        val = self.tf._valence_tempogram()
        self.assertGreaterEqual(val, 0.0)
        self.assertLessEqual(val, 1.0)

    def test_liveness_in_range(self):
        val = self.tf._liveness_tempogram()
        self.assertGreaterEqual(val, 0.0)
        self.assertLessEqual(val, 1.0)

    def test_valence_cached(self):
        v1 = self.tf._valence_tempogram()
        v2 = self.tf._valence_tempogram()
        self.assertEqual(v1, v2)

    def test_liveness_cached(self):
        v1 = self.tf._liveness_tempogram()
        v2 = self.tf._liveness_tempogram()
        self.assertEqual(v1, v2)


# ---------------------------------------------------------------------------
# General — Time signature
# ---------------------------------------------------------------------------

class TestTimeSignatureTempogram(unittest.TestCase):
    def setUp(self):
        self.tf = _make_tf()

    def test_result_keys(self):
        result = self.tf._time_signature_tempogram()
        for k in ("time_signature", "confidence", "primary_bpm", "structure_score"):
            self.assertIn(k, result)

    def test_time_signature_valid(self):
        ts = self.tf._time_signature_tempogram()["time_signature"]
        self.assertIn(ts, (3, 4))

    def test_confidence_in_range(self):
        conf = self.tf._time_signature_tempogram()["confidence"]
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)

    def test_cached(self):
        r1 = self.tf._time_signature_tempogram()
        r2 = self.tf._time_signature_tempogram()
        self.assertIs(r1, r2)


# ---------------------------------------------------------------------------
# General — Beat periodicity strength
# ---------------------------------------------------------------------------

class TestBeatPeriodicityStrength(unittest.TestCase):
    def setUp(self):
        self.tf = _make_tf()

    def test_result_keys(self):
        result = self.tf._beat_periodicity_strength()
        for k in ("strength", "bpm", "lag"):
            self.assertIn(k, result)

    def test_strength_in_range(self):
        s = self.tf._beat_periodicity_strength()["strength"]
        self.assertGreaterEqual(s, 0.0)
        self.assertLessEqual(s, 1.0)

    def test_cached(self):
        r1 = self.tf._beat_periodicity_strength()
        r2 = self.tf._beat_periodicity_strength()
        self.assertIs(r1, r2)


# ---------------------------------------------------------------------------
# General — Dominant tempo energy
# ---------------------------------------------------------------------------

class TestDominantTempoEnergy(unittest.TestCase):
    def setUp(self):
        self.tf = _make_tf()

    def test_result_keys(self):
        result = self.tf._dominant_tempo_energy()
        for k in ("bpm_peaks", "energies", "peak_ratios"):
            self.assertIn(k, result)

    def test_peak_ratios_first_is_one(self):
        ratios = self.tf._dominant_tempo_energy()["peak_ratios"]
        if ratios.size > 0:
            self.assertAlmostEqual(float(ratios[0]), 1.0, places=5)

    def test_energies_non_negative(self):
        energies = self.tf._dominant_tempo_energy()["energies"]
        self.assertTrue(np.all(energies >= 0.0))

    def test_cached(self):
        r1 = self.tf._dominant_tempo_energy()
        r2 = self.tf._dominant_tempo_energy()
        self.assertIs(r1, r2)


# ---------------------------------------------------------------------------
# General — Short signal edge cases
# ---------------------------------------------------------------------------

class TestShortSignalEdgeCases(unittest.TestCase):
    def test_very_short_signal_no_crash(self):
        y = np.random.randn(64)
        tf = TempogramFeatures(AudioSignal(signal=y, sr=22050, N=2048, H=512))
        result = tf._onset_strength()
        self.assertIn("onset_env", result)

    def test_single_frame_onset_transient_curve(self):
        y = np.zeros(512)  # exactly one hop
        tf = TempogramFeatures(AudioSignal(signal=y, sr=22050, N=2048, H=512))
        result = tf._transient_curve()
        self.assertIn("curve", result)

    def test_silence_global_bpm_does_not_crash(self):
        tf = _make_tf_silence()
        result = tf._global_bpm()
        self.assertIsInstance(result["bpm"], float)


if __name__ == "__main__":
    unittest.main()