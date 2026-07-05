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

from audio_features.mfcc_features import MFCCFeatures

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
 
def _make_signal(duration_sec=2.0, sr=22050, N=2048, H=512, freq_hz=440.0):
    t = np.linspace(0, duration_sec, int(duration_sec * sr), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * freq_hz * t)
    y += 0.25 * np.sin(2 * np.pi * 2 * freq_hz * t)
    return AudioSignal(signal=y, sr=sr, N=N, H=H)
 
 
def _make_silence(duration_sec=2.0, sr=22050, N=2048, H=512):
    y = np.zeros(int(duration_sec * sr))
    return AudioSignal(signal=y, sr=sr, N=N, H=H)
 
 
def _make_mf(duration_sec=2.0, freq_hz=440.0, n_mfcc=13, **kwargs):
    sig = _make_signal(duration_sec=duration_sec, freq_hz=freq_hz)
    return MFCCFeatures(sig, n_mfcc=n_mfcc, **kwargs)
 
 
def _make_mf_silence(n_mfcc=13):
    return MFCCFeatures(_make_silence(), n_mfcc=n_mfcc)
 
 
# ---------------------------------------------------------------------------
# Bug 1 — _mfcc_autocorrelation normalises by r[0] (wrong) vs r[center]
# ---------------------------------------------------------------------------
 
class TestMfccAutocorrelation(unittest.TestCase):
    """
    Bug 1: np.correlate(..., mode='full') returns length-(2T-1).
    Zero-lag peak is at r[len(r)//2], not r[0].
    Dividing by r[0] (near-zero for zero-mean signals) blows up the ACF.
    """
 
    def setUp(self):
        self.mf = _make_mf()
 
    def test_zero_lag_is_one_when_normalized(self):
        result = self.mf._mfcc_autocorrelation(normalize=True)
        acf = result["acf"]
        np.testing.assert_allclose(
            acf[:, 0], 1.0, atol=1e-5,
            err_msg="ACF lag-0 must be 1.0. Huge values indicate Bug 1 (r[0] normalisation).",
        )
 
    def test_acf_bounded_after_normalization(self):
        acf = self.mf._mfcc_autocorrelation(normalize=True)["acf"]
        self.assertTrue(
            np.all(acf >= -1.0 - 1e-6) and np.all(acf <= 1.0 + 1e-6),
            f"ACF out of [-1,1]: min={acf.min():.4f}, max={acf.max():.4f}.",
        )
 
    def test_acf_finite(self):
        acf = self.mf._mfcc_autocorrelation(normalize=True)["acf"]
        self.assertTrue(np.all(np.isfinite(acf)),
                        "ACF contains non-finite values — Bug 1 blowup likely present.")
 
    def test_acf_shape(self):
        max_lag = 20
        acf = self.mf._mfcc_autocorrelation(max_lag=max_lag)["acf"]
        self.assertEqual(acf.shape, (self.mf.n_mfcc, max_lag + 1))
 
    def test_lags_shape_and_bounds(self):
        max_lag = 20
        lags = self.mf._mfcc_autocorrelation(max_lag=max_lag)["lags"]
        self.assertEqual(lags.shape, (max_lag + 1,))
        self.assertEqual(int(lags[0]), 0)
        self.assertEqual(int(lags[-1]), max_lag)
 
    def test_unnormalized_finite(self):
        acf = self.mf._mfcc_autocorrelation(normalize=False)["acf"]
        self.assertTrue(np.all(np.isfinite(acf)))
 
    def test_cached(self):
        r1 = self.mf._mfcc_autocorrelation()
        r2 = self.mf._mfcc_autocorrelation()
        self.assertIs(r1, r2)
 
 
# ---------------------------------------------------------------------------
# Bug 2 — _mfcc_entropy uses log(n + EPS) instead of log(n)
# ---------------------------------------------------------------------------
 
class TestMfccEntropy(unittest.TestCase):
    """Bug 2: normalisation denominator must be log(n_mfcc), not log(n_mfcc + EPS)."""
 
    def setUp(self):
        self.mf = _make_mf(n_mfcc=13)
 
    def test_entropy_in_range(self):
        ent = self.mf._mfcc_entropy(normalize=True)
        self.assertGreaterEqual(ent, 0.0)
        self.assertLessEqual(ent, 1.0 + 1e-9)
 
    def test_uniform_distribution_entropy_is_one(self):
        n = 13
        log_clean = np.log(n)
        log_buggy = np.log(n + EPS_VALUE)
        H_raw = np.log(n)
        self.assertAlmostEqual(H_raw / log_clean, 1.0, places=12)
        self.assertNotAlmostEqual(H_raw / log_buggy, 1.0, places=12,
                                  msg="log(n+EPS) should differ from 1.0 at places=12.")
 
    def test_entropy_denominator_is_log_n_mfcc(self):
        ent_norm = self.mf._mfcc_entropy(normalize=True)
        X = np.abs(self.mf.mfcc)
        col = np.mean(X, axis=1)
        p = col / (np.sum(col) + EPS_VALUE)
        ent_raw = float(-np.sum(p * np.log(p + EPS_VALUE)))
        if abs(ent_norm) < 1e-12:
            self.skipTest("Entropy is zero — cannot validate denominator.")
        implied = ent_raw / ent_norm
        expected = float(np.log(self.mf.n_mfcc))
        buggy    = float(np.log(self.mf.n_mfcc + EPS_VALUE))
        self.assertLess(
            abs(implied - expected), abs(implied - buggy),
            f"Denominator {implied:.12f} closer to log(n+EPS)={buggy:.12f} "
            f"than log(n)={expected:.12f}. Bug 2 may still be present.",
        )
 
    def test_unnormalized_non_negative(self):
        self.assertGreaterEqual(self.mf._mfcc_entropy(normalize=False), 0.0)
 
    def test_cached(self):
        v1 = self.mf._mfcc_entropy()
        v2 = self.mf._mfcc_entropy()
        self.assertEqual(v1, v2)
 
 
# ---------------------------------------------------------------------------
# Bug 3 — _mfcc_high_order_energy RMS denominator is wrong
# ---------------------------------------------------------------------------
 
class TestMfccHighOrderEnergy(unittest.TestCase):
    """
    Bug 3: order='rms' must use sqrt(mean(mfcc**2)) as denominator,
    not sum(|mfcc|). With start_coeff=0 (full matrix), normalised RMS == 1.0.
    """
 
    def setUp(self):
        self.mf = _make_mf()
 
    def test_l2_full_matrix_normalised_is_one(self):
        val = self.mf._mfcc_high_order_energy(start_coeff=0, normalize=True, order='l2')
        self.assertAlmostEqual(val, 1.0, places=5)
 
    def test_l1_full_matrix_normalised_is_one(self):
        val = self.mf._mfcc_high_order_energy(start_coeff=0, normalize=True, order='l1')
        self.assertAlmostEqual(val, 1.0, places=5)
 
    def test_rms_full_matrix_normalised_is_one(self):
        val = self.mf._mfcc_high_order_energy(start_coeff=0, normalize=True, order='rms')
        self.assertAlmostEqual(
            val, 1.0, places=4,
            msg=f"RMS-normalised energy={val:.6f}, expected ~1.0. Bug 3 may still be present.",
        )
 
    def test_rms_partial_leq_one(self):
        val = self.mf._mfcc_high_order_energy(start_coeff=6, normalize=True, order='rms')
        self.assertLessEqual(val, 1.0 + 1e-6)
        self.assertGreaterEqual(val, 0.0)
 
    def test_invalid_order_raises(self):
        with self.assertRaises(ValueError):
            self.mf._mfcc_high_order_energy(order='invalid')
 
    def test_unnormalized_non_negative(self):
        for order in ('l2', 'l1', 'rms'):
            val = self.mf._mfcc_high_order_energy(normalize=False, order=order)
            self.assertGreaterEqual(val, 0.0, msg=f"order={order} returned negative.")
 
    def test_cached(self):
        v1 = self.mf._mfcc_high_order_energy()
        v2 = self.mf._mfcc_high_order_energy()
        self.assertEqual(v1, v2)
 
 
# ---------------------------------------------------------------------------
# Bug 4 — log_mels scale consistency + S_db always populated
# ---------------------------------------------------------------------------
 
class TestComputeMfccLogMels(unittest.TestCase):
    """Bug 4: S_db must always be populated; MFCC must be finite for both modes."""
 
    def test_s_db_always_populated(self):
        for log_mels in (True, False):
            mf = _make_mf(log_mels=log_mels)
            s_db = getattr(mf, 'S_db', None) if getattr(mf, 'S_db', None) is not None else getattr(mf, 'S_dB', None)
            self.assertIsNotNone(
                s_db,
                f"S_db is None with log_mels={log_mels}. Fix 4 requires S_db always computed.",
            )
 
    def test_s_db_non_positive(self):
        mf = _make_mf(log_mels=False)
        s_db = getattr(mf, 'S_db', None) if getattr(mf, 'S_db', None) is not None else getattr(mf, 'S_dB', None)
        if s_db is not None:
            self.assertTrue(np.all(s_db <= 0.0 + 1e-4))
 
    def test_mfcc_shape_both_modes(self):
        for log_mels in (True, False):
            mf = _make_mf(log_mels=log_mels, n_mfcc=13)
            self.assertEqual(mf.mfcc.shape[0], 13)
            self.assertGreater(mf.mfcc.shape[1], 0)
 
    def test_mfcc_finite_both_modes(self):
        for log_mels in (True, False):
            mf = _make_mf(log_mels=log_mels)
            self.assertTrue(np.all(np.isfinite(mf.mfcc)),
                            f"Non-finite MFCC with log_mels={log_mels}.")
 
    def test_s_non_negative(self):
        self.assertTrue(np.all(_make_mf().S >= 0.0))
 
    def test_times_matches_mfcc_frames(self):
        mf = _make_mf()
        self.assertEqual(mf.times.size, mf.mfcc.shape[1])
 
    def test_times_monotonically_increasing(self):
        mf = _make_mf()
        self.assertTrue(np.all(np.diff(mf.times) > 0.0))
 
 
# ---------------------------------------------------------------------------
# Bug 5 — _mfcc_variance default ddof inconsistent with skewness/kurtosis
# ---------------------------------------------------------------------------
 
class TestMfccVarianceDdof(unittest.TestCase):
    """Bug 5: default ddof must be 0 to match the population moments in skew/kurt."""
 
    def setUp(self):
        self.mf = _make_mf()
 
    def test_default_ddof_is_zero(self):
        import inspect
        src = inspect.getsource(self.mf._mfcc_variance)
        self.assertNotIn("ddof=1", src,
                         "_mfcc_variance still has default ddof=1. Bug 5 not fixed.")
 
    def test_variance_matches_population_m2(self):
        var = self.mf._mfcc_variance(ddof=0)
        mu = self.mf._mfcc_mean()
        x = self.mf.mfcc - mu[:, np.newaxis]
        m2 = np.mean(x**2, axis=1)
        np.testing.assert_allclose(var, m2, rtol=1e-5)
 
    def test_variance_shape(self):
        self.assertEqual(self.mf._mfcc_variance().shape, (self.mf.n_mfcc,))
 
    def test_variance_non_negative(self):
        self.assertTrue(np.all(self.mf._mfcc_variance() >= 0.0))
 
    def test_sample_variance_geq_population(self):
        var0 = self.mf._mfcc_variance(ddof=0)
        var1 = self.mf._mfcc_variance(ddof=1)
        self.assertTrue(np.all(var1 >= var0 - 1e-10))
 
    def test_skewness_finite_and_correct_shape(self):
        skew = self.mf._mfcc_skewness()
        self.assertEqual(skew.shape, (self.mf.n_mfcc,))
        self.assertTrue(np.all(np.isfinite(skew)))
 
    def test_cached(self):
        v1 = self.mf._mfcc_variance()
        v2 = self.mf._mfcc_variance()
        np.testing.assert_array_equal(v1, v2)
 
 
# ---------------------------------------------------------------------------
# Bug 6 — _mfcc_sustain_stability fragile when mu ≈ 0
# ---------------------------------------------------------------------------
 
class TestMfccSustainStability(unittest.TestCase):
    """Bug 6: 1/(1+cv) form must always stay in (0,1] without relying on clip."""
 
    def setUp(self):
        self.mf = _make_mf()
 
    def test_stability_in_range(self):
        val = self.mf._mfcc_sustain_stability(normalize=True)
        self.assertGreaterEqual(val, 0.0)
        self.assertLessEqual(val, 1.0)
 
    def test_stability_positive(self):
        self.assertGreater(self.mf._mfcc_sustain_stability(), 0.0)
 
    def test_near_zero_mean_no_blowup(self):
        mf = _make_mf()
        rng = np.random.default_rng(42)
        mf.mfcc = (rng.standard_normal(mf.mfcc.shape) * 0.001).astype(np.float32)
        mf._cache_mfcc = {k: v for k, v in mf._cache_mfcc.items()
                          if k == "mfcc_default"}
        val = mf._mfcc_sustain_stability(normalize=True)
        self.assertGreaterEqual(val, 0.0,
                                "Stability negative — Bug 6 may still be present.")
        self.assertLessEqual(val, 1.0)
 
    def test_unnormalized_in_range(self):
        val = self.mf._mfcc_sustain_stability(normalize=False)
        self.assertGreaterEqual(val, 0.0)
        self.assertLessEqual(val, 1.0)
 
    def test_explicit_frames(self):
        val = self.mf._mfcc_sustain_stability(attack_frames=5, sustain_frames=20)
        self.assertGreaterEqual(val, 0.0)
        self.assertLessEqual(val, 1.0)
 
    def test_cached(self):
        v1 = self.mf._mfcc_sustain_stability()
        v2 = self.mf._mfcc_sustain_stability()
        self.assertEqual(v1, v2)
 
 
# ---------------------------------------------------------------------------
# Bug 7 — _mfcc_attack_smoothness subtraction formula goes negative
# ---------------------------------------------------------------------------
 
class TestMfccAttackSmoothness(unittest.TestCase):
    """Bug 7: 1/(1 + ratio) form must always stay in (0,1]."""
 
    def setUp(self):
        self.mf = _make_mf()
 
    def test_smoothness_in_range(self):
        val = self.mf._mfcc_attack_smoothness(normalize=True)
        self.assertGreaterEqual(val, 0.0)
        self.assertLessEqual(val, 1.0)
 
    def test_smoothness_positive(self):
        self.assertGreater(self.mf._mfcc_attack_smoothness(), 0.0)
 
    def test_high_step_energy_non_negative(self):
        mf = _make_mf()
        n_mfcc, T = mf.mfcc.shape
        spike = np.zeros((n_mfcc, T), dtype=np.float32)
        spike[:, ::2]  =  100.0
        spike[:, 1::2] = -100.0
        mf.mfcc = spike
        mf._cache_mfcc = {k: v for k, v in mf._cache_mfcc.items()
                          if k == "mfcc_default"}
        val = mf._mfcc_attack_smoothness(normalize=True)
        self.assertGreaterEqual(
            val, 0.0,
            f"Smoothness={val:.6f} negative under high step energy — Bug 7 still present.",
        )
 
    def test_unnormalized_in_range(self):
        val = self.mf._mfcc_attack_smoothness(normalize=False)
        self.assertGreaterEqual(val, 0.0)
        self.assertLessEqual(val, 1.0)
 
    def test_explicit_attack_frames(self):
        val = self.mf._mfcc_attack_smoothness(attack_frames=5)
        self.assertGreaterEqual(val, 0.0)
        self.assertLessEqual(val, 1.0)
 
    def test_cached(self):
        v1 = self.mf._mfcc_attack_smoothness()
        v2 = self.mf._mfcc_attack_smoothness()
        self.assertEqual(v1, v2)
 
 
# ---------------------------------------------------------------------------
# Residual — S_db vs S_dB capitalisation mismatch
# ---------------------------------------------------------------------------
 
class TestSDbAttribute(unittest.TestCase):
    """Residual: whichever capitalisation is canonical, it must be non-None."""
 
    def test_s_db_attribute_not_none(self):
        mf = _make_mf()
        s_db = getattr(mf, 'S_db', None) if getattr(mf, 'S_db', None) is not None else getattr(mf, 'S_dB', None)
        self.assertIsNotNone(s_db,
                             "Neither S_db nor S_dB populated — capitalisation mismatch.")
 
    def test_s_db_is_ndarray(self):
        mf = _make_mf()
        s_db = getattr(mf, 'S_db', None) if getattr(mf, 'S_db', None) is not None else getattr(mf, 'S_dB', None)
        self.assertIsInstance(s_db, np.ndarray)
 
    def test_s_db_shape_matches_mel(self):
        mf = _make_mf()
        s_db = getattr(mf, 'S_db', None) if getattr(mf, 'S_db', None) is not None else getattr(mf, 'S_dB', None)
        self.assertEqual(s_db.shape, mf.S.shape)
 
    def test_s_db_non_positive(self):
        mf = _make_mf()
        s_db = getattr(mf, 'S_db', None) if getattr(mf, 'S_db', None) is not None else getattr(mf, 'S_dB', None)
        self.assertTrue(np.all(s_db <= 0.0 + 1e-4))
 
    def test_at_least_one_capitalisation_non_none(self):
        mf = _make_mf()
        self.assertTrue(
            mf.S_db is not None or getattr(mf, 'S_dB', None) is not None,
        )
 
 
# ---------------------------------------------------------------------------
# General — Core pipeline
# ---------------------------------------------------------------------------
 
class TestComputeMfcc(unittest.TestCase):
    def setUp(self):
        self.mf = _make_mf()
 
    def test_mfcc_shape(self):
        self.assertEqual(self.mf.mfcc.shape[0], self.mf.n_mfcc)
        self.assertGreater(self.mf.mfcc.shape[1], 0)
 
    def test_mfcc_finite(self):
        self.assertTrue(np.all(np.isfinite(self.mf.mfcc)))
 
    def test_mel_spectrogram_non_negative(self):
        self.assertTrue(np.all(self.mf.S >= 0.0))
 
    def test_mel_spectrogram_shape(self):
        self.assertEqual(self.mf.S.shape[0], self.mf.n_mels)
 
    def test_times_length(self):
        self.assertEqual(self.mf.times.size, self.mf.mfcc.shape[1])
 
    def test_cached_result(self):
        r1 = self.mf._compute_mfcc()
        r2 = self.mf._compute_mfcc()
        self.assertIs(r1, r2)
 
    def test_different_n_mfcc(self):
        for n in (8, 13, 20):
            mf = _make_mf(n_mfcc=n)
            self.assertEqual(mf.mfcc.shape[0], n)
 
    def test_compute_false_leaves_mfcc_none(self):
        mf = MFCCFeatures(_make_signal(), compute=False)
        self.assertIsNone(mf.mfcc)
 
 
# ---------------------------------------------------------------------------
# General — Statistical features
# ---------------------------------------------------------------------------
 
class TestMfccStatistics(unittest.TestCase):
    def setUp(self):
        self.mf = _make_mf()
 
    def test_mean_shape(self):
        self.assertEqual(self.mf._mfcc_mean().shape, (self.mf.n_mfcc,))
 
    def test_mean_finite(self):
        self.assertTrue(np.all(np.isfinite(self.mf._mfcc_mean())))
 
    def test_variance_shape(self):
        self.assertEqual(self.mf._mfcc_variance().shape, (self.mf.n_mfcc,))
 
    def test_variance_non_negative(self):
        self.assertTrue(np.all(self.mf._mfcc_variance() >= 0.0))
 
    def test_skewness_shape(self):
        self.assertEqual(self.mf._mfcc_skewness().shape, (self.mf.n_mfcc,))
 
    def test_skewness_finite(self):
        self.assertTrue(np.all(np.isfinite(self.mf._mfcc_skewness())))
 
    def test_kurtosis_shape(self):
        self.assertEqual(self.mf._mfcc_kurtosis().shape, (self.mf.n_mfcc,))
 
    def test_excess_kurtosis_subtracts_three(self):
        raw = self.mf._mfcc_kurtosis(excess=False)
        exc = self.mf._mfcc_kurtosis(excess=True)
        np.testing.assert_allclose(exc, raw - 3.0, atol=1e-10)
 
    def test_raw_kurtosis_non_negative(self):
        raw = self.mf._mfcc_kurtosis(excess=False)
        self.assertTrue(np.all(raw >= 0.0 - 1e-9))
 
    def test_mean_cached(self):
        m1 = self.mf._mfcc_mean()
        m2 = self.mf._mfcc_mean()
        self.assertIs(m1, m2)
 
 
# ---------------------------------------------------------------------------
# General — Temporal features
# ---------------------------------------------------------------------------
 
class TestMfccTemporalFeatures(unittest.TestCase):
    def setUp(self):
        self.mf = _make_mf()
 
    def test_delta_shape(self):
        self.assertEqual(self.mf._mfcc_delta().shape, self.mf.mfcc.shape)
 
    def test_delta2_shape(self):
        self.assertEqual(self.mf._mfcc_delta2().shape, self.mf.mfcc.shape)
 
    def test_delta_finite(self):
        self.assertTrue(np.all(np.isfinite(self.mf._mfcc_delta())))
 
    def test_temporal_stability_shape(self):
        stab = self.mf._mfcc_temporal_stability()
        self.assertEqual(stab.shape, (self.mf.n_mfcc,))
 
    def test_temporal_stability_in_range(self):
        stab = self.mf._mfcc_temporal_stability()
        self.assertTrue(np.all(stab >= 0.0))
        self.assertTrue(np.all(stab <= 1.0))
 
    def test_delta_cached(self):
        d1 = self.mf._mfcc_delta()
        d2 = self.mf._mfcc_delta()
        self.assertIs(d1, d2)
 
    def test_delta_narrow_width(self):
        d = self.mf._mfcc_delta(width=3)
        self.assertEqual(d.shape, self.mf.mfcc.shape)
 
 
# ---------------------------------------------------------------------------
# General — Spectral shape proxies
# ---------------------------------------------------------------------------
 
class TestSpectralShapeProxies(unittest.TestCase):
    def setUp(self):
        self.mf = _make_mf()
 
    def test_spectral_slope_proxy_all_aggregates(self):
        for agg in ("mean", "median", "rms"):
            val = self.mf._mfcc_spectral_slope_proxy(coeff=0, aggregate=agg)
            self.assertTrue(np.isfinite(val), f"aggregate={agg} non-finite.")
 
    def test_spectral_slope_invalid_aggregate_raises(self):
        with self.assertRaises(ValueError):
            self.mf._mfcc_spectral_slope_proxy(aggregate="invalid")
 
    def test_brightness_proxy_finite(self):
        self.assertTrue(np.isfinite(self.mf._mfcc_brightness_proxy()))
 
    def test_brightness_invert_negates(self):
        pos = self.mf._mfcc_brightness_proxy(invert=False)
        neg = self.mf._mfcc_brightness_proxy(invert=True)
        self.assertAlmostEqual(pos, -neg, places=10)
 
    def test_sharpness_non_negative_when_absolute(self):
        self.assertGreaterEqual(self.mf._mfcc_sharpness_proxy(absolute=True), 0.0)
 
    def test_sharpness_invalid_aggregate_raises(self):
        with self.assertRaises(ValueError):
            self.mf._mfcc_sharpness_proxy(aggregate="invalid")
 
    def test_brightness_cached(self):
        v1 = self.mf._mfcc_brightness_proxy()
        v2 = self.mf._mfcc_brightness_proxy()
        self.assertEqual(v1, v2)
 
 
# ---------------------------------------------------------------------------
# General — Energy features
# ---------------------------------------------------------------------------
 
class TestMfccEnergyFeatures(unittest.TestCase):
    def setUp(self):
        self.mf = _make_mf()
 
    def test_frame_energy_shape(self):
        E = self.mf._mfcc_frame_energy(normalize=True)
        self.assertEqual(E.shape, (self.mf.mfcc.shape[1],))
 
    def test_frame_energy_max_one_normalized(self):
        E = self.mf._mfcc_frame_energy(normalize=True)
        self.assertLessEqual(float(np.max(E)), 1.0 + 1e-6)
 
    def test_frame_energy_non_negative(self):
        self.assertTrue(np.all(self.mf._mfcc_frame_energy() >= 0.0))
 
    def test_energy_scalar_in_range(self):
        val = self.mf._mfcc_energy(normalize=True)
        self.assertGreaterEqual(val, 0.0)
        self.assertLessEqual(val, 1.0)
 
    def test_rms_energy_non_negative(self):
        self.assertGreaterEqual(self.mf._mfcc_rms_energy(), 0.0)
 
    def test_rms_energy_normalized_leq_one(self):
        self.assertLessEqual(self.mf._mfcc_rms_energy(normalize=True), 1.0 + 1e-6)
 
    def test_high_order_variance_non_negative(self):
        self.assertGreaterEqual(self.mf._mfcc_high_order_variance(), 0.0)
 
    def test_energy_cached(self):
        v1 = self.mf._mfcc_energy()
        v2 = self.mf._mfcc_energy()
        self.assertEqual(v1, v2)
 
 
# ---------------------------------------------------------------------------
# General — Flux, smoothness, entropy
# ---------------------------------------------------------------------------
 
class TestMfccFluxSmoothnessEntropy(unittest.TestCase):
    def setUp(self):
        self.mf = _make_mf()
 
    def test_flux_non_negative(self):
        self.assertGreaterEqual(self.mf._mfcc_flux(), 0.0)
 
    def test_smoothness_in_range(self):
        val = self.mf._mfcc_smoothness(normalize=True)
        self.assertGreaterEqual(val, 0.0)
        self.assertLessEqual(val, 1.0)
 
    def test_smoothness_index_in_range(self):
        val = self.mf._mfcc_smoothness_index(normalize=True)
        self.assertGreaterEqual(val, 0.0)
        self.assertLessEqual(val, 1.0)
 
    def test_entropy_in_range(self):
        val = self.mf._mfcc_entropy(normalize=True)
        self.assertGreaterEqual(val, 0.0)
        self.assertLessEqual(val, 1.0 + 1e-9)
 
    def test_silence_flux_zero(self):
        mf = _make_mf_silence()
        self.assertAlmostEqual(mf._mfcc_flux(), 0.0, places=5)
 
    def test_flux_cached(self):
        v1 = self.mf._mfcc_flux()
        v2 = self.mf._mfcc_flux()
        self.assertEqual(v1, v2)
 
    def test_smoothness_cached(self):
        v1 = self.mf._mfcc_smoothness()
        v2 = self.mf._mfcc_smoothness()
        self.assertEqual(v1, v2)
 
 
# ---------------------------------------------------------------------------
# General — Noise/speech proxies
# ---------------------------------------------------------------------------
 
class TestMfccNoiseProxies(unittest.TestCase):
    def setUp(self):
        self.mf = _make_mf()
 
    def test_high_order_magnitude_l2_in_range(self):
        val = self.mf._mfcc_high_order_magnitude(mode='l2', normalize=True)
        self.assertGreaterEqual(val, 0.0)
        self.assertLessEqual(val, 1.0 + 1e-6)
 
    def test_high_order_magnitude_l1_in_range(self):
        val = self.mf._mfcc_high_order_magnitude(mode='l1', normalize=True)
        self.assertGreaterEqual(val, 0.0)
        self.assertLessEqual(val, 1.0 + 1e-6)
 
    def test_high_order_magnitude_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            self.mf._mfcc_high_order_magnitude(mode='invalid')
 
    def test_noise_inharmonicity_proxy_non_negative(self):
        self.assertGreaterEqual(self.mf._mfcc_noise_inharmonicity_proxy(), 0.0)
 
    def test_transient_roughness_non_negative(self):
        self.assertGreaterEqual(self.mf._mfcc_transient_roughness(), 0.0)
 
    def test_formant_shape_result_keys(self):
        result = self.mf._mfcc_formant_shape_detection()
        self.assertIn("score", result)
        self.assertIn("pattern", result)
 
    def test_formant_shape_score_non_negative(self):
        self.assertGreaterEqual(self.mf._mfcc_formant_shape_detection()["score"], 0.0)
 
    def test_transient_roughness_cached(self):
        v1 = self.mf._mfcc_transient_roughness()
        v2 = self.mf._mfcc_transient_roughness()
        self.assertEqual(v1, v2)
 
 
# ---------------------------------------------------------------------------
# General — Spotify audio features
# ---------------------------------------------------------------------------
 
class TestSpotifyAudioFeatures(unittest.TestCase):
    def setUp(self):
        self.mf = _make_mf()
 
    def test_expected_keys(self):
        result = self.mf.spotify_audio_features()
        expected = {
            "loudness", "energy", "speechiness",
            "acousticness", "valence", "liveness", "instrumentalness"
        }
        self.assertEqual(set(result.keys()), expected)
 
    def test_all_values_in_range(self):
        result = self.mf.spotify_audio_features()
        for key, val in result.items():
            self.assertGreaterEqual(val, 0.0, msg=f"{key} < 0")
            self.assertLessEqual(val, 1.0, msg=f"{key} > 1")
 
    def test_all_values_finite(self):
        result = self.mf.spotify_audio_features()
        for key, val in result.items():
            self.assertTrue(np.isfinite(val), msg=f"{key} not finite")
 
    def test_silence_features_in_range(self):
        result = _make_mf_silence().spotify_audio_features()
        for key, val in result.items():
            self.assertGreaterEqual(val, 0.0, msg=f"silence: {key} < 0")
            self.assertLessEqual(val, 1.0, msg=f"silence: {key} > 1")
 
 
# ---------------------------------------------------------------------------
# General — Edge cases
# ---------------------------------------------------------------------------
 
class TestMfccEdgeCases(unittest.TestCase):
    def test_very_short_signal_no_crash(self):
        mf = MFCCFeatures(AudioSignal(signal=np.random.randn(256), sr=22050, N=2048, H=512))
        self.assertIsNotNone(mf.mfcc)
 
    def test_single_sample_no_crash(self):
        mf = MFCCFeatures(AudioSignal(signal=np.array([0.5]), sr=22050, N=2048, H=512))
        self.assertIsNotNone(mf.mfcc)
 
    def test_different_n_mels(self):
        for n_mels in (20, 40, 80):
            mf = MFCCFeatures(_make_signal(), n_mels=n_mels, n_mfcc=13)
            self.assertEqual(mf.S.shape[0], n_mels)
 
    def test_custom_hop_and_fft(self):
        mf = MFCCFeatures(_make_signal(), n_fft=1024, hop_length=256, n_mfcc=13)
        self.assertEqual(mf.N, 1024)
        self.assertEqual(mf.H, 256)
 
    def test_lifter_no_crash(self):
        mf = MFCCFeatures(_make_signal(), lifter=22, n_mfcc=13)
        self.assertTrue(np.all(np.isfinite(mf.mfcc)))
 
    def test_silence_mfcc_finite(self):
        self.assertTrue(np.all(np.isfinite(_make_mf_silence().mfcc)))
 
    def test_silence_energy_near_zero(self):
        self.assertAlmostEqual(
            _make_mf_silence()._mfcc_energy(normalize=False), 0.0, places=3
        )
 
 
if __name__ == "__main__":
    unittest.main()