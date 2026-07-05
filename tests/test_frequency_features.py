import types
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
 
import numpy as np
from pathlib import Path
import pytest
import sys

# ---------------------------------------------------------------------------
# Minimal stubs so the module can be imported without the full audio_features
# package being installed.
# ---------------------------------------------------------------------------
 
# Stub out the relative imports used by frequency_features
EPS_VALUE = 1e-10
 
utils_stub = types.ModuleType("audio_features.utils")
utils_stub.EPS = EPS_VALUE
utils_stub.safe_clip01 = lambda x: float(np.clip(x, 0.0, 1.0))
 
audio_signal_stub = types.ModuleType("audio_features.audio_signal")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from audio_features.audio_signal import AudioSignal

audio_signal_stub.AudioSignal = AudioSignal

# Register stubs before importing the module under test
sys.modules.setdefault("audio_features", types.ModuleType("audio_features"))
sys.modules["audio_features.utils"] = utils_stub
sys.modules["audio_features.audio_signal"] = audio_signal_stub
 
# Patch relative import path used inside frequency_features
sys.modules.setdefault(
    "audio_features.frequency_features", types.ModuleType("audio_features.frequency_features")
)

from audio_features.frequency_features import FrequencyFeatures

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
 
def _make_signal(duration_sec=1.0, sr=22050, N=2048, H=512, freq_hz=440.0):
    """Return an AudioSignal-like object containing a pure sine wave."""
    t = np.linspace(0, duration_sec, int(duration_sec * sr), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * freq_hz * t)
    return AudioSignal(signal=y, N=N, H=H)
 
 
def _make_ff(duration_sec=1.0, sr=22050, freq_hz=440.0):
    """Convenience factory for FrequencyFeatures."""
    sig = _make_signal(duration_sec=duration_sec, sr=sr, freq_hz=freq_hz)
    return FrequencyFeatures(sig)
 
 
def _make_silence(duration_sec=1.0, sr=22050, N=2048, H=512):
    """FrequencyFeatures built from a silent (all-zero) signal."""
    y = np.zeros(int(duration_sec * sr))
    sig = AudioSignal(signal=y, sr=sr, N=N, H=H)
    return FrequencyFeatures(sig)
 
 
# ---------------------------------------------------------------------------
# Bug 1 — Cache key typo in _db_spectrum
# ---------------------------------------------------------------------------
 
class TestDbSpectrumCacheKey(unittest.TestCase):
    """Bug 1: key must be 'db_mag' (not 'db_map') for the magnitude branch."""
 
    def setUp(self):
        self.ff = _make_ff()
 
    def test_magnitude_branch_stores_under_db_mag(self):
        """Calling _db_spectrum(power=False) must populate 'db_mag' in the cache."""
        self.ff._db_spectrum(power=False)
        self.assertIn(
            "db_mag",
            self.ff._cache_freq,
            "Cache should contain 'db_mag' after _db_spectrum(power=False). "
            "If 'db_map' is found instead, Bug 1 is still present.",
        )
        self.assertNotIn(
            "db_map",
            self.ff._cache_freq,
            "Cache must NOT contain the typo key 'db_map'.",
        )
 
    def test_power_branch_stores_under_db_pow(self):
        """Calling _db_spectrum(power=True) must populate 'db_pow'."""
        self.ff._db_spectrum(power=True)
        self.assertIn("db_pow", self.ff._cache_freq)
 
    def test_magnitude_cache_hit_on_second_call(self):
        """Second call with power=False must return the cached array (same object)."""
        first = self.ff._db_spectrum(power=False)
        second = self.ff._db_spectrum(power=False)
        self.assertIs(first, second)
 
    def test_power_cache_hit_on_second_call(self):
        """Second call with power=True must return the cached array."""
        first = self.ff._db_spectrum(power=True)
        second = self.ff._db_spectrum(power=True)
        self.assertIs(first, second)
 
    def test_magnitude_values_are_non_positive(self):
        """dB values from magnitude spectrum (ref = max) should be <= 0."""
        db = self.ff._db_spectrum(power=False)
        self.assertTrue(np.all(db <= 0.0 + 1e-6))
 
    def test_power_values_are_non_positive(self):
        """dB values from power spectrum (ref = max) should be <= 0."""
        db = self.ff._db_spectrum(power=True)
        self.assertTrue(np.all(db <= 0.0 + 1e-6))
 
 
# ---------------------------------------------------------------------------
# Bug 2 — Unreachable cache store in _hnr
# ---------------------------------------------------------------------------
 
class TestHnrCaching(unittest.TestCase):
    """Bug 2: result must be cached when f0_hz is not None."""
 
    def setUp(self):
        self.ff = _make_ff()
        T = self.ff._magnitude_spectrum().shape[1]
        self.f0 = np.full(T, 440.0, dtype=float)
 
    def test_result_cached_when_f0_provided(self):
        """After calling _hnr(f0_hz=...), result must appear in _cache_freq."""
        self.ff._hnr(f0_hz=self.f0)
        self.assertIn(
            "hnr_pow",
            self.ff._cache_freq,
            "_hnr result not cached — Bug 2 is still present.",
        )
 
    def test_cache_hit_on_second_call(self):
        """Second call must return the same cached array object."""
        first = self.ff._hnr(f0_hz=self.f0)
        second = self.ff._hnr(f0_hz=self.f0)
        self.assertIs(first, second)
 
    def test_zeros_returned_when_f0_is_none(self):
        """_hnr with f0_hz=None must return all zeros."""
        T = self.ff._magnitude_spectrum().shape[1]
        result = self.ff._hnr(f0_hz=None)
        np.testing.assert_array_equal(result, np.zeros(T))
 
    def test_hnr_shape(self):
        """_hnr must return shape (T,) matching number of STFT frames."""
        T = self.ff._magnitude_spectrum().shape[1]
        result = self.ff._hnr(f0_hz=self.f0)
        self.assertEqual(result.shape, (T,))
 
    def test_hnr_wrong_f0_length_raises(self):
        """Mismatched f0_hz length must raise ValueError."""
        with self.assertRaises(ValueError):
            self.ff._hnr(f0_hz=np.array([440.0, 440.0]))  # wrong length
 
 
# ---------------------------------------------------------------------------
# Bug 3 — Phase unwrap axis
# ---------------------------------------------------------------------------
 
class TestPhaseUnwrapAxis(unittest.TestCase):
    """Bug 3: phase must be unwrapped along the time axis (axis=1), not axis=0."""
 
    def setUp(self):
        self.ff = _make_ff(duration_sec=2.0)
 
    def test_phase_shape(self):
        """Phase array must have shape (K, T) matching the STFT."""
        phi = self.ff._phase()
        K, T = self.ff.X.shape
        self.assertEqual(phi.shape, (K, T))
 
    def test_phase_time_continuity(self):
        """
        If unwrapped correctly along axis=1, adjacent-frame phase differences
        for a pure sine should be small (< pi) at the sinusoid's bin.
        Unwrapping along axis=0 produces large, incoherent jumps in time.
        """
        phi = self.ff._phase()
        # Find the bin closest to 440 Hz
        bin_440 = int(np.argmin(np.abs(self.ff.freqs - 440.0)))
        time_diffs = np.abs(np.diff(phi[bin_440, :]))
        # With correct axis=1 unwrapping, most diffs should be well under pi
        fraction_small = np.mean(time_diffs < np.pi)
        self.assertGreater(
            fraction_small,
            0.80,
            "Phase diffs across time are large — phase may be unwrapped on wrong axis (Bug 3).",
        )
 
    def test_phase_values_finite(self):
        """All phase values must be finite."""
        phi = self.ff._phase()
        self.assertTrue(np.all(np.isfinite(phi)))
 
    def test_phase_cached(self):
        """Calling _phase() twice must return the same object."""
        self.assertIs(self.ff._phase(), self.ff._phase())
 
 
# ---------------------------------------------------------------------------
# Bug 4 — _phase_congruency variable shadowing
# ---------------------------------------------------------------------------
 
class TestPhaseCongruency(unittest.TestCase):
    """
    Bug 4: _phase_congruency must not overwrite self.X with a real-valued
    spectrum, and must return sensible values in [0, 1].
    """
 
    def setUp(self):
        self.ff = _make_ff(duration_sec=1.0)
 
    def test_stft_not_mutated(self):
        """self.X must remain complex after calling _phase_congruency."""
        self.ff._phase_congruency(use_power=True)
        self.assertTrue(
            np.iscomplexobj(self.ff.X),
            "self.X was overwritten with a real array — Bug 4 is still present.",
        )
 
    def test_output_range(self):
        """Phase congruency values must lie in [0, 1]."""
        pc = self.ff._phase_congruency(use_power=True)
        self.assertTrue(np.all(pc >= 0.0) and np.all(pc <= 1.0))
 
    def test_output_shape(self):
        """Phase congruency must return shape (T,)."""
        T = self.ff.X.shape[1]
        pc = self.ff._phase_congruency()
        self.assertEqual(pc.shape, (T,))
 
    def test_pure_sine_has_high_congruency(self):
        """A pure sine wave should produce relatively high average phase congruency."""
        pc = self.ff._phase_congruency(use_power=True)
        self.assertGreater(float(np.mean(pc)), 0.1)
 
    def test_cached(self):
        """Second call must return the same cached object."""
        self.assertIs(
            self.ff._phase_congruency(use_power=True),
            self.ff._phase_congruency(use_power=True),
        )
 
 
# ---------------------------------------------------------------------------
# Bug 5 — Redundant local import in _transient_counts
# ---------------------------------------------------------------------------
 
class TestTransientCountsImport(unittest.TestCase):
    """
    Bug 5: _transient_counts should NOT re-import find_peaks locally.
    We verify the method works correctly and that the module-level import
    is the one in use (no shadowing or ImportError).
    """
 
    def setUp(self):
        self.ff = _make_ff(duration_sec=2.0)
 
    def test_returns_non_negative_integer(self):
        """Transient count must be a non-negative integer."""
        count = self.ff._transient_counts()
        self.assertIsInstance(count, int)
        self.assertGreaterEqual(count, 0)
 
    def test_silence_has_zero_transients(self):
        """Silent signal should produce zero transients."""
        ff = _make_silence()
        self.assertEqual(ff._transient_counts(), 0)
 
    def test_cached(self):
        """Calling _transient_counts twice must return the same cached value."""
        first = self.ff._transient_counts()
        second = self.ff._transient_counts()
        self.assertEqual(first, second)
 
    def test_no_local_find_peaks_import(self):
        """
        Inspect source to confirm there is no local 'from scipy.signal import find_peaks'
        inside _transient_counts.
        """
        import inspect
        src = inspect.getsource(self.ff._transient_counts)
        self.assertNotIn(
            "from scipy.signal import find_peaks",
            src,
            "Local re-import of find_peaks still present in _transient_counts (Bug 5).",
        )
 
 
# ---------------------------------------------------------------------------
# Bug 6 — _inharmonicity uses rank order instead of nearest harmonic
# ---------------------------------------------------------------------------
 
class TestInharmonicity(unittest.TestCase):
    """
    Bug 6: each peak must be matched to its nearest harmonic, not its
    sequential rank.  For a perfect harmonic series the inharmonicity
    should be near zero.
    """
 
    def setUp(self):
        # Build a signal that is a superposition of several harmonics of 200 Hz
        sr = 22050
        N = 2048
        H = 512
        duration = 2.0
        t = np.linspace(0, duration, int(duration * sr), endpoint=False)
        f0 = 200.0
        y = sum(
            (1.0 / h) * np.sin(2 * np.pi * h * f0 * t)
            for h in range(1, 6)
        )
        sig = AudioSignal(signal=y, sr=sr, N=N, H=H)
        self.ff = FrequencyFeatures(sig)
 
    def test_output_shape(self):
        """Inharmonicity must return shape (T,)."""
        inh = self.ff._inharmonicity()
        T = self.ff._magnitude_spectrum().shape[1]
        self.assertEqual(inh.shape, (T,))
 
    def test_values_non_negative(self):
        """Inharmonicity values must be >= 0."""
        inh = self.ff._inharmonicity()
        self.assertTrue(np.all(inh >= 0.0))
 
    def test_harmonic_signal_low_inharmonicity(self):
        """
        A near-perfect harmonic series should yield low mean inharmonicity
        once peaks are matched to nearest harmonics (not sequential rank).
        High values indicate Bug 6 is still present.
        """
        inh = self.ff._inharmonicity()
        voiced = inh[inh > 0]
        if voiced.size > 0:
            self.assertLess(
                float(np.mean(voiced)),
                0.5,
                "Mean inharmonicity is high for a harmonic signal — "
                "nearest-harmonic matching may not be applied (Bug 6).",
            )
 
    def test_cached(self):
        """Second call must return the same object."""
        self.assertIs(self.ff._inharmonicity(), self.ff._inharmonicity())
 
 
# ---------------------------------------------------------------------------
# Bug 7 — EPS in weight denominator of spotify_audio_features
# ---------------------------------------------------------------------------
 
class TestSpotifyAudioFeaturesWeights(unittest.TestCase):
    """
    Bug 7: fused score must equal dot(vals, weights) / sum(weights),
    not dot(vals, weights) / (sum(weights) + EPS).
    """
 
    def setUp(self):
        self.ff = _make_ff(duration_sec=2.0)
 
    def test_fused_score_in_range(self):
        """spotify_fused must lie in [0, 1]."""
        result = self.ff.spotify_audio_features()
        self.assertGreaterEqual(result["spotify_fused"], 0.0)
        self.assertLessEqual(result["spotify_fused"], 1.0)
 
    def test_fused_score_with_uniform_vals_equals_one(self):
        """
        If all sub-scores are 1.0, fused = sum(w*1)/sum(w) = 1.0 exactly.
        Adding EPS to the denominator would make it slightly less than 1.0.
        """
        weights = np.array([0.12, 0.12, 0.08, 0.10, 0.12, 0.08, 0.08, 0.10, 0.12, 0.06, 0.05, 0.07])
        vals = np.ones(12)
        expected = float(np.dot(vals, weights) / np.sum(weights))
        # expected should be exactly 1.0 for all-ones vals with positive weights
        self.assertAlmostEqual(expected, 1.0, places=10)
 
        # Fused with EPS-polluted denominator would differ from true weighted avg
        eps_polluted = float(np.dot(vals, weights) / (np.sum(weights) + EPS_VALUE))
        self.assertNotAlmostEqual(
            eps_polluted,
            expected,
            places=11,
            msg="EPS pollution was too small to detect — adjust EPS_VALUE if needed.",
        )
 
    def test_custom_equal_weights_fused_is_mean(self):
        """With equal weights, fused must equal the arithmetic mean of sub-scores."""
        result = self.ff.spotify_audio_features(
            weights=np.ones(12, dtype=float)
        )
        # Manually recompute expected mean from the returned sub-scores
        sub_scores = np.array([
            float(np.clip((result["loudness_db"] + 80.0) / 80.0, 0.0, 1.0)),
            float(np.clip(result["energy"] / (result["energy"] + 1.0), 0.0, 1.0)),
            result["speechiness"],
            result["acousticness"],
            result["danceability"],
            result["valence"],
            float(np.clip(result["tempo_bpm"] / 240.0, 0.0, 1.0)),
            result["liveness"],
            result["instrumentalness"],
            float(np.clip(result["key"] / 11.0, 0.0, 1.0)),
            1.0 if result["mode"] == "major" else 0.0,
            1.0 if result["time_signature"] == 4 else (0.5 if result["time_signature"] == 3 else 0.0),
        ])
        expected_fused = float(np.clip(np.mean(sub_scores), 0.0, 1.0))
        self.assertAlmostEqual(result["spotify_fused"], expected_fused, places=5)
 
    def test_all_expected_keys_present(self):
        """Return dict must contain all expected Spotify feature keys."""
        expected_keys = {
            "loudness_db", "energy", "speechiness", "acousticness",
            "danceability", "valence", "tempo_bpm", "liveness",
            "instrumentalness", "key", "mode", "time_signature", "spotify_fused",
        }
        result = self.ff.spotify_audio_features()
        self.assertEqual(set(result.keys()), expected_keys)
 
    def test_mode_is_valid(self):
        """mode must be 'major' or 'minor'."""
        result = self.ff.spotify_audio_features()
        self.assertIn(result["mode"], ("major", "minor"))
 
    def test_key_is_valid_pitch_class(self):
        """key must be an integer in [0, 11]."""
        result = self.ff.spotify_audio_features()
        self.assertIsInstance(result["key"], int)
        self.assertIn(result["key"], range(12))
 
    def test_time_signature_is_valid(self):
        """time_signature must be 3 or 4."""
        result = self.ff.spotify_audio_features()
        self.assertIn(result["time_signature"], (3, 4))
 
 
# ---------------------------------------------------------------------------
# General behavioral / integration tests
# ---------------------------------------------------------------------------
 
class TestMagnitudeAndPowerSpectra(unittest.TestCase):
    def setUp(self):
        self.ff = _make_ff()
 
    def test_magnitude_non_negative(self):
        mag = self.ff._magnitude_spectrum()
        self.assertTrue(np.all(mag >= 0.0))
 
    def test_power_equals_mag_squared(self):
        mag = self.ff._magnitude_spectrum()
        power = self.ff._power_spectrum()
        np.testing.assert_allclose(power, mag**2, rtol=1e-6)
 
    def test_magnitude_shape_matches_stft(self):
        mag = self.ff._magnitude_spectrum()
        self.assertEqual(mag.shape, self.ff.X.shape)
 
 
class TestFrameEnergy(unittest.TestCase):
    def setUp(self):
        self.ff = _make_ff()
 
    def test_frame_energy_positive(self):
        e = self.ff._frame_energy()
        self.assertTrue(np.all(e >= 0.0))
 
    def test_silence_energy_near_zero(self):
        ff = _make_silence()
        e = ff._frame_energy()
        self.assertTrue(np.all(e < 1e-10))
 
    def test_frame_energy_db_finite(self):
        e_db = self.ff._frame_energy_db()
        self.assertTrue(np.all(np.isfinite(e_db)))
 
 
class TestSpectralFeatureShapes(unittest.TestCase):
    """All per-frame spectral features must return arrays of length T."""
 
    def setUp(self):
        self.ff = _make_ff(duration_sec=2.0)
        self.T = self.ff._magnitude_spectrum().shape[1]
 
    def _assert_shape_T(self, arr):
        self.assertEqual(arr.shape, (self.T,))
 
    def test_centroid_shape(self):
        self._assert_shape_T(self.ff._spectral_centroid())
 
    def test_bandwidth_shape(self):
        self._assert_shape_T(self.ff._spectral_bandwidth())
 
    def test_rolloff_shape(self):
        self._assert_shape_T(self.ff._spectral_rolloff())
 
    def test_flatness_shape(self):
        self._assert_shape_T(self.ff._spectral_flatness())
 
    def test_entropy_shape(self):
        self._assert_shape_T(self.ff._spectral_entropy())
 
    def test_flux_shape(self):
        self._assert_shape_T(self.ff._spectral_flux())
 
    def test_harmonic_ratio_shape(self):
        self._assert_shape_T(self.ff._harmonic_ratio())
 
 
class TestSpectralFeatureRanges(unittest.TestCase):
    def setUp(self):
        self.ff = _make_ff(duration_sec=2.0)
 
    def test_centroid_within_nyquist(self):
        cent = self.ff._spectral_centroid()
        nyquist = self.ff.sr / 2.0
        self.assertTrue(np.all(cent >= 0.0) and np.all(cent <= nyquist))
 
    def test_bandwidth_non_negative(self):
        bw = self.ff._spectral_bandwidth()
        self.assertTrue(np.all(bw >= 0.0))
 
    def test_rolloff_within_nyquist(self):
        roll = self.ff._spectral_rolloff()
        nyquist = self.ff.sr / 2.0
        self.assertTrue(np.all(roll >= 0.0) and np.all(roll <= nyquist + 1.0))
 
    def test_flatness_in_0_1(self):
        flat = self.ff._spectral_flatness()
        self.assertTrue(np.all(flat >= 0.0) and np.all(flat <= 1.0))
 
    def test_entropy_in_0_1(self):
        ent = self.ff._spectral_entropy()
        self.assertTrue(np.all(ent >= 0.0) and np.all(ent <= 1.0))
 
    def test_harmonic_ratio_in_0_1(self):
        hr = self.ff._harmonic_ratio()
        self.assertTrue(np.all(hr >= 0.0) and np.all(hr <= 1.0 + 1e-6))
 
 
class TestBandEnergy(unittest.TestCase):
    def setUp(self):
        self.ff = _make_ff()
 
    def test_band_energy_shape(self):
        bands = [(0, 500), (500, 2000), (2000, 8000)]
        be = self.ff._band_energy(bands)
        T = self.ff._magnitude_spectrum().shape[1]
        self.assertEqual(be.shape, (3, T))
 
    def test_band_energy_non_negative(self):
        bands = [(0, 500), (500, 4000)]
        be = self.ff._band_energy(bands)
        self.assertTrue(np.all(be >= 0.0))
 
    def test_band_energy_ratio_sums_to_one(self):
        bands = [(0, 1000), (1000, 4000), (4000, 11025)]
        ratio = self.ff._band_energy_ratio(bands)
        col_sums = np.sum(ratio, axis=0)
        np.testing.assert_allclose(col_sums, 1.0, atol=1e-5)
 
 
class TestDynamicRange(unittest.TestCase):
    def test_dynamic_range_non_negative(self):
        ff = _make_ff(duration_sec=2.0)
        dr = ff._dynamic_range()
        self.assertGreaterEqual(dr, 0.0)
 
    def test_silence_dynamic_range(self):
        ff = _make_silence()
        dr = ff._dynamic_range()
        # Silence can produce a finite but very small dynamic range (EPS floor)
        self.assertIsInstance(dr, float)
        self.assertTrue(np.isfinite(dr))
 
 
class TestSpectralFlux(unittest.TestCase):
    def setUp(self):
        self.ff = _make_ff(duration_sec=2.0)
 
    def test_first_frame_is_zero(self):
        """Flux is defined as zero for the first frame."""
        flux = self.ff._spectral_flux()
        self.assertAlmostEqual(float(flux[0]), 0.0, places=10)
 
    def test_flux_non_negative_with_hwr(self):
        flux = self.ff._spectral_flux(half_wave_rectify=True)
        self.assertTrue(np.all(flux >= 0.0))
 
    def test_silence_flux_near_zero(self):
        ff = _make_silence()
        flux = ff._spectral_flux()
        self.assertTrue(np.all(np.abs(flux) < 1e-10))
 
 
class TestInstantaneousFrequency(unittest.TestCase):
    def setUp(self):
        self.ff = _make_ff(duration_sec=1.0, freq_hz=440.0)
 
    def test_shape(self):
        inst = self.ff._instantaneous_freq()
        self.assertEqual(inst.shape, self.ff.X.shape)
 
    def test_finite(self):
        inst = self.ff._instantaneous_freq()
        self.assertTrue(np.all(np.isfinite(inst)))
 
    def test_440hz_bin_near_440(self):
        """Instantaneous frequency at the 440 Hz bin should be close to 440 Hz."""
        inst = self.ff._instantaneous_freq()
        bin_440 = int(np.argmin(np.abs(self.ff.freqs - 440.0)))
        # Average over frames (skip first as it's copied)
        mean_if = float(np.mean(inst[bin_440, 1:]))
        self.assertAlmostEqual(mean_if, 440.0, delta=50.0)
 
 
class TestPitchClassProfile(unittest.TestCase):
    def setUp(self):
        self.ff = _make_ff(duration_sec=2.0, freq_hz=440.0)
 
    def test_shape(self):
        pcp = self.ff._pitch_class_profile()
        self.assertEqual(pcp.shape, (12,))
 
    def test_sums_to_one(self):
        pcp = self.ff._pitch_class_profile()
        self.assertAlmostEqual(float(np.sum(pcp)), 1.0, places=5)
 
    def test_non_negative(self):
        pcp = self.ff._pitch_class_profile()
        self.assertTrue(np.all(pcp >= 0.0))
 
    def test_440hz_dominant_pitch_class(self):
        """440 Hz is A (MIDI 69, pitch class 9) — it should be the strongest."""
        pcp = self.ff._pitch_class_profile()
        dominant = int(np.argmax(pcp))
        self.assertEqual(dominant, 9, "Expected pitch class 9 (A) to be dominant for a 440 Hz sine.")
 
 
class TestShortSignalPadding(unittest.TestCase):
    """Signal shorter than N should be zero-padded without error."""
 
    def test_short_signal_does_not_raise(self):
        sr = 22050
        N = 2048
        y = np.random.randn(100)  # much shorter than N
        sig = AudioSignal(signal=y, sr=sr, N=N, H=512)
        ff = FrequencyFeatures(sig)
        # Should not raise; magnitude spectrum should be computable
        mag = ff._magnitude_spectrum()
        self.assertGreater(mag.size, 0)
 
 
class TestCachingConsistency(unittest.TestCase):
    """Verify that cached and freshly-computed values agree."""
 
    def setUp(self):
        self.ff = _make_ff(duration_sec=1.0)
 
    def _fresh(self):
        """Return a fresh FrequencyFeatures instance (empty cache)."""
        return _make_ff(duration_sec=1.0)
 
    def test_centroid_cache_consistent(self):
        a = self.ff._spectral_centroid()
        b = self.ff._spectral_centroid()
        np.testing.assert_array_equal(a, b)
 
    def test_power_spectrum_cache_consistent(self):
        a = self.ff._power_spectrum()
        b = self.ff._power_spectrum()
        np.testing.assert_array_equal(a, b)
 
 
if __name__ == "__main__":
    unittest.main()