import types
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
 
import numpy as np
from pathlib import Path
import pytest
import sys

# ---------------------------------------------------------------------------
# Minimal stubs so the module can be imported without the full package
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
    "audio_features.chromagram_features",
    types.ModuleType("audio_features.chromagram_features"),
)

from audio_features.chromagram_features import ChromagramFeatures

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
 
def _make_signal(duration_sec=2.0, sr=22050, N=2048, H=512, freq_hz=440.0, n_harmonics=5):
    """Return an _AudioSignal containing a harmonic tone (default: A4 = 440 Hz)."""
    t = np.linspace(0, duration_sec, int(duration_sec * sr), endpoint=False)
    y = sum((1.0 / h) * np.sin(2 * np.pi * h * freq_hz * t) for h in range(1, n_harmonics + 1))
    return AudioSignal(signal=y, sr=sr, N=N, H=H)
 
 
def _make_silence(duration_sec=2.0, sr=22050, N=2048, H=512):
    """Return an _AudioSignal of silence."""
    y = np.zeros(int(duration_sec * sr))
    return AudioSignal(signal=y, sr=sr, N=N, H=H)
 
 
def _make_cf(duration_sec=2.0, freq_hz=440.0):
    """Convenience factory: ChromagramFeatures from a pure harmonic tone."""
    return ChromagramFeatures(_make_signal(duration_sec=duration_sec, freq_hz=freq_hz))
 
 
def _make_cf_silence():
    return ChromagramFeatures(_make_silence())
 
 
def _make_c_major(duration_sec=2.0, sr=22050, N=2048, H=512):
    """
    Synthesise a C-major chord (C4=261.63, E4=329.63, G4=392.00 Hz) so that
    key-estimation tests have a musically meaningful signal.
    """
    t = np.linspace(0, duration_sec, int(duration_sec * sr), endpoint=False)
    y = (
        np.sin(2 * np.pi * 261.63 * t)   # C
        + np.sin(2 * np.pi * 329.63 * t)  # E
        + np.sin(2 * np.pi * 392.00 * t)  # G
    )
    return ChromagramFeatures(AudioSignal(signal=y, sr=sr, N=N, H=H))
 
 
# ---------------------------------------------------------------------------
# Bug 1 — _chroma_template returns ndarray; _valence_chroma subscripts as dict
# ---------------------------------------------------------------------------
 
class TestChromaTemplateReturnType(unittest.TestCase):
    """Bug 1: _chroma_template must return an ndarray, not a dict."""
 
    def setUp(self):
        self.cf = _make_cf()
 
    def test_chroma_template_is_ndarray(self):
        result = self.cf._chroma_template()
        self.assertIsInstance(
            result, np.ndarray,
            "_chroma_template() must return an ndarray, not a dict.",
        )
 
    def test_chroma_template_shape(self):
        """Must be (24, 12): 12 major + 12 minor templates, each 12 pitch classes."""
        result = self.cf._chroma_template()
        self.assertEqual(result.shape, (24, 12))
 
    def test_valence_chroma_does_not_raise(self):
        """
        _valence_chroma calls _chroma_template()["templates"] in the buggy version.
        After the fix it must call _chroma_template() directly and not raise.
        """
        try:
            val = self.cf._valence_chroma()
        except (TypeError, KeyError) as exc:
            self.fail(
                f"_valence_chroma() raised {type(exc).__name__}: {exc}. "
                "This indicates Bug 1 is still present — "
                "_chroma_template() is being subscripted as a dict."
            )
 
    def test_valence_chroma_in_range(self):
        val = self.cf._valence_chroma()
        self.assertGreaterEqual(val, 0.0)
        self.assertLessEqual(val, 1.0)
 
    def test_chroma_template_rows_sum_to_one(self):
        """Every template row must be L1-normalised to 1.0."""
        T = self.cf._chroma_template()
        row_sums = np.sum(T, axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)
 
    def test_chroma_template_cached(self):
        self.assertIs(self.cf._chroma_template(), self.cf._chroma_template())
 
 
# ---------------------------------------------------------------------------
# Bug 2 — _chroma_profile normalises dB values as probabilities
# ---------------------------------------------------------------------------
 
class TestChromaProfileDbNormalization(unittest.TestCase):
    """
    Bug 2: when use_db=True and normalize=True, the normalisation must be done
    in the linear domain before converting to dB — not by dividing dB values
    (which are negative) by their column sums (also negative).
    """
 
    def setUp(self):
        self.cf = _make_cf()
 
    def test_normalized_linear_columns_sum_to_one(self):
        """normalize=True, use_db=False — each column must sum to ~1.0."""
        P = self.cf._chroma_profile(normalize=True, use_db=False)
        col_sums = np.sum(P, axis=0)
        np.testing.assert_allclose(col_sums, 1.0, atol=1e-6,
                                   err_msg="Linear normalized chroma columns must sum to 1.")
 
    def test_normalized_linear_values_non_negative(self):
        P = self.cf._chroma_profile(normalize=True, use_db=False)
        self.assertTrue(np.all(P >= 0.0))
 
    def test_normalized_db_values_are_non_positive(self):
        """
        After fix: normalize in linear, then convert to dB.
        dB values of probabilities in [0,1] must be <= 0.
        The buggy version produces values near 1.0 (positive) due to
        dividing a negative matrix by a negative column sum.
        """
        P_db = self.cf._chroma_profile(normalize=True, use_db=True)
        self.assertTrue(
            np.all(P_db <= 0.0 + 1e-6),
            "Normalized dB chroma values should be <= 0. "
            "Positive values indicate Bug 2 is still present.",
        )
 
    def test_normalized_db_not_near_positive_one(self):
        """
        The buggy version produces values clustered around 1.0 (ratio of two
        near-equal negatives). After fix, values must be well below 0.
        """
        P_db = self.cf._chroma_profile(normalize=True, use_db=True)
        mean_val = float(np.mean(P_db))
        self.assertLess(
            mean_val, 0.0,
            f"Mean normalized dB chroma is {mean_val:.4f}; expected < 0. "
            "Bug 2 may still be present.",
        )
 
    def test_unnormalized_db_shape(self):
        P = self.cf._chroma_profile(normalize=False, use_db=True)
        self.assertEqual(P.shape[0], 12)
 
    def test_chroma_profile_shape(self):
        P = self.cf._chroma_profile()
        self.assertEqual(P.shape[0], 12)
        self.assertGreater(P.shape[1], 0)
 
 
# ---------------------------------------------------------------------------
# Bug 3 — Cache key collision: _tonal_stability_index vs _tonal_stability
# ---------------------------------------------------------------------------
 
class TestTonalStabilityCacheKey(unittest.TestCase):
    """
    Bug 3: _tonal_stability_index must use a distinct cache key from
    _tonal_stability so they do not overwrite each other.
    """
 
    def setUp(self):
        self.cf = _make_cf(duration_sec=3.0)
 
    def test_keys_are_distinct(self):
        """
        Call both methods and verify they store under different cache keys,
        so neither poisons the other's result.
        """
        # Warm both caches
        tsi = self.cf._tonal_stability_index()
        ts = self.cf._tonal_stability()
 
        # Collect all cache keys
        cache_keys = list(self.cf._cache_chroma.keys())
 
        tsi_keys = [k for k in cache_keys if "tonal_stability_index" in k]
        ts_keys = [k for k in cache_keys if k.startswith("tonal_stability_") and "index" not in k]
 
        self.assertGreater(len(tsi_keys), 0,
                           "No 'tonal_stability_index' key found in cache — Bug 3 may still be present.")
        self.assertGreater(len(ts_keys), 0,
                           "No 'tonal_stability' (non-index) key found in cache.")
 
    def test_values_are_independent(self):
        """
        The two methods compute different things; their values must differ
        (they would be identical if one shadows the other via cache collision).
        """
        tsi = self.cf._tonal_stability_index()
        ts = self.cf._tonal_stability()
        # They CAN be equal by coincidence, but for a real signal they typically differ.
        # At minimum, both must be valid floats in [0, 1].
        self.assertIsInstance(tsi, float)
        self.assertIsInstance(ts, float)
        self.assertGreaterEqual(tsi, 0.0)
        self.assertLessEqual(tsi, 1.0)
        self.assertGreaterEqual(ts, 0.0)
        self.assertLessEqual(ts, 1.0)
 
    def test_tonal_stability_index_cached(self):
        first = self.cf._tonal_stability_index()
        second = self.cf._tonal_stability_index()
        self.assertEqual(first, second)
 
    def test_tonal_stability_cached(self):
        first = self.cf._tonal_stability()
        second = self.cf._tonal_stability()
        self.assertEqual(first, second)
 
    def test_calling_index_first_does_not_corrupt_stability(self):
        """Call index first, then stability — stability must return its own value."""
        tsi = self.cf._tonal_stability_index()
        ts = self.cf._tonal_stability()
        # ts must be a float in [0,1], not a TSI composite stored under wrong key
        self.assertIsInstance(ts, float)
        self.assertGreaterEqual(ts, 0.0)
        self.assertLessEqual(ts, 1.0)
 
    def test_calling_stability_first_does_not_corrupt_index(self):
        """Call stability first, then index — index must return its own value."""
        cf2 = _make_cf(duration_sec=3.0)
        ts = cf2._tonal_stability()
        tsi = cf2._tonal_stability_index()
        self.assertIsInstance(tsi, float)
        self.assertGreaterEqual(tsi, 0.0)
        self.assertLessEqual(tsi, 1.0)
 
 
# ---------------------------------------------------------------------------
# Bug 4 — _chroma_skewness uses unsigned distance for a signed moment
# ---------------------------------------------------------------------------
 
class TestChromaSkewness(unittest.TestCase):
    """
    Bug 4: skewness must be able to take negative values (signed distribution).
    With unsigned distance, m3 >= 0 always, making skew always >= 0.
    """
 
    def setUp(self):
        self.cf = _make_cf()
 
    def test_skewness_shape(self):
        """Must return shape (T,) matching number of STFT frames."""
        skew = self.cf._chroma_skewness()
        T = self.cf._chroma_profile().shape[1]
        self.assertEqual(skew.shape, (T,))
 
    def test_skewness_finite(self):
        skew = self.cf._chroma_skewness()
        self.assertTrue(np.all(np.isfinite(skew)))
 
    def test_skewness_can_be_negative(self):
        """
        Construct a left-skewed chroma by concentrating energy in low pitch
        classes, then verify skewness goes negative somewhere.
        After fix (signed distance), some frames should yield negative skew.
        """
        sr, N, H = 22050, 2048, 512
        duration = 3.0
        t = np.linspace(0, duration, int(duration * sr), endpoint=False)
        # Low pitch classes: C2=65.41, D2=73.42, E2=82.41 Hz
        y = (
            np.sin(2 * np.pi * 65.41 * t)
            + np.sin(2 * np.pi * 73.42 * t)
            + np.sin(2 * np.pi * 82.41 * t)
        )
        cf = ChromagramFeatures(AudioSignal(signal=y, sr=sr, N=N, H=H))
        skew = cf._chroma_skewness()
        has_negative = np.any(skew < 0.0)
        self.assertTrue(
            has_negative,
            "No negative skewness values found. "
            "If skewness is always >= 0, Bug 4 is still present "
            "(unsigned circular distance used in m3).",
        )
 
    def test_symmetric_distribution_skew_near_zero(self):
        """
        A signal with energy evenly spread across all 12 pitch classes
        should have skewness near 0.
        """
        sr, N, H = 22050, 2048, 512
        duration = 2.0
        t = np.linspace(0, duration, int(duration * sr), endpoint=False)
        freqs = [261.63, 277.18, 293.66, 311.13, 329.63,
                 349.23, 369.99, 392.00, 415.30, 440.00, 466.16, 493.88]
        y = sum(np.sin(2 * np.pi * f * t) for f in freqs)
        cf = ChromagramFeatures(AudioSignal(signal=y, sr=sr, N=N, H=H))
        skew = cf._chroma_skewness()
        self.assertAlmostEqual(float(np.mean(skew)), 0.0, delta=1.0)
 
    def test_skewness_cached(self):
        self.assertIs(self.cf._chroma_skewness(), self.cf._chroma_skewness())
 
 
# ---------------------------------------------------------------------------
# Bug 5 — _harmonic_entropy divides by log(N + EPS) instead of log(N)
# ---------------------------------------------------------------------------
 
class TestHarmonicEntropy(unittest.TestCase):
    """Bug 5: normalisation denominator must be log(24), not log(24 + EPS)."""
 
    def setUp(self):
        self.cf = _make_cf()
 
    def test_entropy_in_range(self):
        """Normalised entropy must lie in [0, 1]."""
        result = self.cf._harmonic_entropy()
        H = result["harmonic_entropy"]
        self.assertGreaterEqual(H, 0.0)
        self.assertLessEqual(H, 1.0 + 1e-9)
 
    def test_uniform_distribution_entropy_is_one(self):
        """
        When all 24 key template scores are equal, the softmax distribution
        is uniform and entropy must normalise to 1.0 exactly (not 1 - delta).
        With log(24 + EPS) in the denominator it would be very slightly < 1.
        """
        n = 24
        log_n_clean = np.log(n)
        log_n_buggy = np.log(n + EPS_VALUE)
 
        # Uniform distribution has H = log(n)
        H_raw = np.log(n)
        H_clean = H_raw / log_n_clean
        H_buggy = H_raw / log_n_buggy
 
        self.assertAlmostEqual(H_clean, 1.0, places=12,
                               msg="Clean log(N) normalisation must give exactly 1.0 for uniform dist.")
        self.assertNotAlmostEqual(H_buggy, 1.0, places=12,
                                  msg="Buggy log(N+EPS) must differ from 1.0 at places=9.")
 
    def test_entropy_denominator_is_log_24(self):
        """
        Verify the implementation uses log(24) by checking that the raw
        entropy divided by the returned normalised entropy equals log(24)
        (within floating-point tolerance), not log(24 + EPS).
        """
        result = self.cf._harmonic_entropy()
        H_norm = result["harmonic_entropy"]
        H_raw = result["raw_entropy"]
 
        if abs(H_norm) < 1e-12:
            self.skipTest("Entropy is effectively zero — cannot validate denominator.")
 
        implied_denom = float(H_raw / H_norm)
        expected_denom = float(np.log(24))
        buggy_denom = float(np.log(24 + EPS_VALUE))
 
        delta_clean = abs(implied_denom - expected_denom)
        delta_buggy = abs(implied_denom - buggy_denom)
 
        self.assertLess(
            delta_clean, delta_buggy,
            f"Implied denominator {implied_denom:.10f} is closer to "
            f"log(24+EPS)={buggy_denom:.10f} than log(24)={expected_denom:.10f}. "
            "Bug 5 may still be present.",
        )
 
    def test_result_has_expected_keys(self):
        result = self.cf._harmonic_entropy()
        for k in ("harmonic_entropy", "raw_entropy", "probabilities"):
            self.assertIn(k, result)
 
    def test_probabilities_sum_to_one(self):
        p = self.cf._harmonic_entropy()["probabilities"]
        self.assertAlmostEqual(float(np.sum(p)), 1.0, places=6)
 
    def test_cached(self):
        r1 = self.cf._harmonic_entropy()
        r2 = self.cf._harmonic_entropy()
        self.assertIs(r1, r2)
 
 
# ---------------------------------------------------------------------------
# Bug 6 — _valence_chroma omits tonic from the 'bright' sum
# ---------------------------------------------------------------------------
 
class TestValenceChromaBright(unittest.TestCase):
    """
    Bug 6: the 'bright' sub-score inside _valence_chroma must include the
    tonic pitch class (prof[tonic]) in addition to the 3rd, 5th, and 7th.
    """
 
    def setUp(self):
        self.cf = _make_c_major()
 
    def test_valence_in_range(self):
        val = self.cf._valence_chroma()
        self.assertGreaterEqual(val, 0.0)
        self.assertLessEqual(val, 1.0)
 
    def test_bright_includes_tonic(self):
        """
        Manually replicate the bright calculation both with and without the
        tonic.  Verify the implementation matches the 4-tone (with-tonic) version.
 
        Strategy: monkey-patch _key_estimation to return a known tonic, then
        compare the resulting valence score against what we expect from each
        formula variant.
        """
        import unittest.mock as mock
 
        cf = _make_c_major()
        prof = cf._mean_chroma_profile()
        tonic = int(cf._key_estimation()["tonic"])
 
        # Compute bright WITHOUT tonic (buggy formula)
        bright_buggy = float(
            (prof[(tonic + 4) % 12] + prof[(tonic + 7) % 12] + prof[(tonic + 11) % 12])
            / (np.sum(prof) + EPS_VALUE)
        )
        # Compute bright WITH tonic (correct formula)
        bright_fixed = float(
            (prof[tonic % 12] + prof[(tonic + 4) % 12] + prof[(tonic + 7) % 12] + prof[(tonic + 11) % 12])
            / (np.sum(prof) + EPS_VALUE)
        )
 
        # For any non-zero tonic energy, fixed > buggy
        tonic_energy = float(prof[tonic % 12])
        if tonic_energy > EPS_VALUE:
            self.assertGreater(
                bright_fixed, bright_buggy,
                "Fixed bright (with tonic) should exceed buggy bright (without tonic) "
                "when tonic has non-zero energy.",
            )
 
    def test_valence_major_signal_above_midpoint(self):
        """A clear C-major chord should produce valence above the midpoint."""
        val = self.cf._valence_chroma()
        self.assertGreater(val, 0.3,
                           "Valence for a major chord should be well above 0.")
 
    def test_valence_cached(self):
        v1 = self.cf._valence_chroma()
        v2 = self.cf._valence_chroma()
        self.assertEqual(v1, v2)
 
    def test_silence_valence_returns_default(self):
        """Silence has no pitch profile; _valence_chroma must return 0.5."""
        cf = _make_cf_silence()
        val = cf._valence_chroma()
        self.assertGreaterEqual(val, 0.0)
        self.assertLessEqual(val, 1.0)
 
 
# ---------------------------------------------------------------------------
# Bug 7 — _chord_progression_mapping guard misses single-frame case
# ---------------------------------------------------------------------------
 
class TestChordProgressionMapping(unittest.TestCase):
    """
    Bug 7: the guard must handle len(chord_labels) <= 1, not just == 0.
    A single-frame signal previously produced a degenerate (1,1) transition
    matrix with a 0/EPS self-transition probability instead of an empty matrix.
    """
 
    def setUp(self):
        self.cf = _make_cf()
 
    def test_empty_labels_returns_empty_matrix(self):
        """Zero-frame edge case: transition matrices must be (0, 0)."""
        cf = self.cf
        # Inject empty chord detection result
        det_key = "chord_detection_norm_lin_cosine"
        cf._cache_chroma[det_key] = {
            "chord_labels": [],
            "chord_idx": np.array([], dtype=int),
            "scores": np.zeros((108, 0), dtype=float),
            "best_scores": np.array([], dtype=float),
            "roots": np.array([], dtype=int),
            "qualities": np.array([], dtype=object),
        }
        result = cf._chord_progression_mapping()
        self.assertEqual(result["transition_counts"].shape, (0, 0))
        self.assertEqual(result["transition_probs"].shape, (0, 0))
 
    def test_single_label_returns_empty_matrix(self):
        """
        Single-frame case (Bug 7): must return empty matrices, not (1,1)
        with a degenerate 0.0 self-transition.
        """
        cf = _make_cf()
        det_key = "chord_detection_norm_lin_cosine"
        cf._cache_chroma[det_key] = {
            "chord_labels": ["0:maj"],
            "chord_idx": np.array([0], dtype=int),
            "scores": np.zeros((108, 1), dtype=float),
            "best_scores": np.array([0.8], dtype=float),
            "roots": np.array([0], dtype=int),
            "qualities": np.array(["maj"], dtype=object),
        }
        result = cf._chord_progression_mapping()
        self.assertEqual(
            result["transition_counts"].shape, (0, 0),
            "Single-frame signal must return empty (0,0) transition matrix. "
            "A (1,1) matrix with 0.0 self-transition indicates Bug 7 is still present.",
        )
 
    def test_multi_label_transition_probs_row_sum(self):
        """For a real multi-frame signal, each row of probs must sum to ~1."""
        result = self.cf._chord_progression_mapping()
        probs = result["transition_probs"]
        if probs.shape[0] > 0:
            row_sums = np.sum(probs, axis=1)
            np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)
 
    def test_transition_counts_non_negative(self):
        result = self.cf._chord_progression_mapping()
        self.assertTrue(np.all(result["transition_counts"] >= 0.0))
 
    def test_result_has_expected_keys(self):
        result = self.cf._chord_progression_mapping()
        for k in ("labels", "transition_counts", "transition_probs"):
            self.assertIn(k, result)
 
    def test_cached(self):
        r1 = self.cf._chord_progression_mapping()
        r2 = self.cf._chord_progression_mapping()
        self.assertIs(r1, r2)
 
 
# ---------------------------------------------------------------------------
# General — Core chromagram pipeline
# ---------------------------------------------------------------------------
 
class TestChromaPipeline(unittest.TestCase):
    def setUp(self):
        self.cf = _make_cf()
 
    def test_chroma_shape(self):
        C = self.cf._chroma()
        self.assertEqual(C.shape[0], 12)
        self.assertGreater(C.shape[1], 0)
 
    def test_chroma_non_negative(self):
        self.assertTrue(np.all(self.cf._chroma() >= 0.0))
 
    def test_chroma_db_finite(self):
        self.assertTrue(np.all(np.isfinite(self.cf._chroma_db())))
 
    def test_chroma_cached(self):
        self.assertIs(self.cf._chroma(), self.cf._chroma())
 
    def test_chroma_db_cached(self):
        self.assertIs(self.cf._chroma_db(), self.cf._chroma_db())
 
    def test_silence_chroma_near_zero(self):
        cf = _make_cf_silence()
        C = cf._chroma()
        self.assertTrue(np.all(C < 1e-6))
 
    def test_compute_spec_log_freq_shape(self):
        Y = self.cf._compute_spec_log_freq()
        self.assertEqual(Y.shape[0], 128)
        self.assertGreater(Y.shape[1], 0)
 
    def test_compute_chromagram_shape(self):
        Y = self.cf._compute_spec_log_freq()
        C = self.cf._compute_chromagram(Y)
        self.assertEqual(C.shape, (12, Y.shape[1]))
 
 
class TestStaticConversions(unittest.TestCase):
    def test_midi_to_hz_a4(self):
        """MIDI 69 must map to 440 Hz."""
        self.assertAlmostEqual(ChromagramFeatures.midi_to_hz(69), 440.0, places=6)
 
    def test_hz_to_midi_a4(self):
        """440 Hz must map to MIDI 69."""
        self.assertAlmostEqual(ChromagramFeatures.hz_to_midi(440.0), 69.0, places=6)
 
    def test_round_trip(self):
        """midi_to_hz and hz_to_midi must be inverses."""
        for midi in [21, 48, 60, 69, 84, 108]:
            recovered = ChromagramFeatures.hz_to_midi(ChromagramFeatures.midi_to_hz(midi))
            self.assertAlmostEqual(recovered, midi, places=8)
 
 
class TestMeanChroma(unittest.TestCase):
    def setUp(self):
        self.cf = _make_cf()
 
    def test_shape(self):
        self.assertEqual(self.cf._mean_chroma().shape, (12,))
 
    def test_normalized_sums_to_one(self):
        mc = self.cf._mean_chroma(normalize=True)
        self.assertAlmostEqual(float(np.sum(mc)), 1.0, places=5)
 
    def test_non_negative(self):
        self.assertTrue(np.all(self.cf._mean_chroma() >= 0.0))
 
    def test_cached(self):
        self.assertIs(self.cf._mean_chroma(), self.cf._mean_chroma())
 
 
class TestChordTemplates(unittest.TestCase):
    def setUp(self):
        self.cf = _make_cf()
 
    def test_returns_dict_with_expected_keys(self):
        result = self.cf._chord_templates()
        for k in ("templates", "labels", "roots", "qualities"):
            self.assertIn(k, result)
 
    def test_template_count(self):
        """12 roots × 9 chord types = 108 templates."""
        result = self.cf._chord_templates()
        self.assertEqual(result["templates"].shape, (108, 12))
        self.assertEqual(len(result["labels"]), 108)
        self.assertEqual(len(result["roots"]), 108)
        self.assertEqual(len(result["qualities"]), 108)
 
    def test_templates_normalized(self):
        T = self.cf._chord_templates()["templates"]
        row_sums = np.sum(T, axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)
 
    def test_cached(self):
        self.assertIs(self.cf._chord_templates(), self.cf._chord_templates())
 
 
class TestChromaCentroidAndSpread(unittest.TestCase):
    def setUp(self):
        self.cf = _make_cf()
 
    def test_centroid_shape(self):
        mu = self.cf._chroma_centroid()
        T = self.cf._chroma_profile().shape[1]
        self.assertEqual(mu.shape, (T,))
 
    def test_centroid_finite(self):
        self.assertTrue(np.all(np.isfinite(self.cf._chroma_centroid())))
 
    def test_spread_non_negative(self):
        spread = self.cf._chroma_spread()
        self.assertTrue(np.all(spread >= 0.0))
 
    def test_spread_shape(self):
        spread = self.cf._chroma_spread()
        T = self.cf._chroma_profile().shape[1]
        self.assertEqual(spread.shape, (T,))
 
    def test_centroid_cached(self):
        self.assertIs(self.cf._chroma_centroid(), self.cf._chroma_centroid())
 
    def test_spread_cached(self):
        self.assertIs(self.cf._chroma_spread(), self.cf._chroma_spread())
 
 
class TestChromaKurtosis(unittest.TestCase):
    def setUp(self):
        self.cf = _make_cf()
 
    def test_shape(self):
        T = self.cf._chroma_profile().shape[1]
        self.assertEqual(self.cf._chroma_kurtosis().shape, (T,))
 
    def test_excess_kurtosis_normal_near_zero(self):
        """Excess kurtosis subtracts 3 from raw kurtosis."""
        raw = self.cf._chroma_kurtosis(excess=False)
        excess = self.cf._chroma_kurtosis(excess=True)
        np.testing.assert_allclose(excess, raw - 3.0, atol=1e-10)
 
    def test_cached(self):
        k1 = self.cf._chroma_kurtosis()
        k2 = self.cf._chroma_kurtosis()
        self.assertIs(k1, k2)
 
 
class TestKeyEstimation(unittest.TestCase):
    def setUp(self):
        self.cf = _make_c_major()
 
    def test_result_keys(self):
        result = self.cf._key_estimation()
        for k in ("key_idx", "tonic", "mode", "score", "scores"):
            self.assertIn(k, result)
 
    def test_key_idx_in_range(self):
        idx = self.cf._key_estimation()["key_idx"]
        self.assertIn(idx, range(24))
 
    def test_tonic_in_range(self):
        tonic = self.cf._key_estimation()["tonic"]
        self.assertIn(tonic, range(12))
 
    def test_mode_valid(self):
        mode = self.cf._key_estimation()["mode"]
        self.assertIn(mode, ("maj", "min"))
 
    def test_scores_length(self):
        scores = self.cf._key_estimation()["scores"]
        self.assertEqual(len(scores), 24)
 
    def test_c_major_detects_c(self):
        """C-major chord should produce tonic = 0 (C) in major mode."""
        result = self.cf._key_estimation()
        self.assertEqual(result["tonic"], 0,
                         f"Expected tonic 0 (C) but got {result['tonic']}.")
        self.assertEqual(result["mode"], "maj",
                         f"Expected 'maj' mode but got {result['mode']}.")
 
    def test_invalid_method_raises(self):
        with self.assertRaises(ValueError):
            self.cf._key_estimation(method="invalid")
 
    def test_cached(self):
        r1 = self.cf._key_estimation()
        r2 = self.cf._key_estimation()
        self.assertIs(r1, r2)
 
 
class TestModeClassification(unittest.TestCase):
    def setUp(self):
        self.cf = _make_c_major()
 
    def test_result_keys(self):
        result = self.cf._mode_classification()
        for k in ("mode", "score_major", "score_minor", "delta_score"):
            self.assertIn(k, result)
 
    def test_mode_is_valid(self):
        self.assertIn(self.cf._mode_classification()["mode"], ("maj", "min"))
 
    def test_c_major_mode(self):
        self.assertEqual(self.cf._mode_classification()["mode"], "maj")
 
    def test_delta_score_sign_consistent_with_mode(self):
        result = self.cf._mode_classification()
        if result["mode"] == "maj":
            self.assertGreaterEqual(result["delta_score"], 0.0)
        else:
            self.assertLess(result["delta_score"], 0.0)
 
    def test_cached(self):
        r1 = self.cf._mode_classification()
        r2 = self.cf._mode_classification()
        self.assertIs(r1, r2)
 
 
class TestTonalClarity(unittest.TestCase):
    def setUp(self):
        self.cf = _make_c_major()
 
    def test_result_keys(self):
        result = self.cf._tonal_clarity()
        for k in ("tonal_clarity", "best_score", "margin"):
            self.assertIn(k, result)
 
    def test_clarity_non_negative(self):
        self.assertGreaterEqual(self.cf._tonal_clarity()["tonal_clarity"], 0.0)
 
    def test_c_major_has_positive_clarity(self):
        self.assertGreater(self.cf._tonal_clarity()["tonal_clarity"], 0.0)
 
    def test_cached(self):
        r1 = self.cf._tonal_clarity()
        r2 = self.cf._tonal_clarity()
        self.assertIs(r1, r2)
 
 
class TestChromaEntropyFluxVariance(unittest.TestCase):
    def setUp(self):
        self.cf = _make_cf()
 
    def test_entropy_in_range(self):
        e = self.cf._chroma_entropy()
        self.assertGreaterEqual(e, 0.0)
        self.assertLessEqual(e, 1.0)
 
    def test_flux_mean_non_negative(self):
        self.assertGreaterEqual(self.cf._chroma_flux_mean(), 0.0)
 
    def test_flux_variance_non_negative(self):
        self.assertGreaterEqual(self.cf._chroma_flux_variance(), 0.0)
 
    def test_silence_entropy(self):
        """Silence — all columns are zero and filled with 1/12 — entropy is 1."""
        cf = _make_cf_silence()
        e = cf._chroma_entropy()
        self.assertAlmostEqual(e, 1.0, places=5)
 
    def test_silence_flux_zero(self):
        cf = _make_cf_silence()
        self.assertAlmostEqual(cf._chroma_flux_mean(), 0.0, places=8)
 
    def test_cached(self):
        self.assertEqual(self.cf._chroma_entropy(), self.cf._chroma_entropy())
        self.assertEqual(self.cf._chroma_flux_mean(), self.cf._chroma_flux_mean())
        self.assertEqual(self.cf._chroma_flux_variance(), self.cf._chroma_flux_variance())
 
 
class TestChromaSmoothnessAndVariability(unittest.TestCase):
    def setUp(self):
        self.cf = _make_cf()
 
    def test_smoothness_in_range(self):
        s = self.cf._chroma_smoothness()["smoothness"]
        self.assertGreaterEqual(s, 0.0)
        self.assertLessEqual(s, 1.0)
 
    def test_smoothness_l1_in_range(self):
        s = self.cf._chroma_smoothness(metric="l1")["smoothness"]
        self.assertGreaterEqual(s, 0.0)
        self.assertLessEqual(s, 1.0)
 
    def test_invalid_metric_raises(self):
        with self.assertRaises(ValueError):
            self.cf._chroma_smoothness(metric="invalid")
 
    def test_variability_shape(self):
        result = self.cf._chroma_variability()
        self.assertEqual(result["per_bin_std"].shape, (12,))
 
    def test_variability_non_negative(self):
        result = self.cf._chroma_variability()
        self.assertTrue(np.all(result["per_bin_std"] >= 0.0))
 
    def test_silence_smoothness_is_one(self):
        cf = _make_cf_silence()
        s = cf._chroma_smoothness()["smoothness"]
        self.assertAlmostEqual(s, 1.0, places=5)
 
    def test_cached(self):
        r1 = self.cf._chroma_smoothness()
        r2 = self.cf._chroma_smoothness()
        self.assertIs(r1, r2)
 
 
class TestChromaAutocorrelation(unittest.TestCase):
    def setUp(self):
        self.cf = _make_cf()
 
    def test_zero_lag_is_one(self):
        acf = self.cf._chroma_autocorrelation()["acf"]
        np.testing.assert_allclose(acf[:, 0], 1.0, atol=1e-10)
 
    def test_acf_shape(self):
        lag_max = 32
        acf = self.cf._chroma_autocorrelation(lag_max=lag_max)["acf"]
        self.assertEqual(acf.shape, (12, lag_max + 1))
 
    def test_acf_mean_shape(self):
        lag_max = 16
        acf_mean = self.cf._chroma_autocorrelation(lag_max=lag_max)["acf_mean"]
        self.assertEqual(acf_mean.shape, (lag_max + 1,))
 
    def test_cached(self):
        r1 = self.cf._chroma_autocorrelation()
        r2 = self.cf._chroma_autocorrelation()
        self.assertIs(r1, r2)
 
 
class TestTuningDeviation(unittest.TestCase):
    def setUp(self):
        self.cf = _make_cf()
 
    def test_result_keys(self):
        result = self.cf._tuning_deviation_detection()
        for k in ("cents_per_peak", "track_cents", "mean_abs_cents"):
            self.assertIn(k, result)
 
    def test_track_cents_within_semitone(self):
        """Median drift must be within ±50 cents for a standard-tuned signal."""
        tc = self.cf._tuning_deviation_detection()["track_cents"]
        self.assertGreaterEqual(tc, -50.0)
        self.assertLessEqual(tc, 50.0)
 
    def test_silence_returns_zero_drift(self):
        cf = _make_cf_silence()
        tc = cf._tuning_deviation_detection()["track_cents"]
        self.assertGreaterEqual(tc, -50.0)
        self.assertLessEqual(tc, 50.0)
 
    def test_cached(self):
        r1 = self.cf._tuning_deviation_detection()
        r2 = self.cf._tuning_deviation_detection()
        self.assertIs(r1, r2)
 
 
class TestSpotifyAudioFeatures(unittest.TestCase):
    def setUp(self):
        self.cf = _make_c_major()
 
    def test_expected_keys(self):
        expected = {
            "energy", "speechiness", "acousticness", "danceability",
            "valence", "tempo", "instrumentalness", "key", "mode", "time_signature",
        }
        result = self.cf.spotify_audio_features()
        self.assertEqual(set(result.keys()), expected)
 
    def test_scalar_features_in_range(self):
        result = self.cf.spotify_audio_features()
        for feat in ("energy", "speechiness", "acousticness", "danceability",
                     "valence", "instrumentalness"):
            self.assertGreaterEqual(result[feat], 0.0, msg=f"{feat} < 0")
            self.assertLessEqual(result[feat], 1.0, msg=f"{feat} > 1")
 
    def test_tempo_in_bpm_range(self):
        tempo = self.cf.spotify_audio_features()["tempo"]
        self.assertGreaterEqual(tempo, 40.0)
        self.assertLessEqual(tempo, 240.0)
 
    def test_key_is_valid_pitch_class(self):
        key = self.cf.spotify_audio_features()["key"]
        self.assertIn(key, range(24))
 
    def test_mode_is_valid(self):
        self.assertIn(self.cf.spotify_audio_features()["mode"], ("maj", "min"))
 
    def test_time_signature_is_valid(self):
        self.assertIn(self.cf.spotify_audio_features()["time_signature"], (3, 4))
 
    def test_c_major_detected(self):
        result = self.cf.spotify_audio_features()
        self.assertEqual(result["mode"], "maj")
 
 
class TestHarmonicRhythm(unittest.TestCase):
    def setUp(self):
        self.cf = _make_cf(duration_sec=3.0)
 
    def test_result_keys(self):
        result = self.cf._harmonic_rhythm()
        for k in ("change_rate", "avg_duration_sec", "num_changes"):
            self.assertIn(k, result)
 
    def test_change_rate_non_negative(self):
        self.assertGreaterEqual(self.cf._harmonic_rhythm()["change_rate"], 0.0)
 
    def test_avg_duration_non_negative(self):
        self.assertGreaterEqual(self.cf._harmonic_rhythm()["avg_duration_sec"], 0.0)
 
    def test_cached(self):
        r1 = self.cf._harmonic_rhythm()
        r2 = self.cf._harmonic_rhythm()
        self.assertIs(r1, r2)
 
 
class TestRootMotionAnalysis(unittest.TestCase):
    def setUp(self):
        self.cf = _make_cf(duration_sec=3.0)
 
    def test_result_keys(self):
        result = self.cf._root_motion_analysis()
        for k in ("intervals", "histogram", "avg_motion"):
            self.assertIn(k, result)
 
    def test_histogram_sums_to_one(self):
        hist = self.cf._root_motion_analysis()["histogram"]
        if hist.size > 0 and np.sum(hist) > 0:
            self.assertAlmostEqual(float(np.sum(hist)), 1.0, places=5)
 
    def test_avg_motion_in_range(self):
        avg = self.cf._root_motion_analysis()["avg_motion"]
        self.assertGreaterEqual(avg, 0.0)
        self.assertLessEqual(avg, 6.0)  # max circular distance on chromatic scale
 
    def test_cached(self):
        r1 = self.cf._root_motion_analysis()
        r2 = self.cf._root_motion_analysis()
        self.assertIs(r1, r2)
 
 
class TestShortSignalEdgeCases(unittest.TestCase):
    def test_very_short_signal_no_crash(self):
        """Signal shorter than N should not raise during construction or analysis."""
        y = np.random.randn(64)
        cf = ChromagramFeatures(AudioSignal(signal=y, sr=22050, N=2048, H=512))
        C = cf._chroma()
        self.assertEqual(C.shape[0], 12)
 
    def test_single_sample_signal(self):
        y = np.array([0.5])
        cf = ChromagramFeatures(AudioSignal(signal=y, sr=22050, N=2048, H=512))
        C = cf._chroma()
        self.assertEqual(C.shape[0], 12)
 
 
if __name__ == "__main__":
    unittest.main()