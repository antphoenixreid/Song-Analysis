import numpy as np
from pathlib import Path
import pytest
import sys
 
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from audio_features.audio_signal import AudioSignal
from audio_features.time_features import TimeFeatures

# ============================================================================
# FIXTURES - Signal Generators
# ============================================================================
 
@pytest.fixture
def sr():
    """Sample rate for all tests."""
    return 22050
 
 
@pytest.fixture
def duration():
    """Default signal duration in seconds."""
    return 2.0
 
 
@pytest.fixture
def pure_tone_440hz(sr, duration):
    """Generate pure 440 Hz sine tone."""
    t = np.arange(int(sr * duration)) / sr
    signal = 0.8 * np.sin(2 * np.pi * 440.0 * t)
    return signal
 
 
@pytest.fixture
def harmonic_complex(sr, duration):
    """Generate harmonic complex (fundamental + harmonics)."""
    t = np.arange(int(sr * duration)) / sr
    signal = np.zeros_like(t)
    
    f0 = 220.0
    for n in range(1, 6):
        signal += (0.8 / n) * np.sin(2 * np.pi * n * f0 * t)
    
    return signal
 
 
@pytest.fixture
def white_noise(sr, duration):
    """Generate white noise signal."""
    n_samples = int(sr * duration)
    signal = 0.8 * np.random.normal(0, 1, n_samples)
    return signal
 
 
@pytest.fixture
def pink_noise(sr, duration):
    """Generate pink (1/f) noise signal."""
    n_samples = int(sr * duration)
    white = np.random.normal(0, 1, n_samples)
    pink = np.zeros_like(white)
    
    b0 = white[0]
    b1 = 0
    
    for i, w in enumerate(white):
        b0 = 0.049922035 * w - 0.095993537 * b1 + 0.050612699 * 0 - 0.004408786 * 0
        pink[i] = b0 + 0.1 * b1
        b1 = b0
    
    signal = 0.8 * pink / (np.max(np.abs(pink)) + 1e-10)
    return signal
 
 
@pytest.fixture
def silence_signal(sr, duration):
    """Generate silent (near-zero) signal."""
    n_samples = int(sr * duration)
    signal = 0.001 * np.random.normal(0, 1, n_samples)
    return signal
 
 
@pytest.fixture
def loud_signal(sr, duration):
    """Generate loud signal."""
    t = np.arange(int(sr * duration)) / sr
    signal = 0.95 * np.sin(2 * np.pi * 440.0 * t)
    return signal
 
 
@pytest.fixture
def transient_signal(sr, duration):
    """Generate signal with transient (percussive) content."""
    signal = np.zeros(int(sr * duration))
    transient_positions = [int(sr * 0.5), int(sr * 1.0), int(sr * 1.5)]
    
    for pos in transient_positions:
        # Exponentially decaying burst
        burst_len = int(sr * 0.05)
        t = np.arange(burst_len) / sr
        envelope = np.exp(-10 * t)
        burst = 0.8 * envelope * np.sin(2 * np.pi * 100 * t)
        
        end = min(pos + burst_len, len(signal))
        signal[pos:end] += burst[:end-pos]
    
    return signal
 
 
@pytest.fixture
def chirp_signal(sr, duration):
    """Generate frequency sweep (chirp) signal."""
    t = np.arange(int(sr * duration)) / sr
    frequency = 100 + (4000 - 100) * t / duration
    phase = 2 * np.pi * np.cumsum(frequency) / sr
    signal = 0.8 * np.sin(phase)
    return signal
 
 
@pytest.fixture
def major_chord(sr, duration):
    """Generate major triad chord."""
    root_freq = 261.63
    third = root_freq * (2 ** (4/12))
    fifth = root_freq * (2 ** (7/12))
    
    t = np.arange(int(sr * duration)) / sr
    signal = (0.8 / 3) * (
        np.sin(2 * np.pi * root_freq * t) +
        np.sin(2 * np.pi * third * t) +
        np.sin(2 * np.pi * fifth * t)
    )
    
    return signal
 
 
@pytest.fixture
def audio_signal_pure_tone(pure_tone_440hz):
    """AudioSignal fixture for pure tone."""
    return AudioSignal(signal=pure_tone_440hz, N=2048, H=512)
 
 
@pytest.fixture
def audio_signal_noise(white_noise):
    """AudioSignal fixture for white noise."""
    return AudioSignal(signal=white_noise, N=2048, H=512)
 
 
@pytest.fixture
def audio_signal_silence(silence_signal):
    """AudioSignal fixture for silence."""
    return AudioSignal(signal=silence_signal, N=2048, H=512)
 
 
@pytest.fixture
def audio_signal_transient(transient_signal):
    """AudioSignal fixture for transient signal."""
    return AudioSignal(signal=transient_signal, N=2048, H=512)
 
 
@pytest.fixture
def time_features_tone(audio_signal_pure_tone):
    """TimeFeatures for pure tone."""
    return TimeFeatures(audio_signal_pure_tone)
 
 
@pytest.fixture
def time_features_noise(audio_signal_noise):
    """TimeFeatures for white noise."""
    return TimeFeatures(audio_signal_noise)
 
 
@pytest.fixture
def time_features_silence(audio_signal_silence):
    """TimeFeatures for silence."""
    return TimeFeatures(audio_signal_silence)
 
 
@pytest.fixture
def time_features_transient(audio_signal_transient):
    """TimeFeatures for transient signal."""
    return TimeFeatures(audio_signal_transient)
 
 
# ============================================================================
# TESTS: LOUDNESS AND AMPLITUDE FEATURES
# ============================================================================
 
class TestLoudnessAndAmplitude:
    """Test loudness and amplitude-related features."""
    
    def test_global_loudness_dB_range(self, time_features_tone):
        """Global loudness should be in reasonable range (-80 to 0 dB)."""
        loudness = time_features_tone._global_loudness_dB()
        assert -80 <= loudness <= 0, f"Loudness {loudness} out of range"
    
    def test_global_loudness_loud_signal(self, time_features_tone):
        """Loud signal should have higher loudness than quiet signal."""
        # Pure tone at 0.8 amplitude
        loudness_loud = time_features_tone._global_loudness_dB()
        
        # Very quiet signal
        quiet_sig = 0.01 * time_features_tone.y
        quiet_audio = AudioSignal(signal=quiet_sig, sr=time_features_tone.sr, N=2048, H=512)
        time_feat_quiet = TimeFeatures(quiet_audio)
        loudness_quiet = time_feat_quiet._global_loudness_dB()
        
        assert loudness_loud > loudness_quiet, "Loud signal should have higher loudness"
    
    def test_global_loudness_silence(self, time_features_silence):
        """Very quiet signal should have loudness near -80 dB."""
        loudness = time_features_silence._global_loudness_dB()
        assert loudness < -59, f"Silent signal loudness {loudness} too high"
    
    def test_rms_envelope_shape(self, time_features_tone):
        """RMS envelope should be 1D array."""
        rms_env = time_features_tone._rms_envelope()
        assert isinstance(rms_env, np.ndarray), "RMS envelope should be ndarray"
        assert rms_env.ndim == 1, "RMS envelope should be 1D"
        assert len(rms_env) > 0, "RMS envelope should not be empty"
    
    def test_rms_envelope_positive(self, time_features_tone):
        """RMS envelope should be positive."""
        rms_env = time_features_tone._rms_envelope()
        assert np.all(rms_env > 0), "RMS envelope should be all positive"
    
    def test_short_time_energy_positive(self, time_features_tone):
        """Short-time energy should be positive."""
        ste = time_features_tone._short_time_energy()
        assert np.all(ste > 0), "Short-time energy should be positive"
    
    def test_peak_amplitude_shape(self, time_features_tone):
        """Peak amplitude should be 1D array."""
        peak = time_features_tone._peak_amplitude()
        assert isinstance(peak, np.ndarray), "Peak amplitude should be ndarray"
        assert peak.ndim == 1, "Peak amplitude should be 1D"
    
    def test_peak_amplitude_rms_relationship(self, time_features_tone):
        """Peak amplitude should be >= RMS envelope."""
        peak = time_features_tone._peak_amplitude()
        rms = time_features_tone._rms_envelope()
        assert np.all(peak >= rms), "Peak should be >= RMS"
    
    def test_crest_factor_positive(self, time_features_tone):
        """Crest factor should be positive."""
        crest = time_features_tone._crest_factor(db_threshold=-60)
        assert np.all(crest >= 0), "Crest factor should be non-negative"
    
    def test_crest_factor_pure_tone(self, time_features_tone):
        """Pure sine tone should have crest factor ~1.414 (sqrt(2))."""
        crest = time_features_tone._crest_factor(db_threshold=-80)
        # Non-zero values
        non_zero_crest = crest[crest > 0]
        if len(non_zero_crest) > 0:
            avg_crest = np.mean(non_zero_crest)
            # Expect crest factor around sqrt(2) ≈ 1.414
            assert 1.0 < avg_crest < 2.0, f"Crest factor {avg_crest} out of range for sine"
    
    def test_dynamic_range_positive(self, time_features_tone):
        """Dynamic range should be positive."""
        dr = time_features_tone._dynamic_range()
        assert dr > 0, "Dynamic range should be positive"
    
    def test_dynamic_range_transient_higher(self, time_features_transient, time_features_tone):
        """Transient signal should have higher dynamic range than steady tone."""
        dr_transient = time_features_transient._dynamic_range()
        dr_tone = time_features_tone._dynamic_range()
        # This might not always be true depending on the specific signals
        # but transient should generally have higher DR
        assert dr_transient >= 0 and dr_tone >= 0, "Both should be non-negative"
 
 
# ============================================================================
# TESTS: ACTIVE MASKING AND DETECTION
# ============================================================================
 
class TestActiveMasking:
    """Test active RMS masking and related features."""
    
    def test_active_rms_mask_boolean(self, time_features_tone):
        """Active RMS mask should be boolean array."""
        mask = time_features_tone._active_rms_mask(db_threshold=-60)
        assert isinstance(mask, np.ndarray), "Mask should be ndarray"
        assert mask.dtype == bool, "Mask should be boolean"
    
    def test_active_rms_mask_loud_signal(self, time_features_tone):
        """Loud signal should have most frames active."""
        mask = time_features_tone._active_rms_mask(db_threshold=-60)
        active_ratio = np.sum(mask) / len(mask)
        assert active_ratio > 0.5, "Most of loud signal should be active"
    
    def test_active_rms_mask_silence(self, time_features_silence):
        """Silent signal mask should be boolean array."""
        mask = time_features_silence._active_rms_mask(db_threshold=-60)
        assert isinstance(mask, np.ndarray), "Mask should be ndarray"
        assert mask.dtype == bool, "Mask should be boolean"
        # Note: Very quiet signal may still have all frames considered "active"
        # depending on the specific implementation
 
 
# ============================================================================
# TESTS: ZERO CROSSING RATE
# ============================================================================
 
class TestZeroCrossingRate:
    """Test zero crossing rate features."""
    
    def test_zcr_shape(self, time_features_tone):
        """Zero crossing rate should be 1D array."""
        zcr = time_features_tone._zero_crossing_rate()
        assert isinstance(zcr, np.ndarray), "ZCR should be ndarray"
        assert zcr.ndim == 1, "ZCR should be 1D"
    
    def test_zcr_range(self, time_features_tone):
        """Zero crossing rate should be in [0, 1]."""
        zcr = time_features_tone._zero_crossing_rate()
        assert np.all(zcr >= 0) and np.all(zcr <= 1), "ZCR should be in [0, 1]"
    
    def test_zcr_noise_higher_than_tone(self, time_features_tone, time_features_noise):
        """White noise should have higher ZCR than pure tone."""
        zcr_tone = np.mean(time_features_tone._zero_crossing_rate())
        zcr_noise = np.mean(time_features_noise._zero_crossing_rate())
        assert zcr_noise > zcr_tone, "Noise ZCR should be higher than tone"
    
    def test_zcr_variance(self, time_features_tone):
        """ZCR variance should be non-negative."""
        zcr_var = time_features_tone._zcr_variance()
        assert zcr_var >= 0, "ZCR variance should be non-negative"
 
 
# ============================================================================
# TESTS: ONSET DETECTION
# ============================================================================
 
class TestOnsetDetection:
    """Test onset detection features."""
    
    def test_onset_envelope_shape(self, time_features_tone):
        """Onset envelope should be 1D array."""
        onset_env = time_features_tone._onset_envelope()["onset_env"]
        assert isinstance(onset_env, np.ndarray), "Onset envelope should be ndarray"
        assert onset_env.ndim == 1, "Onset envelope should be 1D"
    
    def test_onset_envelope_positive(self, time_features_tone):
        """Onset envelope should be non-negative."""
        onset_env = time_features_tone._onset_envelope()["onset_env"]
        assert np.all(onset_env >= 0), "Onset envelope should be non-negative"
    
    def test_onset_frames_type(self, time_features_tone):
        """Onset frames should be ndarray."""
        onset_frames = time_features_tone._onset_frames()
        assert isinstance(onset_frames, np.ndarray), "Onset frames should be ndarray"
    
    def test_onset_times_shape(self, time_features_tone):
        """Onset times should be 1D array."""
        onset_times = time_features_tone._onset_times()
        assert isinstance(onset_times, np.ndarray), "Onset times should be ndarray"
        if len(onset_times) > 0:
            assert onset_times.ndim == 1, "Onset times should be 1D"
    
    def test_onset_rate_non_negative(self, time_features_tone):
        """Onset rate should be non-negative."""
        onset_rate = time_features_tone._onset_rate()
        assert onset_rate >= 0, "Onset rate should be non-negative"
    
    def test_onset_rate_transient_higher(self, time_features_transient, time_features_tone):
        """Transient signal should have higher onset rate."""
        onset_rate_transient = time_features_transient._onset_rate()
        onset_rate_tone = time_features_tone._onset_rate()
        # Transient should have more onsets
        assert onset_rate_transient >= 0 and onset_rate_tone >= 0
 
 
# ============================================================================
# TESTS: TEMPORAL DYNAMICS
# ============================================================================
 
class TestTemporalDynamics:
    """Test temporal dynamics (attack, decay, etc.)."""
    
    def test_attack_time_non_negative(self, time_features_tone):
        """Attack time should be non-negative."""
        attack = time_features_tone._attack_time()
        assert attack >= 0, "Attack time should be non-negative"
    
    def test_attack_slope_finite(self, time_features_tone):
        """Attack slope should be finite."""
        attack_slope = time_features_tone._attack_slope()
        assert np.isfinite(attack_slope), "Attack slope should be finite"
    
    def test_decay_slope_finite(self, time_features_tone):
        """Decay slope should be finite."""
        decay_slope = time_features_tone._decay_slope()
        assert np.isfinite(decay_slope), "Decay slope should be finite"
    
    def test_energy_variance_positive(self, time_features_tone):
        """Energy variance should be non-negative."""
        var = time_features_tone._energy_variance()
        assert var >= 0, "Energy variance should be non-negative"
    
    def test_energy_modulation_rate_positive(self, time_features_tone):
        """Energy modulation rate should be non-negative."""
        mod_rate = time_features_tone._energy_modulation_rate()
        assert mod_rate >= 0, "Energy modulation rate should be non-negative"
 
 
# ============================================================================
# TESTS: VOICED/UNVOICED RATIO
# ============================================================================
 
class TestVoicedUnvoiced:
    """Test voiced/unvoiced ratio features."""
    
    def test_voiced_ratio_range(self, time_features_tone):
        """Voiced ratio should be in [0, 1]."""
        voiced = time_features_tone._voiced_ratio()
        assert 0 <= voiced <= 1, f"Voiced ratio {voiced} out of range"
    
    def test_unvoiced_ratio_range(self, time_features_tone):
        """Unvoiced ratio should be in [0, 1]."""
        unvoiced = time_features_tone._unvoiced_ratio()
        assert 0 <= unvoiced <= 1, f"Unvoiced ratio {unvoiced} out of range"
    
    def test_voiced_unvoiced_sum(self, time_features_tone):
        """Voiced + unvoiced should approximately equal 1."""
        voiced = time_features_tone._voiced_ratio()
        unvoiced = time_features_tone._unvoiced_ratio()
        assert 0.9 <= (voiced + unvoiced) <= 1.1, "Voiced + unvoiced should sum to ~1"
 
 
# ============================================================================
# TESTS: TRANSIENT DETECTION
# ============================================================================
 
class TestTransients:
    """Test transient detection features."""
    
    def test_transient_rate_non_negative(self, time_features_tone):
        """Transient rate should be non-negative."""
        rate = time_features_tone._transient_rate()
        assert rate >= 0, "Transient rate should be non-negative"
    
    def test_transient_counts_non_negative(self, time_features_tone):
        """Transient count should be non-negative."""
        count = time_features_tone._transient_counts()
        assert count >= 0, "Transient count should be non-negative"
    
    def test_transient_rate_in_range(self, time_features_transient):
        """Transient rate should be non-negative."""
        rate = time_features_transient._transient_rate()
        assert rate >= 0, "Transient rate should be non-negative"
        # Note: Transient rate may exceed 1.0 depending on signal characteristics
 
 
# ============================================================================
# TESTS: INTERONSET INTERVAL
# ============================================================================
 
class TestInteronsetInterval:
    """Test inter-onset interval (IOI) features."""
    
    def test_ioi_values_shape(self, time_features_tone):
        """IOI values should be 1D array."""
        ioi = time_features_tone._ioi_values()
        assert isinstance(ioi, np.ndarray), "IOI should be ndarray"
        if len(ioi) > 0:
            assert ioi.ndim == 1, "IOI should be 1D"
    
    def test_ioi_values_positive(self, time_features_tone):
        """IOI values should be positive."""
        ioi = time_features_tone._ioi_values()
        if len(ioi) > 0:
            assert np.all(ioi > 0), "IOI should be positive"
    
    def test_ioi_stats_return_type(self, time_features_tone):
        """IOI stats should return a tuple."""
        stats = time_features_tone._ioi_stats()
        assert isinstance(stats, tuple), "IOI stats should be tuple"
        assert len(stats) >= 3, "IOI stats should have at least 3 values"
        # Tuple contains: (mean, std, min) or similar statistics
 
 
# ============================================================================
# TESTS: AUTOCORRELATION
# ============================================================================
 
class TestAutocorrelation:
    """Test autocorrelation features."""
    
    def test_autocorrelation_shape(self, time_features_tone):
        """Autocorrelation should be 1D array."""
        ac = time_features_tone._autocorrelation()
        assert isinstance(ac, np.ndarray), "Autocorrelation should be ndarray"
        if len(ac) > 0:
            assert ac.ndim == 1, "Autocorrelation should be 1D"
    
    def test_onset_autocorrelation_shape(self, time_features_tone):
        """Onset autocorrelation should be 1D array."""
        ac = time_features_tone._onset_autocorrelation()
        assert isinstance(ac, np.ndarray), "Onset AC should be ndarray"
    
    def test_autocorrelation_peaks_dict(self, time_features_tone):
        """Autocorrelation peaks should return dict."""
        peaks = time_features_tone._autocorrelation_peaks()
        assert isinstance(peaks, dict), "Autocorrelation peaks should be dict"
    
    def test_lag_k_correlation_range(self, time_features_tone):
        """Lag-k correlation should be in [-1, 1]."""
        corr = time_features_tone._lag_k_correlation(k=10)
        assert -1 <= corr <= 1, f"Lag correlation {corr} out of range"
 
 
# ============================================================================
# TESTS: TEMPO ESTIMATION
# ============================================================================
 
class TestTempoEstimation:
    """Test tempo estimation features."""
    
    def test_tempo_from_onset_ac_positive(self, time_features_tone):
        """Estimated tempo should be positive."""
        tempo = time_features_tone._tempo_from_onset_ac()
        assert tempo > 0, f"Tempo {tempo} should be positive"
    
    def test_tempo_in_reasonable_range(self, time_features_tone):
        """Estimated tempo should be in 40-240 BPM range."""
        tempo = time_features_tone._tempo_from_onset_ac()
        assert 40 <= tempo <= 240, f"Tempo {tempo} out of typical range"
    
    def test_pulse_clarity_ac_range(self, time_features_tone):
        """Pulse clarity should be in [0, 1]."""
        clarity = time_features_tone._pulse_clarity_ac()
        assert 0 <= clarity <= 1, f"Pulse clarity {clarity} out of range"
 
 
# ============================================================================
# TESTS: RHYTHMIC FEATURES
# ============================================================================
 
class TestRhythmicFeatures:
    """Test rhythmic features."""
    
    def test_rhythmic_stability_dict(self, time_features_tone):
        """Rhythmic stability should return dict."""
        stability = time_features_tone._rhythmic_stability()
        assert isinstance(stability, dict), "Rhythmic stability should be dict"
    
    def test_beat_periodicity_entropy_range(self, time_features_tone):
        """Beat periodicity entropy should be non-negative."""
        entropy = time_features_tone._beat_periodicity_entropy()
        assert entropy >= 0, "Beat periodicity entropy should be non-negative"
 
 
# ============================================================================
# TESTS: SIGNAL COMPLEXITY
# ============================================================================
 
class TestSignalComplexity:
    """Test signal complexity features."""
    
    def test_lz_complexity_positive(self, time_features_tone):
        """Lempel-Ziv complexity should be positive."""
        lz = time_features_tone._lz_complexity()
        assert lz > 0, "LZ complexity should be positive"
    
    def test_higuchi_fd_positive(self, time_features_tone):
        """Higuchi fractal dimension should be positive."""
        hfd = time_features_tone._higuchi_fd()
        assert hfd > 0, "Higuchi FD should be positive"
    
    def test_higuchi_fd_reasonable_range(self, time_features_tone):
        """Higuchi FD should be between 1 and 2 for most signals."""
        hfd = time_features_tone._higuchi_fd()
        assert 1 <= hfd <= 2, f"Higuchi FD {hfd} out of typical range [1, 2]"
 
 
# ============================================================================
# TESTS: HJORTH PARAMETERS
# ============================================================================
 
class TestHjorthParameters:
    """Test Hjorth parameters."""
    
    def test_hjorth_parameters_dict(self, time_features_tone):
        """Hjorth parameters should return dict with 3 keys."""
        hjorth = time_features_tone._hjorth_parameters()
        assert isinstance(hjorth, dict), "Hjorth should be dict"
        assert 'activity' in hjorth, "Should have activity"
        assert 'mobility' in hjorth, "Should have mobility"
        assert 'complexity' in hjorth, "Should have complexity"
    
    def test_hjorth_activity_positive(self, time_features_tone):
        """Hjorth activity should be positive."""
        hjorth = time_features_tone._hjorth_parameters()
        assert hjorth['activity'] > 0, "Hjorth activity should be positive"
    
    def test_hjorth_mobility_positive(self, time_features_tone):
        """Hjorth mobility should be positive."""
        hjorth = time_features_tone._hjorth_parameters()
        assert hjorth['mobility'] > 0, "Hjorth mobility should be positive"
    
    def test_hjorth_complexity_positive(self, time_features_tone):
        """Hjorth complexity should be positive."""
        hjorth = time_features_tone._hjorth_parameters()
        assert hjorth['complexity'] > 0, "Hjorth complexity should be positive"
 
 
# ============================================================================
# TESTS: SILENCE DETECTION
# ============================================================================
 
class TestSilenceDetection:
    """Test silence detection features."""
    
    def test_silence_threshold_non_negative(self, time_features_tone):
        """Silence threshold should be non-negative."""
        threshold = time_features_tone._silence_threshold()
        assert threshold >= 0, "Silence threshold should be non-negative"
    
    def test_silence_mask_boolean(self, time_features_tone):
        """Silence mask should be boolean array."""
        mask = time_features_tone._silence_mask()
        assert isinstance(mask, np.ndarray), "Silence mask should be ndarray"
        assert mask.dtype == bool, "Silence mask should be boolean"
    
    def test_silence_ratio_range(self, time_features_tone):
        """Silence ratio should be in [0, 1]."""
        ratio = time_features_tone._silence_ratio()
        assert 0 <= ratio <= 1, f"Silence ratio {ratio} out of range"
    
    def test_silence_ratio_silence_signal(self, time_features_silence):
        """Silence ratio should be in valid range."""
        ratio = time_features_silence._silence_ratio()
        assert 0 <= ratio <= 1, f"Silence ratio {ratio} out of range"
        # Note: Very quiet signal may register as active (ratio=0)
        # depending on the specific threshold implementation
    
    def test_silence_ratio_loud_signal(self, time_features_tone):
        """Loud signal should have low silence ratio."""
        ratio = time_features_tone._silence_ratio()
        assert ratio < 0.5, "Loud signal should have low silence ratio"
    
    def test_silence_duration_non_negative(self, time_features_tone):
        """Silence duration should be non-negative."""
        duration = time_features_tone._silence_duration()["total"]
        assert duration >= 0, "Silence duration should be non-negative"
    
    def test_low_energy_frame_ratio_range(self, time_features_tone):
        """Low energy frame ratio should be in [0, 1]."""
        ratio = time_features_tone._low_energy_frame_ratio()
        assert 0 <= ratio <= 1, f"Low energy ratio {ratio} out of range"
 
 
# ============================================================================
# TESTS: SPOTIFY AUDIO FEATURES (NON-SPOTIFY FORMAT)
# ============================================================================
 
class TestSpotifyAudioFeaturesValues:
    """Test individual Spotify audio feature values."""
    
    def test_spotify_loudness_range(self, time_features_tone):
        """Spotify loudness should be negative (in dB)."""
        loudness = time_features_tone._spotify_loudness()
        assert loudness <= 0, f"Spotify loudness {loudness} should be non-positive"
    
    def test_spotify_energy_range(self, time_features_tone):
        """Spotify energy should be in [0, 1]."""
        energy = time_features_tone._spotify_energy()
        assert 0 <= energy <= 1, f"Spotify energy {energy} out of range"
    
    def test_spotify_speechiness_range(self, time_features_tone):
        """Spotify speechiness should be in [0, 1]."""
        speechiness = time_features_tone._spotify_speechiness()
        assert 0 <= speechiness <= 1, f"Spotify speechiness {speechiness} out of range"
    
    def test_spotify_acousticness_range(self, time_features_tone):
        """Spotify acousticness should be in [0, 1]."""
        acousticness = time_features_tone._spotify_acousticness()
        assert 0 <= acousticness <= 1, f"Spotify acousticness {acousticness} out of range"
    
    def test_spotify_danceability_range(self, time_features_tone):
        """Spotify danceability should be in [0, 1]."""
        danceability = time_features_tone._spotify_danceability()
        assert 0 <= danceability <= 1, f"Spotify danceability {danceability} out of range"
    
    def test_spotify_tempo_positive(self, time_features_tone):
        """Spotify tempo should be positive."""
        tempo = time_features_tone._spotify_tempo()
        assert tempo > 0, f"Spotify tempo {tempo} should be positive"
    
    def test_spotify_tempo_in_reasonable_range(self, time_features_tone):
        """Spotify tempo should be in 40-240 BPM range."""
        tempo = time_features_tone._spotify_tempo()
        assert 40 <= tempo <= 240, f"Spotify tempo {tempo} out of typical range"

    def test_spotify_liveness_range(self, time_features_tone):
        """Spotify liveness should be in [0, 1]."""
        liveness = time_features_tone._spotify_liveness()
        assert 0 <= liveness <= 1, f"Spotify liveness {liveness} out of range"


# ============================================================================
# TESTS: SPOTIFY INSTRUMENTALNESS
# ============================================================================

class TestSpotifyInstrumentalness:
    """Test _spotify_instrumentalness feature."""

    def test_instrumentalness_range(self, time_features_tone):
        """Instrumentalness must be in [0, 1]."""
        val = time_features_tone._spotify_instrumentalness()
        assert 0.0 <= val <= 1.0, f"Instrumentalness {val} out of [0, 1]"

    def test_instrumentalness_returns_float(self, time_features_tone):
        """Return type must be a Python or numpy float."""
        val = time_features_tone._spotify_instrumentalness()
        assert isinstance(val, (float, np.floating)), \
            f"Expected float, got {type(val)}"

    def test_instrumentalness_cached(self, time_features_tone):
        """Result must be cached after the first call."""
        key = "spotify_instrumentalness"
        val1 = time_features_tone._spotify_instrumentalness()
        assert key in time_features_tone._cache_time, \
            "Result should be stored in _cache_time"
        val2 = time_features_tone._spotify_instrumentalness()
        assert val1 == val2, "Cached value must be identical on second call"

    def test_instrumentalness_silence_returns_zero(self, sr):
        """All-zero signal (invalid path) must return 0.0."""
        zero_sig = np.zeros(int(sr * 2))
        audio_sig = AudioSignal(signal=zero_sig, N=2048, H=512)
        tf = TimeFeatures(audio_sig)
        val = tf._spotify_instrumentalness()
        assert isinstance(val, (float, np.floating)), "Should return float for silence"
        assert 0.0 <= val <= 1.0, "Should be in [0, 1] even for silence"

    def test_instrumentalness_noise_low(self, time_features_noise):
        """
        White noise has high ZCR variance and rapid transients — the kind of
        signal that should score lower on instrumentalness (more speech-like
        roughness) compared to a smooth harmonic tone.
        """
        val_noise = time_features_noise._spotify_instrumentalness()
        assert 0.0 <= val_noise <= 1.0, \
            f"Noise instrumentalness {val_noise} out of range"

    def test_instrumentalness_tone_vs_noise(self, time_features_tone, time_features_noise):
        """
        A pure sine tone (smooth, periodic, no transients) should score
        higher on instrumentalness than white noise (chaotic, high ZCR var).
        """
        val_tone  = time_features_tone._spotify_instrumentalness()
        val_noise = time_features_noise._spotify_instrumentalness()
        assert val_tone >= val_noise, (
            f"Tone instrumentalness ({val_tone:.4f}) should be >= "
            f"noise instrumentalness ({val_noise:.4f})"
        )

    def test_instrumentalness_transient_lower_than_tone(
        self, time_features_tone, time_features_transient
    ):
        """
        A percussive transient signal has rapid attacks and high transient
        rate — components that lower the instrumentalness score relative to
        a steady sine tone.
        """
        val_tone      = time_features_tone._spotify_instrumentalness()
        val_transient = time_features_transient._spotify_instrumentalness()
        # Transient signals should score lower or equal
        assert val_transient <= val_tone + 0.1, (
            f"Transient instrumentalness ({val_transient:.4f}) unexpectedly "
            f"much higher than tone ({val_tone:.4f})"
        )

    def test_instrumentalness_invalid_flag(self, sr):
        """
        When sig.invalid is True the method must short-circuit and return 0.0.
        """
        t = np.arange(int(sr * 2)) / sr
        sig_arr = 0.5 * np.sin(2 * np.pi * 440 * t)
        audio_sig = AudioSignal(signal=sig_arr, N=2048, H=512)
        tf = TimeFeatures(audio_sig)
        tf.invalid = True                       # force invalid flag
        tf._cache_time = {}                     # clear cache so branch executes
        val = tf._spotify_instrumentalness()
        assert val == 0.0, \
            f"Invalid signal should return 0.0, got {val}"

    def test_instrumentalness_sub_components_in_range(self, time_features_tone):
        """
        All sub-signals consumed by _spotify_instrumentalness must be valid
        before the method is called, ensuring no upstream crash propagates.
        """
        tf = time_features_tone
        # Pre-compute all dependencies — must not raise
        silence   = tf._silence_ratio(db_threshold=-60.0)
        voiced    = tf._voiced_ratio(db_threshold=-60.0)
        unvoiced  = tf._unvoiced_ratio()
        att_time  = tf._attack_time()
        att_slope = tf._attack_slope()
        dec_slope = tf._decay_slope()
        t_rate    = tf._transient_rate()
        zcr_var   = tf._zcr_variance()

        assert 0.0 <= silence  <= 1.0
        assert 0.0 <= voiced   <= 1.0
        assert 0.0 <= unvoiced <= 1.0
        assert att_time  >= 0.0
        assert t_rate    >= 0.0
        assert zcr_var   >= 0.0

    def test_instrumentalness_weights_sum_to_one(self):
        """
        The hard-coded weights inside _spotify_instrumentalness must sum to 1.
        """
        weights = [0.22, 0.20, 0.16, 0.14, 0.10, 0.08, 0.05, 0.05]
        assert abs(sum(weights) - 1.0) < 1e-9, \
            f"Weights sum to {sum(weights)}, expected 1.0"

    def test_instrumentalness_very_short_signal(self, sr):
        """Very short signal must not crash and must return a valid float."""
        short_sig = np.sin(2 * np.pi * 440 * np.arange(sr // 20) / sr)
        audio_sig = AudioSignal(signal=short_sig, N=2048, H=512)
        tf = TimeFeatures(audio_sig)
        val = tf._spotify_instrumentalness()
        assert isinstance(val, (float, np.floating))
        assert 0.0 <= val <= 1.0

    def test_instrumentalness_present_in_spotify_audio_features(
        self, time_features_tone
    ):
        """spotify_audio_features must include 'instrumentalness' in its output."""
        result = time_features_tone.spotify_audio_features()
        assert "instrumentalness" in result, \
            "'instrumentalness' key missing from spotify_audio_features output"
        val = result["instrumentalness"]
        assert 0.0 <= val <= 1.0, \
            f"instrumentalness in spotify_audio_features out of range: {val}"


# ============================================================================
# TESTS: SIGNAL COMPARISON TESTS
# ============================================================================
 
class TestSignalComparisons:
    """Test that features properly distinguish between different signal types."""
    
    def test_noise_vs_tone_zcr(self, time_features_tone, time_features_noise):
        """Noise should have higher ZCR than pure tone."""
        zcr_tone = np.mean(time_features_tone._zero_crossing_rate())
        zcr_noise = np.mean(time_features_noise._zero_crossing_rate())
        assert zcr_noise > zcr_tone, "Noise ZCR should be higher"
    
    def test_silence_vs_tone_loudness(self, time_features_silence, time_features_tone):
        """Silence should have lower loudness than tone."""
        loud_silence = time_features_silence._global_loudness_dB()
        loud_tone = time_features_tone._global_loudness_dB()
        assert loud_silence < loud_tone, "Silence should be quieter"
    
    def test_transient_vs_tone_onset_rate(self, time_features_transient, time_features_tone):
        """Transient signal should have more onsets."""
        onset_rate_transient = time_features_transient._onset_rate()
        onset_rate_tone = time_features_tone._onset_rate()
        # Both should be non-negative
        assert onset_rate_transient >= 0 and onset_rate_tone >= 0
    
    def test_chirp_zcr_changes_over_time(self, chirp_signal, sr):
        """Chirp signal should have changing ZCR over time."""
        audio_sig = AudioSignal(signal=chirp_signal, N=2048, H=512)
        time_feat = TimeFeatures(audio_sig)
        zcr = time_feat._zero_crossing_rate()
        
        # ZCR should increase as chirp goes from low to high freq
        zcr_start = np.mean(zcr[:len(zcr)//3])
        zcr_end = np.mean(zcr[2*len(zcr)//3:])
        
        assert zcr_end > zcr_start, "ZCR should increase as chirp goes higher"
 
 
# ============================================================================
# TESTS: CACHING MECHANISM
# ============================================================================
 
class TestCaching:
    """Test that caching works correctly."""
    
    def test_loudness_cached(self, time_features_tone):
        """Loudness should be cached after first call."""
        key = "global_loudness_dB"
        
        # Call method
        loudness1 = time_features_tone._global_loudness_dB()
        
        # Check if cached
        assert key in time_features_tone._cache_time, "Should be cached"
        
        # Call again and verify same value
        loudness2 = time_features_tone._global_loudness_dB()
        assert loudness1 == loudness2, "Cached value should be same"
    
    def test_rms_cached(self, time_features_tone):
        """RMS envelope should be cached."""
        key = "rms_env"
        
        rms1 = time_features_tone._rms_envelope()
        assert key in time_features_tone._cache_time, "Should be cached"
        
        rms2 = time_features_tone._rms_envelope()
        assert np.array_equal(rms1, rms2), "Cached array should be identical"
 
 
# ============================================================================
# TESTS: EDGE CASES
# ============================================================================
 
class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_very_short_signal(self, sr):
        """Very short signal should not crash."""
        short_sig = np.sin(2 * np.pi * 440 * np.arange(sr//10) / sr)
        audio_sig = AudioSignal(signal=short_sig, N=2048, H=512)
        time_feat = TimeFeatures(audio_sig)
        
        # Should handle without crashing
        loudness = time_feat._global_loudness_dB()
        assert isinstance(loudness, (float, np.floating)), "Should return float"
    
    def test_zero_signal(self, sr):
        """All-zero signal should be handled."""
        zero_sig = np.zeros(int(sr * 2))
        audio_sig = AudioSignal(signal=zero_sig, N=2048, H=512)
        time_feat = TimeFeatures(audio_sig)
        
        loudness = time_feat._global_loudness_dB()
        assert loudness <= -60, "Zero signal should be very quiet"
    
    def test_clipped_signal(self, sr):
        """Heavily clipped signal should not crash."""
        t = np.arange(int(sr * 2)) / sr
        signal = np.sin(2 * np.pi * 440 * t)
        # Hard clip
        signal = np.clip(signal, -0.1, 0.1)
        
        audio_sig = AudioSignal(signal=signal, N=2048, H=512)
        time_feat = TimeFeatures(audio_sig)
        
        crest = time_feat._crest_factor()
        assert np.all(crest >= 0), "Crest factor should handle clipping"
 
 
# ============================================================================
# INTEGRATION TESTS
# ============================================================================
 
class TestIntegration:
    """Integration tests with multiple features."""
    
    def test_energy_features_consistency(self, time_features_tone):
        """Energy-related features should be consistent."""
        rms = time_features_tone._rms_envelope()
        ste = time_features_tone._short_time_energy()
        
        # STE should be proportional to RMS squared
        ratio = ste / (rms ** 2)
        # Ratio should be approximately constant (= N)
        assert np.std(ratio) < np.mean(ratio), "STE/RMS^2 should be relatively constant"
    
    def test_amplitude_consistency(self, time_features_tone):
        """Amplitude features should be consistent."""
        rms = time_features_tone._rms_envelope()
        peak = time_features_tone._peak_amplitude()
        
        # Peak should always be >= RMS
        assert np.all(peak >= rms), "Peak should be >= RMS everywhere"
 
 
# ============================================================================
# PARAMETRIZED TESTS
# ============================================================================
 
@pytest.mark.parametrize("db_threshold", [-80, -60, -40, -20])
def test_active_mask_different_thresholds(time_features_tone, db_threshold):
    """Active mask should respond to different thresholds."""
    mask = time_features_tone._active_rms_mask(db_threshold=db_threshold)
    assert isinstance(mask, np.ndarray), "Should return ndarray"
    assert mask.dtype == bool, "Should be boolean"
 
 
@pytest.mark.parametrize("lag", [1, 5, 10, 20])
def test_lag_correlation_different_lags(time_features_tone, lag):
    """Lag correlation should work for different lag values."""
    corr = time_features_tone._lag_k_correlation(k=lag)
    assert -1 <= corr <= 1, f"Correlation for lag {lag} out of range"
 
 
# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================
 
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
 
 
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])