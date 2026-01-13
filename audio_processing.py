import os
import numpy as np
import librosa
from scipy.signal import find_peaks
from typing import Tuple, Optional, Dict

# ---------- Utilities ----------
EPS = 1e-12

def safe_median(x: np.ndarray, fallback: float = 0.0) -> float:
    m = np.nanmedian(x)
    return float(m) if not np.isnan(m) else float(fallback)

def safe_clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))

def ensure_1d(x):
    x = np.asarray(x)
    if x.ndim == 0:
        return x.reshape(1)
    return x

def robust_normalize(x: np.ndarray, lowp: float = 1.0, highp: float = 99.0) -> np.ndarray:
    """
    Robust normalization to [0,1] using percentile clipping.
    Handles NaNs, constant vectors, empty vectors gracefully.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    if np.all(np.isnan(x)):
        return np.ones_like(x) * 0.5
    med = np.nanmedian(x)
    if np.isnan(med):
        med = 0.0
    x = np.nan_to_num(x, nan=med)
    lo = np.percentile(x, lowp)
    hi = np.percentile(x, highp)
    rng = hi - lo
    if rng < 1e-12:
        return np.ones_like(x) * 0.5
    y = (x - lo) / (rng + EPS)
    return np.clip(y, 0.0, 1.0)

def min_length(*arrays):
    arrs = [np.asarray(a) for a in arrays if a is not None and np.asarray(a).size > 0]
    if not arrs:
        return 0
    return min(a.shape[0] for a in arrs)

def fft_autocorr(x: np.ndarray) -> np.ndarray:
    """FFT-based autocorrelation (returns length 2N-1 raw; caller may slice)."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.array([])
    x = x - np.mean(x)
    f = np.fft.rfft(x, n=2 * x.size)
    ac = np.fft.irfft(f * np.conj(f))
    ac = ac[:2 * x.size - 1]
    return ac

def squash(x: float, lo: float, hi: float) -> float:
    """
    Docstring for squash
    
    :param x: Description
    :type x: float
    :param lo: Description
    :type lo: float
    :param hi: Description
    :type hi: float
    :return: Description
    :rtype: float
    """
    return float(np.clip((x - lo) / (hi - lo + EPS), 0.0, 1.0))

def safe_load(path: str) -> Tuple[np.ndarray, int]:
    try:
        y, sr = librosa.load(path, sr=None, mono=True)

        # Detect EMPTY or SILENT files
        if y is None or len(y) == 0:
            print(f"[ERROR] Empty audio file: {path}")
            return None, None, True
        
        if np.allclose(y, 0):
            print(f"[ERROR] Pure silence detected: {path}")
            return None, None, True
        
        return y, sr, False

    except Exception as e:
        print(f"[ERROR] Could not load {path}: {e}")
        return None, None, True
    
def confidence_weighted(value, confidence, neutral=0.5):
    if confidence < 0.15:
        return neutral
    
    return value
    
# Parent Class
class AudioSignal:
    def __init__(self, audio_path: str, N: int = 2048, H: int = 512):
        if not isinstance(audio_path, str):
            raise ValueError("audio_path must be a string path")
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        self.audio_path = audio_path
        self.N = N
        self.H = H

        # load
        self.y, self.sr, invalid = safe_load(audio_path)

        # -----------------------------
        # GLOBAL VALIDITY CHECKS
        # -----------------------------
        self.invalid = False

        if invalid:
            self.invalid = True
            return

        # failed load / empty file
        if self.y is None or self.y.size == 0:
            self.invalid = True
            return

        # near-constant (all zeros or DC)
        if np.std(self.y) < 1e-6:
            self.invalid = True
            return
        
        # Detect clipping / malformed decode
        if np.max(np.abs(self.y)) > 1.5:
            print(f"[WARN] Clipped / malformed audio: {audio_path}")
            self.invalid = True
            return

        # compute once lazily — requires cache initialized
        self._cache: Dict[str, object] = {}

# ---------- Base Time Features ----------
class TimeFeatures():
    def __init__(self, sig: AudioSignal):
        # super().__init__(audio_path, N, H)
        self.y = sig.y
        self.sr = sig.sr
        self.N = sig.N
        self.H = sig.H
        self._cache_time = sig._cache

        # global loudness check (too quiet to analyze)
        gl = self._global_loudness_dB()
        if gl is None or gl < -70.0:
            self.invalid = True
            return

        # pad short audio
        if len(self.y) < self.N:
            pad = self.N - len(self.y)
            self.y = np.pad(self.y, (0, pad), mode="constant")

        # precompute frequencies
        self._fft_freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.N)

    def _global_loudness_dB(self) -> float:
        if getattr(self, "invalid", False):
            return None
        if "global_loudness_dB" in self._cache_time:
            return self._cache_time["global_loudness_dB"]
        
        rms = np.sqrt(np.mean(self.y**2)) + EPS
        loud_db = 20.0*np.log10(rms)

        self._cache_time["global_loudness_dB"] = loud_db
        return loud_db
    
    def _is_globally_silent(self, thresh_dB: float = -60.0) -> bool:
        loud_db = self._global_loudness_dB()
        return loud_db < thresh_dB
    
    def _valid_energy_mask(self, db_threshold: float = -50.0) -> np.ndarray:
        rms = self._amplitude()
        rms_db = 20.0 * np.log10(np.maximum(rms, EPS))
        return rms_db > db_threshold

    # --- Basic time-domain features ---
    def _amplitude(self) -> np.ndarray:
        if getattr(self, "invalid", False):
            return None
        if "amplitude" in self._cache_time:
            return self._cache_time["amplitude"]
        rms = librosa.feature.rms(y=self.y, frame_length=self.N, hop_length=self.H)[0]
        self._cache_time["amplitude"] = rms
        return rms

    def _amplitude_dB(self) -> np.ndarray:
        if getattr(self, "invalid", False):
            return None
        if "amplitude_dB" in self._cache_time:
            return self._cache_time["amplitude_dB"]
        amp = self._amplitude()
        db = 20 * np.log10(np.maximum(amp, EPS))
        db = np.clip(db, -120.0, 0.0)
        self._cache_time["amplitude_dB"] = db
        return db
    
    def _peak_amplitude(self) -> np.ndarray:
        if getattr(self, "invalid", False):
            return None
        if "peak_amplitude" in self._cache_time:
            return self._cache_time["peak_amplitude"]
        
        # Frame the signal (shape: [frame_length, num_frames])
        frames = librosa.util.frame(
            self.y,
            frame_length=self.N,
            hop_length=self.H
        )

        # Peak absolute value per frame
        peak = np.max(np.abs(frames), axis=0)

        self._cache_time["peak_amplitude"] = peak
        return peak
    
    def _peak_amplitude_dB(self) -> np.ndarray:
        if getattr(self, "invalid", False):
            return None
        if "peak_amplitude_dB" in self._cache_time:
            return self._cache_time["peak_amplitude_dB"]
        
        peak = self._peak_amplitude()
        peak_dB = 20.0*np.log10(np.maximum(peak, EPS))
        peak_dB = np.clip(peak_dB, -120.0, 0.0)

        self._cache_time["peak_amplitude_dB"] = peak_dB
        return peak_dB
    
    def _crest_factor(self) -> np.ndarray:
        if getattr(self, "invalid", False):
            return None
        if "crest_factor" in self._cache_time:
            return self._cache_time["crest_factor"]
        
        if self._is_globally_silent():
            crest = np.zeros_like(self._amplitude())
            self._cache_time["crest_factor"] = crest
            return crest
        
        peak = self._peak_amplitude()
        rms = self._amplitude()

        # --- ensure equal length ---
        L = min(len(peak), len(rms))
        peak = peak[:L]
        rms  = rms[:L]

        mask = self._valid_energy_mask()

        # also trim mask to same length
        mask = mask[:L]

        crest = np.zeros_like(rms)
        crest[mask] = peak[mask] / np.maximum(rms[mask], EPS)

        self._cache_time["crest_factor"] = crest
        return crest
    
    def _crest_factor_dB(self) -> np.ndarray:
        if getattr(self, "invalid", False):
            return None
        if "crest_factor_dB" in self._cache_time:
            return self._cache_time["crest_factor_dB"]
        
        crest = self._crest_factor()
        
        if self._is_globally_silent():
            cf_db = np.zeros_like(crest)
            self._cache_time["crest_factor_dB"] = cf_db
            return cf_db
        
        crest_dB = 20.0 * np.log10(np.maximum(crest, EPS))
        crest_dB = np.clip(crest_dB, -6.0, 40.0)

        self._cache_time["crest_factor_dB"] = crest_dB
        return crest_dB
    
    def _crest_factor_track(self, stat: str = 'median') -> float:
        """
        Agregate crest factor to a single scalar
        """
        if getattr(self, "invalid", False):
            return None
        crest = self._crest_factor()
        if crest.size == 0:
            return 0.0
        
        if stat == 'median':
            return float(np.median(crest))
        elif stat == 'mean':
            return float(np.mean(crest))
        elif stat == 'max':
            return float(np.max(crest))

    def _energy_envelope(self) -> np.ndarray:
        if getattr(self, "invalid", False):
            return None
        if "energy_envelope" in self._cache_time:
            return self._cache_time["energy_envelope"]
        # vectorized framing energy
        frames = librosa.util.frame(self.y, frame_length=self.N, hop_length=self.H).astype(float)
        E = np.sum(frames**2, axis=0)
        self._cache_time["energy_envelope"] = E
        return E
    
    def _energy_variance(self) -> float:
        if getattr(self, "invalid", False):
            return None
        if "energy_variance" in self._cache_time:
            return self._cache_time["energy_variance"]
        
        rms = self._amplitude()
        if rms.size == 0:
            self._cache_time["energy_variance"] = 0.0
            return 0.0
        
        # Normalize RMS to reduce dependence on absolute level
        q75, q25 = np.percentile(rms, [75, 25])
        iqr = q75 - q25

        ev = float(iqr/(np.median(rms) + EPS))

        self._cache_time["energy_variance"] = ev
        return ev
    
    def _energy_mod_rate(self) -> float:
        if getattr(self, "invalid", False):
            return None
        if "energy_mod_rate" in self._cache_time:
            return self._cache_time["energy_mod_rate"]
        
        rms = self._amplitude()
        if rms.size < 2:
            self._cache_time["energy_mod_rate"] = 0.0
            return 0.0
        
        # Normalize RMS to remove absolute loudness dependency
        q75, q25 = np.percentile(rms, [75, 25])
        iqr = q75 - q25

        rms_n = (rms - q25) / (iqr + EPS)

        mod_rate = float(np.mean(np.abs(np.diff(rms_n))))
        mod_rate *= self.sr/float(self.H)  # convert to Hz

        self._cache_time["energy_mod_rate"] = mod_rate
        return mod_rate
    
    def _energy_modulation_signal(self) -> np.ndarray:
        """
        Frame-wise absolute energy change.
        """
        if getattr(self, "invalid", False):
            return None
        if "energy_modulation_signal" in self._cache_time:
            return self._cache_time["energy_modulation_signal"]

        rms = self._amplitude()
        if rms.size < 2:
            return np.zeros_like(rms)

        rms_n = rms / (np.mean(rms) + EPS)
        mod_sig = np.abs(np.diff(rms_n))

        self._cache_time["energy_modulation_signal"] = mod_sig
        return mod_sig

    def _zero_crossing_rate(self) -> np.ndarray:
        if getattr(self, "invalid", False):
            return None
        if "zcr" in self._cache_time:
            return self._cache_time["zcr"]
        z = librosa.feature.zero_crossing_rate(y=self.y, frame_length=self.N, hop_length=self.H)[0]
        self._cache_time["zcr"] = z
        return z

    def _dynamic_range(self) -> float:
        if getattr(self, "invalid", False):
            return None
        if "dynamic_range" in self._cache_time:
            return self._cache_time["dynamic_range"]
        
        if self._is_globally_silent():
            self._cache_time["dynamic_range"] = 0.0
            return 0.0

        rms = self._amplitude()
        if np.all(rms == 0):
            dr = 0.0
            self._cache_time["dynamic_range"] = dr
            return dr
        
        # robust percentiles
        low = np.percentile(rms, 10)
        high = np.percentile(rms, 95)
        low = max(low, EPS)
        dr = 20.0 * np.log10(max(high / low, EPS))
        dr = np.clip(dr, 0.0, 60.0)
        self._cache_time["dynamic_range"] = float(dr)
        return float(dr)

    # --- Tempo & onset ---
    def _onset_env(self) -> np.ndarray:
        if getattr(self, "invalid", False):
            return None
        if "onset_env" in self._cache_time:
            return self._cache_time["onset_env"]

        onset = librosa.onset.onset_strength(
            y=self.y, sr=self.sr, hop_length=self.H
        )

        if onset.size == 0 or np.all(onset == 0):
            onset_n = np.zeros_like(onset)
        else:
            onset_n = onset / (np.max(onset) + EPS)

        self._cache_time["onset_env"] = onset_n
        return onset_n

    
    def _onset_autocorr(self) -> np.ndarray:
        if getattr(self, "invalid", False):
            return None
        if "onset_autocorr" in self._cache_time:
            return self._cache_time["onset_autocorr"]

        onset = self._onset_env()

        # Early exit for silence / no onsets
        if onset.size < 2 or np.all(onset == 0):
            ac = np.zeros_like(onset)
            self._cache_time["onset_autocorr"] = ac
            return ac

        onset_dc = onset - np.mean(onset)
        f = np.fft.rfft(onset_dc)
        ac = np.fft.irfft(f * np.conj(f))

        ac = ac[: len(ac)//2]

        # normalize safely
        m = np.max(np.abs(ac))
        if m == 0:
            ac[:] = 0
        else:
            ac /= m

        self._cache_time["onset_autocorr"] = ac
        return ac
    
    def _pulse_clarity(self) -> float:
        if getattr(self, "invalid", False):
            return None
        if "pulse_clarity" in self._cache_time:
            return self._cache_time["pulse_clarity"]
        
        if self._is_globally_silent():
            self._cache_time["pulse_clarity"] = 0.0
            return 0.0
        
        ac = self._onset_autocorr()
        if ac.size < 4:
            self._cache_time["pulse_clarity"] = 0.0
            return 0.0
        
        peak = np.max(ac[1:])
        mean = np.mean(ac[1:])

        clarity = float((peak - mean)/(peak + EPS))
        clarity = safe_clip01(clarity)

        self._cache_time["pulse_clarity"] = clarity
        return clarity

    def _tempo_var(self, window_seconds: float = 8.0) -> Tuple[float, float, float]:
        if getattr(self, "invalid", False):
            return None
        # cached key includes hop & window
        key = f"tempo_var_{self.H}_{window_seconds}"
        if key in self._cache_time:
            return self._cache_time[key]
        
        if self._is_globally_silent():
            self._cache_time[key] = (0.0, 0.0, 0.0)
            return self._cache_time[key]

        onset = self._onset_env()
        if onset.size < 2:
            self._cache_time[key] = (0.0, 0.0, 0.0)
            return self._cache_time[key]
        # compute per-window beat tracking; window in frames
        hop_onset = self.H
        win_frames = max(1, int(round(window_seconds * self.sr / hop_onset)))
        step = max(1, win_frames // 2)
        tempos = []
        for i in range(0, max(1, len(onset) - win_frames), step):
            seg = onset[i:i + win_frames]
            if np.all(seg == 0):
                continue
            # beat_track expects full signal; provide an onset_envelope segment and hop length
            try:
                tempo, _ = librosa.beat.beat_track(onset_envelope=seg, sr=self.sr, hop_length=hop_onset)

                if tempo < 55:
                    tempo *= 2
                elif tempo > 200:
                    tempo /= 2

                if tempo > 0:
                    tempos.append(tempo)
            except Exception:
                continue
        if len(tempos) == 0:
            res = (0.0, 0.0, 0.0)
            self._cache_time[key] = res
            return res
        tv = float(np.var(tempos))
        tm = float(np.mean(tempos))
        stab = float(max(0.0, 1.0 - (tv / (tm + EPS))))
        self._cache_time[key] = (tv, tm, stab)
        return self._cache_time[key]
    
    def _silence_ratio(self) -> float:
        if getattr(self, "invalid", False):
            return None
        if "silence_ratio" in self._cache_time:
            return self._cache_time["silence_ratio"]
        
        rms = self._amplitude()
        if rms.size == 0:
            self._cache_time["silence_ratio"] = 0.0
            return 0.0
        
        # Avoid degenerate threshold
        thresh = max(np.percentile(rms, 10), EPS)

        silent_frames = np.sum(rms < thresh)
        ratio = float(silent_frames) / float(rms.size)

        ratio = safe_clip01(ratio)
        self._cache_time["silence_ratio"] = ratio
        return ratio
    
    # Spotify-style spectral features can be added here (Full)
    def loudness_time(self) -> float:
        if getattr(self, "invalid", False):
            return None
        if "loudness_time" in self._cache_time:
            return self._cache_time["loudness_time"]
        
        rms_db = self._amplitude_dB()
        if rms_db.size == 0:
            self._cache_time["loudness_time"] = 0.0
            return 0.0
        
        # Exclude silence frames
        thresh = np.percentile(rms_db, 10)
        active_db = rms_db[rms_db > thresh]

        if active_db.size == 0:
            self._cache_time["loudness_time"] = 0.0
            return 0.0
        
        # Robust aggregation
        loudness = float(np.percentile(active_db, 95))
        loudness -= 0.1*self._dynamic_range()

        # Clip and return
        loudness = np.clip(loudness, -60.0, 0.0)
        self._cache_time["loudness_time"] = loudness
        return loudness
    
    def speechiness_time(self) -> float:
        if getattr(self, "invalid", False):
            return None
        if "speechiness_time" in self._cache_time:
            return self._cache_time["speechiness_time"]

        zcr = self._zero_crossing_rate()
        zcr_median = safe_median(zcr, fallback=0.0)

        energy_mod = self._energy_mod_rate()
        energy_var = self._energy_variance()
        silence = self._silence_ratio()
        pulse = self._pulse_clarity()
        dyn = self._dynamic_range()

        # ---- periodicity ----
        ac = self._onset_autocorr()
        if ac.size > 3:
            peak = np.max(ac[2:])
            mean_rest = np.mean(ac[2:])
            periodicity = max(0.0, peak - mean_rest)
        else:
            periodicity = 0.0

        periodicity_inv = 1.0 - squash(periodicity, 0.05, 0.35)

        # ---- normalize ----
        zcr_n = squash(zcr_median, 0.02, 0.20)
        noise_penalty = squash(zcr_median, 0.30, 0.45)

        mod_n = squash(energy_mod, 0.5, 10.0)
        var_n = squash(energy_var, 0.1, 2.0)
        silence_n = squash(silence, 0.05, 0.6)

        pulse_inv = 1.0 - squash(pulse, 0.1, 0.8)
        dyn_inv = 1.0 - squash(dyn, 5.0, 25.0)

        noise_guard = 1.0 - squash(energy_mod, 8.0, 20.0)

        # ---- weights ----
        speechiness = (
            0.20 * zcr_n +
            0.18 * mod_n +
            0.14 * var_n +
            0.12 * silence_n +
            0.14 * pulse_inv +
            0.10 * dyn_inv +
            0.08 * periodicity_inv +
            0.04 * noise_guard
        )

        # extra penalty for broadband noise
        speechiness -= 0.12 * noise_penalty

        speechiness = float(np.clip(speechiness, 0.0, 1.0))
        self._cache_time["speechiness_time"] = speechiness
        return speechiness
    
    def instrumentalness_time(self) -> float:
        if getattr(self, "invalid", False):
            return None
        if "instrumentalness_time" in self._cache_time:
            return self._cache_time["instrumentalness_time"]
        
        zcr = self._zero_crossing_rate()
        zcr_median = safe_median(zcr, fallback=0.0)

        energy_mod = self._energy_mod_rate()
        energy_var = self._energy_variance()
        silence = self._silence_ratio()
        pulse = self._pulse_clarity()
        dyn = self._dynamic_range()
        
        # Empirical Ranges
        var_n = squash(energy_var, 0.1, 2.0)
        pulse_n = squash(pulse, 0.1, 0.8)
        dyn_n = squash(dyn, 5.0, 25.0)

        zcr_inv = 1.0 - squash(zcr_median, 0.02, 0.25)
        mod_inv = 1.0 - squash(energy_mod, 0.5, 10.0)
        sil_inv = 1.0 - squash(silence, 0.05, 0.6)

        # Weights
        w_pulse = 0.30
        w_mod = 0.20
        w_sil = 0.15
        w_zcr = 0.15
        w_var = 0.10
        w_dyn = 0.10

        instrumentalness = (
            w_pulse*pulse_n +
            w_mod*mod_inv +
            w_sil*sil_inv +
            w_zcr*zcr_inv +
            w_var*var_n +
            w_dyn*dyn_n
        )

        # Clip to [0,1]
        instrumentalness = float(np.clip(instrumentalness, 0.0, 1.0))

        self._cache_time["instrumentalness_time"] = instrumentalness
        return instrumentalness
    
    # Spotify-style spectral features can be added here (partial)
    def energy_time(self) -> float:
        if getattr(self, "invalid", False):
            return None
        if "energy_time" in self._cache_time:
            return self._cache_time["energy_time"]
        
        # Extract features
        rms = self._amplitude()
        if rms.size == 0:
            self._cache_time["energy_time"] = 0.0
            return 0.0
        
        rms_med = safe_median(rms, fallback=0.0)
        energy_var = self._energy_variance()
        energy_mod = self._energy_mod_rate()
        crest = self._crest_factor()
        crest_med = safe_median(crest, fallback=0.0) if crest.size else 0.0

        silence = self._silence_ratio()
        pulse = self._pulse_clarity()

        # Robust normalization
        rms_n = squash(rms_med, 0.01, 0.2)
        var_n = squash(energy_var, 0.1, 2.0)
        mod_n = squash(energy_mod, 0.5, 10.0)
        crest_n = squash(crest_med, 1.5, 6.0)

        sil_inv = 1.0 - squash(silence, 0.05, 0.6)
        pulse_n = squash(pulse, 0.1, 0.8)

        # Weights
        w_rms = 0.15
        w_var = 0.20
        w_mod = 0.20
        w_crest = 0.15
        w_sil = 0.15
        w_pulse = 0.15

        energy = (
            w_rms*rms_n +
            w_var*var_n +
            w_mod*mod_n +
            w_crest*crest_n +
            w_sil*sil_inv +
            w_pulse*pulse_n
        )

        energy = float(np.clip(energy, 0.0, 1.0))
        self._cache_time["energy_time"] = energy

        return energy
    
    def acousticness_time(self) -> float:
        if getattr(self, "invalid", False):
            return None
        if "acousticness" in self._cache_time:
            return self._cache_time["acousticness"]
        
        rms = self._amplitude()
        if rms.size == 0:
            self._cache_time["acousticness"] = 0.0
            return 0.0
        
        rms_med = safe_median(rms, fallback=0.0)
        dyn = self._dynamic_range()
        var = self._energy_variance()
        mod = self._energy_mod_rate()
        crest = self._crest_factor()
        crest_med = safe_median(crest, fallback=0.0) if crest.size else 0.0
        pulse = self._pulse_clarity()
        silence = self._silence_ratio()

        # Normalize Components
        dyn_n = squash(dyn, 6.0, 30.0)
        var_n = squash(var, 0.2, 7.0)
        sil_n = squash(silence, 0.05, 0.6)

        rms_inv = 1.0 - squash(rms_med, 0.02, 0.25)
        crest_inv = 1.0 - squash(crest_med, 2.0, 7.0)
        mod_inv = 1.0 - squash(mod, 1.0, 12.0)
        pulse_inv = 1.0 - squash(pulse, 0.2, 0.9)

        # Weights
        w_dyn = 0.20
        w_var = 0.15
        w_sil = 0.15
        w_rms = 0.15
        w_crest = 0.15
        w_mod = 0.10
        w_pulse = 0.10

        acousticness = (
            w_dyn*dyn_n +
            w_var*var_n +
            w_sil*sil_n +
            w_rms*rms_inv + 
            w_crest*crest_inv + 
            w_mod*mod_inv +
            w_pulse*pulse_inv
        )

        acousticness = float(np.clip(acousticness, 0.0, 1.0))
        self._cache_time["acousticness"] = acousticness
        return acousticness
    
    def danceability_time(self) -> float:
        if getattr(self, "invalid", False):
            return None
        if "danceability" in self._cache_time:
            return self._cache_time["danceability"]

        rms = self._amplitude()
        if rms.size == 0:
            self._cache_time["danceability"] = 0.0
            return 0.0

        pulse = self._pulse_clarity()
        silence = self._silence_ratio()
        mod = self._energy_mod_rate()
        var = self._energy_variance()
        onset = self._onset_env()
        _, _, tempo_stab = self._tempo_var()

        onset_events = np.sum(onset > np.mean(onset) + 1e-6)

        # Soft normalization instead of hard gate
        onset_n = squash(onset_events, 2, 20)
        pulse_n = squash(pulse, 0.1, 0.7)
        tempo_n = squash(tempo_stab, 0.05, 0.9)

        mod_low = squash(mod, 0.4, 3.0)
        mod_high = 1.0 - squash(mod, 3.0, 10.0)
        mod_n = mod_low * mod_high

        sil_inv = 1.0 - squash(silence, 0.05, 0.6)
        var_inv = 1.0 - squash(var, 0.5, 4.0)

        w_onset = 0.18
        w_pulse = 0.28
        w_tempo = 0.22
        w_mod   = 0.18
        w_sil   = 0.08
        w_var   = 0.06

        danceability = (
            w_onset*onset_n +
            w_pulse*pulse_n +
            w_tempo*tempo_n +
            w_mod*mod_n +
            w_sil*sil_inv +
            w_var*var_inv
        )

        danceability = float(np.clip(danceability, 0.0, 1.0))
        self._cache_time["danceability"] = danceability
        return danceability

    
    def tempo_time(self, bpm_min: float = 40.0, bpm_max: float = 220.0) -> float:
        """
        Estimate tempo from time-domain onset autocorrelation with harmonic voting.
        Adds perceptual bias toward ~100–130 BPM.
        """

        if getattr(self, "invalid", False):
            return None

        if "tempo" in self._cache_time:
            return self._cache_time["tempo"]

        onset = self._onset_env()
        if onset is None or onset.size < 8 or np.all(onset == 0):
            self._cache_time["tempo"] = 0.0
            return 0.0

        ac = self._onset_autocorr()
        if ac.size < 8:
            self._cache_time["tempo"] = 0.0
            return 0.0

        hop_time = self.H / float(self.sr)

        lags = np.arange(1, len(ac))
        ac_lag = ac[1:]

        bpms = 60.0 / (lags * hop_time + EPS)

        mask = (bpms >= bpm_min) & (bpms <= bpm_max)
        if not np.any(mask):
            self._cache_time["tempo"] = 0.0
            return 0.0

        bpms = bpms[mask]
        strengths = ac_lag[mask]

        strengths = strengths - np.min(strengths)
        strengths = strengths / (np.max(strengths) + EPS)

        votes = []

        for bpm, s in zip(bpms, strengths):
            cands = [bpm / 2.0, bpm, bpm * 2.0]
            for c in cands:
                if bpm_min <= c <= bpm_max:
                    votes.append((c, s))

        if not votes:
            self._cache_time["tempo"] = 0.0
            return 0.0

        # ------------------------------
        # perceptual weighting function
        # ------------------------------
        def perceptual_weight(bpm):
            # centered around 115 BPM, gentle falloff
            return np.exp(-((bpm - 115.0) ** 2) / (2 * (30.0 ** 2)))

        bins = {}
        for bpm, s in votes:
            b = int(round(bpm))
            w = s * perceptual_weight(b)
            bins.setdefault(b, 0.0)
            bins[b] += w

        best_bpm = float(max(bins, key=bins.get))

        neighbor_vals = []
        weights = []
        for k, v in bins.items():
            w = 1.0 / (1.0 + abs(k - best_bpm))
            neighbor_vals.append(k * w * v)
            weights.append(w * v)

        tempo_est = float(np.sum(neighbor_vals) / (np.sum(weights) + EPS))

        # confidence — soft
        pulse = self._pulse_clarity() or 0.0
        _, _, tempo_stab = self._tempo_var()

        clarity = np.max(strengths) - np.mean(strengths)

        clarity_n = squash(clarity, 0.01, 0.12)
        pulse_n   = squash(pulse, 0.10, 0.60)
        stab_n    = squash(tempo_stab, 0.15, 0.85)

        conf = 0.4 * clarity_n + 0.3 * pulse_n + 0.3 * stab_n

        if conf < 0.25:
            tempo_est = 0.6 * tempo_est + 0.4 * 110.0

        tempo_est = float(np.clip(tempo_est, bpm_min, bpm_max))

        self._cache_time["tempo"] = tempo_est
        return tempo_est

    def liveness_time(self) -> float:
        if getattr(self, "invalid", False):
            return None
        if "liveness" in self._cache_time:
            return self._cache_time["liveness"]
        
        # Feature extraction
        mod = self._energy_mod_rate()
        silence = self._silence_ratio()
        pulse = self._pulse_clarity()
        zcr = self._zero_crossing_rate()
        crest = self._crest_factor()

        # Aggregate statistics
        zcr_var = float(np.var(zcr)) if zcr.size else 0.0
        crest_var = float(np.var(crest)) if crest.size else 0.0

        # Normalize
        mod_n = squash(mod, 1.0, 12.0)
        zcr_var_n = np.clip(zcr_var/0.02, 0.0, 1.0)
        crest_var_n = np.clip(crest_var / 4.0, 0.0, 1.0)

        sil_inv = 1.0 - squash(silence, 0.05, 0.4)
        pulse_inv = 1.0 - squash(pulse, 0.2, 0.9)

        # Weights
        w_mod = 0.30
        w_zcr = 0.20
        w_crest = 0.20
        w_sil = 0.15
        w_pulse = 0.15

        liveness = (
            w_mod*mod_n +
            w_zcr*zcr_var_n +
            w_crest*crest_var_n +
            w_sil*sil_inv +
            w_pulse*pulse_inv
        )

        liveness = 1.0/(1.0 + np.exp(-6.0*(liveness - 0.35)))
        liveness = float(np.clip(liveness, 0.0, 1.0))
        self._cache_time["liveness"] = liveness

        return liveness
    
    def time_sig_time(self) -> int:
        if getattr(self, "invalid", False):
            return None
        """
        Heuristic time-signature estimator from time-domain rhythm structure.
        Falls back to 4 when confidence is weak.
        """
        if "time_signature" in self._cache_time:
            return self._cache_time["time_signature"]

        pulse = self._pulse_clarity()
        ac = self._onset_autocorr()

        # If rhythm is weak, default to 4/4
        if pulse < 0.15 or ac.size < 10:
            self._cache_time["time_signature"] = 4
            return 4

        tempo = self._tempo_var()[1]
        if tempo <= 0:
            self._cache_time["time_signature"] = 4
            return 4

        beat_period_sec = 60.0 / tempo
        frames_per_beat = int(round(beat_period_sec * self.sr / float(self.H)))

        if frames_per_beat <= 1:
            self._cache_time["time_signature"] = 4
            return 4

        candidates = [2, 3, 4, 5, 6, 7]
        scores = {}

        for m in candidates:
            lag = m * frames_per_beat
            if lag < len(ac):
                scores[m] = ac[lag]
            else:
                scores[m] = 0.0

        vals = np.array(list(scores.values()))

        # If no meaningful periodicity → fallback
        if np.max(vals) < 0.05:
            self._cache_time["time_signature"] = 4
            return 4

        # Bias toward common meters
        scores[4] *= 2.0
        scores[3] *= 1.2

        best = max(scores, key=scores.get)

        # Low confidence → fallback
        if best in (5, 6, 7) and scores[best] < 0.2:
            best = 4
        if scores[best] < 0.1:
            best = 4
        
        self._cache_time["time_signature"] = int(best)
        return int(best)


    # ---------- STFT-based features (cached) ----------
class STFTFeatures(AudioSignal):
    def __init__(self, sig: AudioSignal,
                 pad_mode: str = 'constant', center: bool = True):
        # super().__init__(audio_path, N, H)
        self.y = sig.y
        self.sr = sig.sr
        self.N = sig.N
        self.H = sig.H

        # Recompute fft freqs with final N
        self._fft_freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.N)

        # caches
        self._cache_spec = sig._cache
        # narrowband cache keyed by (f1,f2)
        self._nb_cache: Dict[Tuple[int, int], np.ndarray] = {}

        # compute STFT once, robustly
        try:
            self.X = librosa.stft(self.y, n_fft=self.N, hop_length=self.H,
                                  win_length=self.N, window='hann',
                                  center=center, pad_mode=pad_mode)
        except Exception as e:
            raise RuntimeError(f"STFT failed: {e}") from e

    # --- spectrogram helpers ---
    def _amp_spectrogram(self) -> np.ndarray:
        if "X_amp" in self._cache_spec:
            return self._cache_spec["X_amp"]
        S = np.abs(self.X)
        self._cache_spec["X_amp"] = S
        return S

    def _pow_spectrogram(self) -> np.ndarray:
        if "X_pow" in self._cache_spec:
            return self._cache_spec["X_pow"]
        S = self._amp_spectrogram()
        P = S**2
        self._cache_spec["X_pow"] = P
        return P

    def _stft_energy(self) -> np.ndarray:
        if "stft_energy" in self._cache_spec:
            return self._cache_spec["stft_energy"]
        P = self._pow_spectrogram()
        E = np.sum(P, axis=0)
        self._cache_spec["stft_energy"] = E
        return E

    def _stft_dB_spec(self) -> np.ndarray:
        if "S_dB" in self._cache_spec:
            return self._cache_spec["S_dB"]
        S = self._amp_spectrogram()
        S_db = librosa.amplitude_to_db(S, ref=np.max)
        self._cache_spec["S_dB"] = S_db
        return S_db

    def _onset_env(self) -> np.ndarray:
        if "onset_env_stft" in self._cache_spec:
            return self._cache_spec["onset_env_stft"]
        S = self._amp_spectrogram()
        if S.shape[1] < 2:
            env = np.zeros(S.shape[1], dtype=float)
            self._cache_spec["onset_env_stft"] = env
            return env
        # spectral flux (only positive increases)
        flux = np.sqrt(np.sum(np.maximum(0.0, S[:, 1:] - S[:, :-1])**2, axis=0))
        if flux.size == 0 or np.max(flux) <= 0:
            env = flux
        else:
            env = flux / (np.max(flux) + EPS)

        env = np.pad(env, (1, 0), mode='constant')
        self._cache_spec["onset_env_stft"] = env
        return env

    def _spectral_centroid(self) -> np.ndarray:
        if "centroid" in self._cache_spec:
            return self._cache_spec["centroid"]
        S = self._amp_spectrogram()
        freqs = self._fft_freqs[:S.shape[0]]
        denom = np.sum(S, axis=0)
        num = np.sum(S * freqs[:, None], axis=0)
        # safe division
        centroid = np.divide(num, denom, out=np.zeros_like(num), where=denom > 0)
        self._cache_spec["centroid"] = centroid
        return centroid

    def _spectral_bandwidth(self) -> np.ndarray:
        if "bandwidth" in self._cache_spec:
            return self._cache_spec["bandwidth"]
        S = self._amp_spectrogram()
        freqs = self._fft_freqs[:S.shape[0]]
        centroid = self._spectral_centroid()
        # (freqs[:,None] - centroid)**2 * S
        dif = freqs[:, None] - centroid[None, :]
        num = np.sum((dif**2) * S, axis=0)
        denom = np.sum(S, axis=0)
        bw = np.sqrt(np.divide(num, denom, out=np.zeros_like(num), where=denom > 0))
        self._cache_spec["bandwidth"] = bw
        return bw

    def _spectral_skewness(self) -> np.ndarray:
        if "skewness" in self._cache_spec:
            return self._cache_spec["skewness"]
        S = self._amp_spectrogram()
        freqs = self._fft_freqs[:S.shape[0]]
        centroid = self._spectral_centroid()
        bw = self._spectral_bandwidth()
        dif = freqs[:, None] - centroid[None, :]
        num = np.sum((dif**3) * S, axis=0)
        denom = (bw**3) * np.sum(S, axis=0)
        skew = np.divide(num, denom + EPS, out=np.zeros_like(num), where=(denom + EPS) > 0)
        self._cache_spec["skewness"] = skew
        return skew

    def _spectral_kurtosis(self) -> np.ndarray:
        if "kurtosis" in self._cache_spec:
            return self._cache_spec["kurtosis"]
        S = self._amp_spectrogram()
        freqs = self._fft_freqs[:S.shape[0]]
        centroid = self._spectral_centroid()
        bw = self._spectral_bandwidth()
        dif = freqs[:, None] - centroid[None, :]
        num = np.sum((dif**4) * S, axis=0)
        denom = (bw**4) * np.sum(S, axis=0)
        kurt = np.divide(num, denom + EPS, out=np.zeros_like(num), where=(denom + EPS) > 0)
        self._cache_spec["kurtosis"] = kurt
        return kurt

    def _spectral_flatness(self) -> np.ndarray:
        if "flatness" in self._cache_spec:
            return self._cache_spec["flatness"]
        P = self._pow_spectrogram()
        # geometric mean / arithmetic mean
        geo = np.exp(np.mean(np.log(np.maximum(P, 1e-18)), axis=0))
        arith = np.mean(P, axis=0)
        flatness = geo / (arith + EPS)
        self._cache_spec["flatness"] = flatness
        return flatness

    def _spectral_rolloff(self, roll_percent: float = 0.85) -> np.ndarray:
        key = f"rolloff_{roll_percent}"
        if key in self._cache_spec:
            return self._cache_spec[key]

        P = self._pow_spectrogram()
        freqs = self._fft_freqs[:P.shape[0]]

        # cumulative energy per frame
        cum_energy = np.cumsum(P, axis=0)
        total_energy = cum_energy[-1, :]

        rolloff = np.zeros(P.shape[1], dtype=float)

        for t in range(P.shape[1]):
            if total_energy[t] <= EPS:
                rolloff[t] = 0.0
                continue
            threshold = roll_percent * total_energy[t]
            idx = np.searchsorted(cum_energy[:, t], threshold)
            idx = min(idx, len(freqs) - 1)
            rolloff[t] = freqs[idx]

        self._cache_spec[key] = rolloff
        return rolloff

    def _band_ratio(self, f_low=500, f_mid=2000, f_high=8000) -> np.ndarray:
        key = f"bandratio_{f_low}_{f_mid}_{f_high}"
        if key in self._cache_spec:
            return self._cache_spec[key]
        S = self._amp_spectrogram()
        freqs = self._fft_freqs[:S.shape[0]]
        low_idx = np.where((freqs >= 0) & (freqs < f_low))[0]
        high_idx = np.where((freqs >= f_mid) & (freqs < f_high))[0]
        E_low = np.sum(S[low_idx, :]**2, axis=0) if low_idx.size else np.zeros(S.shape[1])
        E_high = np.sum(S[high_idx, :]**2, axis=0) if high_idx.size else np.zeros(S.shape[1])
        ratio = np.divide(E_high, E_low + EPS)
        self._cache_spec[key] = ratio
        return ratio

    def _spectral_entropy(self) -> np.ndarray:
        if "entropy" in self._cache_spec:
            return self._cache_spec["entropy"]
        P = self._pow_spectrogram()
        denom = np.sum(P, axis=0, keepdims=True)
        p = np.divide(P, denom + EPS, out=np.zeros_like(P), where=denom > 0)
        p = np.clip(p, 1e-12, 1.0)
        ent = -np.sum(p * np.log2(p), axis=0)
        ent = ent / np.log2(P.shape[0] + EPS)
        self._cache_spec["entropy"] = ent
        return ent

    def _spectral_slope(self) -> np.ndarray:
        if "slope" in self._cache_spec:
            return self._cache_spec["slope"]
        S = np.abs(self.X)
        S_db = 20 * np.log10(np.maximum(S, 1e-12))
        freqs = self._fft_freqs[:S.shape[0]]
        f_mean = np.mean(freqs)
        X_mean = np.mean(S_db, axis=0)
        num = np.sum((freqs[:, None] - f_mean) * (S_db - X_mean[None, :]), axis=0)
        den = np.sum((freqs - f_mean)**2)
        slope = num / (den + EPS)
        self._cache_spec["slope"] = slope
        return slope

    def _narrowband_energy(self, f1: int = 300, f2: int = 3000) -> np.ndarray:
        key = (int(f1), int(f2))
        if key in self._nb_cache:
            return self._nb_cache[key]
        P = self._pow_spectrogram()
        freqs = self._fft_freqs[:P.shape[0]]
        band_idx = np.where((freqs >= f1) & (freqs <= f2))[0]
        if band_idx.size == 0:
            val = np.zeros(P.shape[1])
        else:
            val = np.sum(P[band_idx, :], axis=0)
        self._nb_cache[key] = val
        return val

    # --- Harmonic / Inharmonicity / Harmonic ratio ---
    def _harmonic_peak_tracking(self, max_harmonics: int = 15, threshold_rel: float = 0.05):
        if "harmonic_peaks" in self._cache_spec:
            return self._cache_spec["harmonic_peaks"]
        X_amp = self._amp_spectrogram()
        freqs = self._fft_freqs[:X_amp.shape[0]]
        # use librosa.yin with hop = self.H (it returns array frames aligned to STFT frames if hop is same)
        try:
            f0 = librosa.yin(self.y, fmin=50, fmax=2000, sr=self.sr, hop_length=self.H)
        except Exception:
            f0 = np.full(X_amp.shape[1], np.nan)
        harmonics = []
        for t in range(X_amp.shape[1]):
            mag = X_amp[:, t]
            if np.max(mag) <= 0:
                harmonics.append([])
                continue
            peaks, props = find_peaks(mag, height=np.max(mag) * threshold_rel)
            f0_t = f0[t] if t < f0.shape[0] else np.nan
            if np.isnan(f0_t) or f0_t <= 0 or peaks.size == 0:
                harmonics.append([])
                continue
            frame_h = []
            for k in range(1, max_harmonics + 1):
                target = k * f0_t
                if target >= self.sr / 2:
                    break
                # tolerance: 50 cents ~ 1.03 factor, use Hz window
                tol = max(20.0, 0.05 * target)
                candidates = peaks[np.abs(freqs[peaks] - target) <= tol]
                if candidates.size == 0:
                    continue
                best = candidates[np.argmax(mag[candidates])]
                frame_h.append((k, freqs[best], float(mag[best])))
            harmonics.append(frame_h)
        self._cache_spec["harmonic_peaks"] = harmonics
        return harmonics

    def _harmonic_ratio(self) -> np.ndarray:
        if "harmonic_ratio" in self._cache_spec:
            return self._cache_spec["harmonic_ratio"]
        X_mag = self._amp_spectrogram()
        # robust f0 track
        try:
            f0_track = librosa.yin(self.y, fmin=50, fmax=2000, sr=self.sr, hop_length=self.H)
        except Exception:
            f0_track = np.full(X_mag.shape[1], np.nan)
        HR = np.zeros(X_mag.shape[1], dtype=float)
        bin_freq = float(self.sr) / float(self.N)
        for i in range(X_mag.shape[1]):
            mag = X_mag[:, i]
            E_total = np.sum(mag**2)
            if E_total <= 0:
                HR[i] = 0.0
                continue
            f0 = f0_track[i] if i < f0_track.shape[0] else np.nan
            if np.isnan(f0) or f0 <= 0:
                HR[i] = 0.0
                continue
            harmonics = []
            h = 1
            while True:
                fh = h * f0
                if fh >= self.sr / 2:
                    break
                harmonics.append(fh)
                h += 1
            if len(harmonics) == 0:
                HR[i] = 0.0
                continue
            harmonic_bins = np.round(np.array(harmonics) * self.N / self.sr).astype(int)
            E_harm = 0.0
            bw = 2
            for k in harmonic_bins:
                k_min = max(0, k - bw)
                k_max = min(len(mag) - 1, k + bw)
                E_harm += np.sum(mag[k_min:k_max + 1]**2)
            HR[i] = float(E_harm / (E_total + EPS))
        self._cache_spec["harmonic_ratio"] = HR
        return HR

    def _inharmonicity(self, max_harmonics: int = 20, search_bins: int = 3) -> np.ndarray:
        if "inharmonicity" in self._cache_spec:
            return self._cache_spec["inharmonicity"]
        X_mag = self._amp_spectrogram()
        try:
            f0_track = librosa.yin(self.y, fmin=50, fmax=2000, sr=self.sr, hop_length=self.H)
        except Exception:
            f0_track = np.full(X_mag.shape[1], np.nan)
        inharm = np.zeros(X_mag.shape[1], dtype=float)
        bin_freq = float(self.sr) / float(self.N)
        for i in range(X_mag.shape[1]):
            mag = X_mag[:, i]
            f0 = f0_track[i] if i < f0_track.shape[0] else np.nan
            if np.isnan(f0) or f0 <= 0:
                inharm[i] = 0.0
                continue
            harmonic_freqs = []
            for h in range(1, max_harmonics + 1):
                fh = h * f0
                if fh >= self.sr / 2:
                    break
                harmonic_freqs.append(fh)
            deviations = []
            weights = []
            for fh_ideal in harmonic_freqs:
                k_ideal = int(round(fh_ideal / bin_freq))
                k_min = max(0, k_ideal - search_bins)
                k_max = min(len(mag) - 1, k_ideal + search_bins)
                local = mag[k_min:k_max + 1]
                if local.size == 0:
                    continue
                peak_idx = np.argmax(local)
                k_peak = k_min + peak_idx
                f_peak = k_peak * bin_freq
                deviation = (f_peak - fh_ideal) ** 2
                deviations.append(deviation)
                weights.append(float(local[peak_idx]))
            if len(weights) == 0:
                inharm[i] = 0.0
            else:
                inharm[i] = float(np.sum(np.array(deviations) * np.array(weights)) / (np.sum(weights) + EPS))
        self._cache_spec["inharmonicity"] = inharm
        return inharm

    # instantaneous frequency (phase-based)
    def _instantaneous_freq(self) -> np.ndarray:
        if "inst_freq" in self._cache_spec:
            return self._cache_spec["inst_freq"]
        phase = np.unwrap(np.angle(self.X), axis=1)
        dphase = np.diff(phase, axis=1)
        k = np.arange(self.X.shape[0])[:, None]
        expected = 2.0 * np.pi * k * (self.H / float(self.N))
        delta = dphase - expected
        delta = (delta + np.pi) % (2.0 * np.pi) - np.pi
        inst_freq = (expected + delta) * (self.sr / (2.0 * np.pi * self.H))
        inst_freq = np.hstack([inst_freq[:, :1], inst_freq])
        self._cache_spec["inst_freq"] = inst_freq
        return inst_freq

    def _group_delay(self) -> np.ndarray:
        if "group_delay" in self._cache_spec:
            return self._cache_spec["group_delay"]
        phase = np.unwrap(np.angle(self.X), axis=0)
        dphi = np.diff(phase, axis=0)
        # group delay = -d(phi)/d(omega); approximate scaling factor
        GD = -dphi / (2.0 * np.pi / float(self.N))
        GD = np.vstack([np.zeros((1, GD.shape[1])), GD, np.zeros((1, GD.shape[1]))])
        self._cache_spec["group_delay"] = GD
        return GD
    
    def _onset_periodicity(self):
        flux = self._onset_env()
        if flux.size < 8:
            return 0.0
        
        ac = np.correlate(flux - np.mean(flux), flux - np.mean(flux), mode='full')
        ac = ac[len(ac)//2 + 1:]

        if np.max(ac) <= 0:
            return 0.0
        
        return float(np.max(ac)/(np.sum(ac) + EPS))
    
    def _rhythm_confidence(self) -> float:
        onset = self._onset_env()

        # Hard guards
        if onset is None:
            return 0.0

        onset = np.asarray(onset).flatten()

        if onset.size < 8:
            return 0.0

        if np.allclose(onset, 0.0):
            return 0.0

        # Remove DC
        onset_z = onset - np.mean(onset)

        # Autocorrelation
        ac = np.correlate(onset_z, onset_z, mode="full")
        ac = ac[len(ac)//2:]

        if ac.ndim != 1 or ac.size < 3:
            return 0.0

        ac0 = ac[0] + EPS
        ac_peaks = ac[1:]

        if np.max(ac_peaks) <= 0:
            return 0.0

        peak_ratio = np.max(ac_peaks) / ac0

        onset_energy = np.mean(np.abs(onset))

        # Final confidence
        conf = onset_energy * peak_ratio
        return float(np.clip(conf, 0.0, 1.0))

        
    def _harmonic_confidence(self) -> float:
        hr = self._harmonic_ratio()
        flat = self._spectral_flatness()

        if hr.size == 0:
            return 0.0
        
        hr_m = np.median(hr)
        flat_m = np.median(flat)

        conf = hr_m*(1.0 - flat_m)
        return float(np.clip(conf, 0.0, 1.0))

    # ---------- High-level perceptual features ----------
    # All three of these use robust normalization and are safe to call on silence
    def loudness_stft(self) -> float:
        if getattr(self, "invalid", False):
            return 0.0
        if "loudness_stft" in self._cache_spec:
            return self._cache_spec["loudness_stft"]
        
        # Get Power Spectrum
        P = self._pow_spectrogram()
        if P.size == 0:
            self._cache_spec["loudness_stft"] = -np.inf
            return -np.inf
        
        freqs = self._fft_freqs[:P.shape[0]]

        # Build K-weighting curve
        f2 = freqs**2
        hp = (f2/(f2 + 129.4**2))
        shelf = ((f2 + 107.7**2)/(f2 + 737.9**2))

        K = hp*shelf
        K = np.sqrt(K) # amplitude scaling
        K = K[:, None]

        # Apply K-weighting
        P_weighted = (P*(K**2))

        # Frame energy (sum across freq)
        frame_energy = np.sum(P_weighted, axis=0) + EPS

        # Convert to LUFS
        lufs_frames = -0.691 + 10.0*np.log10(frame_energy)

        # ITU-style gating
        gate = lufs_frames[lufs_frames > -70.0]
        if gate.size == 0:
            self._cache_spec["loudness_stft"] = -np.inf
            return -np.inf
        
        # Relative gate: remove frames 10 LU below mean
        mean_gate = np.mean(gate)
        gate_final = gate[gate > (mean_gate - 10.0)]
        if gate_final.size == 0:
            loudness = float(mean_gate)
        else:
            loudness = float(np.mean(gate_final))

        self._cache_spec["loudness_stft"] = loudness
        return loudness
    
    def energy_stft(self) -> float:
        if getattr(self, "invalid", False):
            return 0.0
        
        if "energy_stft" in self._cache_spec:
            return self._cache_spec["energy_stft"]
        
        # Extract features
        E = self._stft_energy()
        flux = self._onset_env()
        flat = self._spectral_flatness()
        ratio = self._band_ratio()

        # Normalize energy to 0-1
        E_log = np.log10(E + EPS)
        E_min = np.percentile(E_log, 5)
        E_max = np.percentile(E_log, 95)
        E_norm = np.clip((E_log - E_min)/(E_max - E_min + EPS), 0.0, 1.0)

        # Normalize Band Ratio
        ratio_norm = np.tanh(ratio)

        # Weights
        w_E = 0.45
        w_flux = 0.25
        w_flat = 0.15
        w_ratio = 0.15

        energy_frame = (
            w_E*E_norm +
            w_flux*flux +
            w_flat*flat +
            w_ratio*ratio_norm
        )

        # Smooth slightly 
        smooth = np.convolve(energy_frame, np.ones(5)/5.0, mode='same')

        # Final Scalar
        energy = float(np.mean(smooth))

        self._cache_spec["energy_stft"] = energy
        return energy
    
    def speechiness_stft(self) -> float:
        if getattr(self, "invalid", False):
            return 0.0
        if "speechiness_stft" in self._cache_spec:
            return self._cache_spec["speechiness_stft"]
        
        # Feature extraction
        E = self._stft_energy()
        band_tel = self._narrowband_energy(300, 3400)
        HR = self._harmonic_ratio()
        flat = self._spectral_flatness()
        ratio = self._band_ratio()
        flux = self._onset_env()
        periodicity = self._onset_periodicity()
        centroid = self._spectral_centroid()
        bandwidth = self._spectral_bandwidth()

        # Normalize Components
        E_log = np.log10(E + EPS)
        E_min, E_max = np.percentile(E_log, [5, 95])
        E_norm = np.clip((E_log - E_min)/(E_max - E_min + EPS), 0.0, 1.0)

        # Telephone band dominance
        tel_ratio = band_tel/(np.mean(E) + EPS)
        tel_ratio = np.clip(np.tanh(tel_ratio), 0.0, 1.0)

        # Low harmonic ratio -> more speech-like
        HR_inv = 1.0 - np.clip(HR, 0.0, 1.0)

        # Flatness -> speech is mid-flat
        flat_centered = 1.0 - np.abs(flat - 0.45)/0.45
        flat_centered = np.clip(flat_centered, 0.0, 1.0)

        # Penalty noise
        noise_penalty = np.clip(np.tanh(ratio), 0.0, 1.0)
        flat_centered *= (1.0 - noise_penalty)

        # Avoid giving credit to bright/bass-heavy
        brightness_penalty = np.clip(np.tanh(ratio), 0.0, 1.0)

        # Modulation gate
        flux_var = np.var(flux)
        flux_score = np.exp(-((flux_var - 0.02)/0.03)**2)
        flux_score = np.clip(flux_score, 0.0, 1.0)

        periodicity_penalty = np.clip(periodicity/0.3, 0.0, 1.0)
        speech_gate = 1.0 - periodicity_penalty

        formant_score = np.clip(bandwidth/(centroid + EPS), 0.0, 1.0)

        # Pitch Stability Gates
        f0 = librosa.yin(self.y, fmin=50, fmax=400, sr=self.sr, hop_length=self.H)
        f0 = f0[~np.isnan(f0)]

        if len(f0) > 5:
            pitch_std = np.std(f0)/(np.mean(f0) + EPS)
        else:
            pitch_std = 0.0

        pitch_gate = np.clip(pitch_std/0.15, 0.0, 1.0)

        # Temporal Flux Variability Gate
        flux_std = np.std(flux)
        flux_gate = np.clip(flux_std/0.2, 0.0, 1.0)

        # Formant Multiplicity Gate
        S = self._amp_spectrogram()
        freqs = self._fft_freqs

        band = np.where((freqs >= 300) & (freqs <= 3400))[0]
        peak_counts = []

        for t in range(S.shape[1]):
            peaks, _ = find_peaks(S[band, t], height=np.max(S[band, t])*0.1)
            peak_counts.append(len(peaks))

        formant_gate = np.clip(np.mean(peak_counts)/3.0, 0.0, 1.0)

        # Weights
        w_tel = 0.35
        w_flux = 0.25
        w_formant = 0.20
        w_HR = 0.20
        w_flat = 0.15
        w_E = 0.05
        w_gate = 0.05

        speechiness = (
            w_tel*tel_ratio +
            w_flux*flux_score +
            w_flat*flat_centered +
            w_HR*HR_inv +
            w_E*(1.0 - E_norm)
        )

        # Smooth and summarize
        speechiness = speechiness*pitch_gate*flux_gate*formant_gate
        speechiness = float(np.clip(np.mean(speechiness), 0.0, 1.0))
        speechiness = np.clip((speechiness - 0.15)/0.35, 0.0, 1.0)

        self._cache_spec["speechiness_stft"] = speechiness
        return speechiness
    
    def acousticness_stft(self) -> float:
        if getattr(self, "invalid", False):
            return 0.0
        if "acousticness_stft" in self._cache_spec:
            return self._cache_spec["acousticness_stft"]
        
        hr_conf = self._harmonic_confidence()
        flat = np.median(self._spectral_flatness())
        centroid = np.median(self._spectral_centroid())
        bw = np.median(self._spectral_bandwidth())

        # Natural timbre heuristic
        spectral_natural = (
            (1.0 - flat) *
            np.exp(-centroid / 5000.0) *
            np.exp(-bw / 4000.0)
        )

        acousticness = hr_conf * spectral_natural
        acousticness = float(np.clip(acousticness, 0.0, 1.0))
        self._cache_spec["acousticness_stft"] = acousticness
        return acousticness
    
    def danceability_stft(self) -> float:
        if getattr(self, "invalid", False):
            return 0.0
        if "danceability_stft" in self._cache_spec:
            return self._cache_spec["danceability_stft"]
        
        # Feature extraction
        onset = self._onset_env()
        if onset.size < 8 or np.all(onset == 0):
            self._cache_spec["danceability_stft"] = 0.05
            return 0.05
        
        # Pulse Clarity
        ac = np.correlate(onset, onset, mode='full')
        ac = ac[len(ac)//2:]
        if ac.size < 4:
            self._cache_spec["danceability_stft"] = 0.05
            return 0.05
        
        ac = ac/(np.max(ac) + EPS)

        main_peak = np.max(ac[1:])
        mean_rest = np.mean(ac[1:])
        pulse_clarity = np.clip((main_peak - mean_rest)/(main_peak + EPS), 0.0, 1.0)

        # Beat periodicity stability
        lags = np.arange(1, len(ac))
        hop_time = self.H/float(self.sr)
        bpms = 60.0/(lags*hop_time + EPS)

        mask = (bpms >= 60) & (bpms <= 180)
        if np.any(mask):
            periodicity = np.mean(ac[1:][mask])
        else:
            periodicity = np.mean(ac[1:])

        periodicity = np.clip(periodicity, 0.0, 1.0)

        # Spectral balance/density
        flat = self._spectral_flatness()
        band = self._band_ratio()
        energy = self._stft_energy()

        # too flat = less danceable
        flat_n = 1.0 - np.clip(flat/0.8, 0.0, 1.0)

        # prefer controlled brightness
        bright = np.clip(np.tanh(band), 0.0, 1.0)
        bright_pref = 1.0 - bright

        # Normalize dynamics
        E_log = np.log10(energy + EPS)
        E_min, E_max = np.percentile(E_log, [5, 95])
        dyn = 1.0 - np.clip((E_log - E_min)/(E_max - E_min + EPS), 0.0, 1.0)

        # Weights
        w_pulse = 0.35
        w_period = 0.25
        w_flat = 0.15
        w_bright = 0.15
        w_dyn = 0.10

        dance_frame = (
            w_pulse*pulse_clarity +
            w_period*periodicity +
            w_flat*flat_n +
            w_bright*bright_pref +
            w_dyn*np.mean(dyn)
        )

        # Smooth and summarize
        smooth = np.convolve(dance_frame, np.ones(7)/7.0, mode='same')
        danceability = float(np.mean(smooth))
        
        conf = self._rhythm_confidence()
        danceability = confidence_weighted(danceability, conf, neutral=0.3)

        danceability = float(np.clip(danceability, 0.0, 1.0))
        self._cache_spec["danceability_stft"] = danceability
        return danceability
    
    def valence_stft(self) -> float:
        if getattr(self, "invalid", False):
            return 0.0
        if "valence_stft" in self._cache_spec:
            return self._cache_spec["valence_stft"]
        
        S = self._amp_spectrogram()
        if S.size == 0:
            self._cache_spec["valence_stft"] = 0.5
            return 0.5
        
        # Confidence (soft gating)
        harm_conf = self._harmonic_confidence()
        rhy_conf = self._rhythm_confidence()
        confidence = np.clip(0.6*harm_conf + 0.4*rhy_conf, 0.0, 1.0)
        
        # Feature extraction
        centroid = self._spectral_centroid()
        flat = self._spectral_flatness()
        HR = self._harmonic_ratio()
        slope = self._spectral_slope()
        band = self._band_ratio()

        # Normalize Components
        nyq = self.sr/2.0
        bright = np.clip(centroid/(0.6*nyq), 0.0, 1.0)

        harm = np.clip(HR/0.4, 0.0, 1.0)

        noise_penalty = np.clip(flat/0.7, 0.0, 1.0)
        noise_inv = 1.0 - 0.6*noise_penalty

        slope_n = np.tanh(0.002*slope)
        slope_pref = 1.0 - np.clip(np.abs(slope_n), 0.0, 1.0)

        band_n = np.clip(np.tanh(band), 0.0, 1.0)
        brightness_balance = 1.0 - np.clip(band_n*0.9, 0.0, 1.0)

        # Weights
        w_bright = 0.30
        w_hr = 0.22
        w_noise = 0.18
        w_slope = 0.18
        w_bal = 0.12

        valence = (
            w_bright*np.mean(bright) +
            w_hr*np.mean(harm) + 
            w_noise*np.mean(noise_inv) + 
            w_slope*np.mean(slope_pref) +
            w_bal*np.mean(brightness_balance)
        )

        valence = np.clip(valence, 0.0, 1.0)

        # Confidence-aware blending
        valence = confidence*valence + (1.0 - confidence)*0.5

        valence = float(np.clip(valence, 0.0, 1.0))
        self._cache_spec["valence_stft"] = valence
        return valence 
    
    def tempo_stft(self, bpm_min: float = 40.0, bpm_max: float = 220.0, bias_around: float = 120.0) -> float:
        if getattr(self, "invalid", False):
            return 0.0
        if "tempo_stft" in self._cache_spec:
            return self._cache_spec["tempo_stft"]
        
        onset = self._onset_env()
        if onset.size < 4:
            self._cache_spec["tempo_stft"] = 0.0
            return 0.0
        
        rhythm_conf = self._rhythm_confidence()
        if rhythm_conf < 0.15:
            self._cache_spec["tempo_stft"] = 0.0
            return 0.0

        # Normalize and smooth 
        onset - np.min(onset)
        if np.max(onset) > 0:
            onset = onset/(np.max(onset) + EPS)

        win = 5
        onset_smooth = np.convolve(onset, np.ones(win)/win, mode="same")

        # Autocorrelation
        ac = np.correlate(onset_smooth, onset_smooth, mode="full")
        ac = ac[ac.size//2:]
        ac = ac[1:]

        # Convert lags -> BPM grid
        lags = np.arange(1, len(ac) + 1)
        hop_time = self.H/float(self.sr)
        bpm = 60.0/(lags*hop_time + EPS)

        # Keep BPM inside constraints
        valid = (bpm >= bpm_min) & (bpm <= bpm_max)
        if not np.any(valid):
            self._cache_spec["tempo_stft"] = 0.0
            return 0.0
        
        bpm = bpm[valid]
        ac = ac[valid]

        # Bias function: center around "typical" tempos
        bias = np.exp(-0.5*((bpm - bias_around)/40.0)**2)
        score = ac*bias

        best = bpm[np.argmax(score)]

        # Snap tempo into perceptually plausible pocket near bias_around
        while best > 1.5*bias_around:
            best /= 2.0
        while best < 0.75*bias_around:
            best *= 2.0

        tempo = float(np.clip(best, bpm_min, bpm_max))
        self._cache_spec["tempo_stft"] = tempo
        return tempo 
    
    def liveness_stft(self) -> float:
        if getattr(self, "invalid", False):
            return 0.0
        if "liveness_stft" in self._cache_spec:
            return self._cache_spec["liveness_stft"]
        
        # Feature Extraction
        flat = self._spectral_flatness()
        entr = self._spectral_entropy()
        inh = self._inharmonicity()
        roll = self._spectral_rolloff(0.90)
        energy = self._stft_energy()
        onset = self._onset_env()

        # Align
        L = min(len(flat), len(entr), len(inh), len(roll), len(energy), len(onset))
        if L == 0:
            self._cache_spec["liveness_stft"] = 0.0
            return 0.0
        
        flat = flat[:L]
        entr = entr[:L]
        inh = inh[:L]
        roll = roll[:L]
        energy = energy[:L]
        onset = onset[:L]

        # Dynamic looseness: variance across frames
        dyn_var = np.var(robust_normalize(energy))

        # Onset Irregularity
        onset_diff = np.abs(np.diff(onset, prepend=onset[0]))
        onset_irreg = np.mean(robust_normalize(onset_diff))

        # Normalize 
        flat_n = robust_normalize(flat)
        entr_n = robust_normalize(entr)
        inh_n = robust_normalize(inh)
        roll_n = robust_normalize(roll)

        # weights (tuned heuristically)
        w_flat = 0.30
        w_ent  = 0.20
        w_inh  = 0.15
        w_roll = 0.10
        w_dyn  = 0.15
        w_ir   = 0.15

        frame_score = (w_flat * flat_n +
                    w_ent  * entr_n +
                    w_inh  * inh_n +
                    w_roll * roll_n)

        track_score = (0.7 * float(np.median(frame_score)) +
                    w_dyn * float(np.clip(dyn_var, 0.0, 1.0)) +
                    w_ir * float(np.clip(onset_irreg, 0.0, 1.0)))

        score = safe_clip01(track_score)

        self._cache_spec["liveness_stft"] = score
        return score
    
    def instrumentalness_stft(self) -> float:
        if getattr(self, "invalid", False):
            return 0.0
        if "instrumentalness_stft" in self._cache_spec:
            return self._cache_spec["instrumentalness_stft"]

        # ---------- Feature Extraction ----------
        hr = self._harmonic_ratio()
        flat = self._spectral_flatness()
        entr = self._spectral_entropy()
        onset = self._onset_env()
        inst_freq = self._instantaneous_freq()

        # ---------- Align ----------
        L = min(len(hr), len(flat), len(entr), len(onset), inst_freq.shape[1])
        if L == 0:
            return 0.0

        hr = np.nan_to_num(hr[:L])
        flat = flat[:L]
        entr = entr[:L]
        onset = onset[:L]
        inst_freq = inst_freq[:, :L]

        # ---------- Normalize ----------
        hr_n = robust_normalize(hr)
        flat_n = robust_normalize(flat)
        entr_n = robust_normalize(entr)

        # Pitch stability (instruments > speech)
        pitch_var = np.std(inst_freq, axis=0)
        pitch_n = robust_normalize(pitch_var)

        # Onset regularity
        onset_diff = np.abs(np.diff(onset, prepend=onset[0]))
        onset_reg = 1.0 - robust_normalize(onset_diff)

        # ---------- Vocal Suppression ----------
        speech = self.speechiness_stft()
        speech_gate = np.clip(1.0 - (speech / 0.25)**2, 0.0, 1.0)

        # ---------- Instrumental Structure ----------
        I_struct = (
            0.4 * hr_n +
            0.3 * (1.0 - pitch_n) +
            0.3 * onset_reg
        )
        I_struct = np.clip(I_struct, 0.0, 1.0)

        # ---------- Noise / Percussive Instrumental ----------
        I_noise = (
            0.5 * flat_n +
            0.5 * entr_n
        )
        I_noise = np.clip(I_noise, 0.0, 1.0)

        # ---------- Fuse ----------
        inst_struct_score = np.median(I_struct * speech_gate)
        inst_noise_score  = np.median(I_noise  * speech_gate)

        instrumentalness = max(inst_struct_score, inst_noise_score)
        instrumentalness = safe_clip01(float(instrumentalness))

        if speech < 0.05:
            instrumentalness = 1.0 - (1.0 - instrumentalness)**1.7

        self._cache_spec["instrumentalness_stft"] = instrumentalness
        return instrumentalness

    def time_sig_freq(self) -> int:
        if getattr(self, "invalid", False):
            return 0.0
        if "time_signature_stft" in self._cache_spec:
            return self._cache_spec["time_signature_stft"]
        
        meters=[2, 3, 4, 5, 6, 7]
        
        onset = self._onset_env()
        if onset.size < 16 or np.all(onset == 0):
            self._cache_spec["time_signature_stft"] = 4
            return 4
        
        # Beat Tracking (frame domain)
        try:
            tempo, beats = librosa.beat.beat_track(onset_envelope=onset,
                                                   sr=self.sr,
                                                   hop_length=self.H,
                                                   tightness=80)
        except Exception:
            self._cache_spec["time_signature_stft"] = 4
            return 4
        
        if beats.size < 6:
            self._cache_spec["time_signature_stft"] = 4
            return 4
        
        beats = beats.astype(int)
        beats = beats[beats < len(onset)]

        # Beat Strength Sequence
        beat_strength = onset[beats]

        if beat_strength.size < 6:
            self._cache_spec["time_signature_stft"] = 4
            return 4
        
        # Normalize
        if np.max(beat_strength) > 0:
            beat_strength = beat_strength/(np.max(beat_strength) + EPS)

        # Score each meter
        best_m = 4
        best_score = -1.0

        for m in meters:
            if len(beat_strength) < 2*m:
                continue

            # reshape into bars with length m (truncate excess)
            L = (len(beat_strength)//m)*m
            seg = beat_strength[:L].reshape(-1, m)

            # Mean Bar Profile
            proto = np.mean(seg, axis=0)

            # Normalize Prototype
            if np.max(proto) > 0:
                proto = proto/(np.max(proto) + EPS)

            # Autocorrelation of prototype - periodic clarity
            ac = np.correlate(proto, proto, mode='full')
            ac = ac[len(ac)//2:]

            # Ignore lag 0
            if ac.size > 1:
                ac = ac/(np.max(ac) + EPS)
                clarity = float(np.max(ac[1:]))
            else:
                clarity = 0.0

            # Penalty for meters that rarely fit popular music
            penalty = 1.0
            if m in (5, 7):
                penalty = 0.9

            score = clarity*penalty
            if score > best_score:
                best_score = score
                best_m = m

            self._cache_spec["time_signature_stft"] = int(best_m)
            return int(best_m)