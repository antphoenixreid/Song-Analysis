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
class TimeFeatures(AudioSignal):
    def __init__(self, audio_path: str, N: int = 2048, H: int = 512):
        super().__init__(audio_path, N, H)

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
        if "global_loudness_dB" in self._cache:
            return self._cache["global_loudness_dB"]
        
        rms = np.sqrt(np.mean(self.y**2)) + EPS
        loud_db = 20.0*np.log10(rms)

        self._cache["global_loudness_dB"] = loud_db
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
        if "amplitude" in self._cache:
            return self._cache["amplitude"]
        rms = librosa.feature.rms(y=self.y, frame_length=self.N, hop_length=self.H)[0]
        self._cache["amplitude"] = rms
        return rms

    def _amplitude_dB(self) -> np.ndarray:
        if getattr(self, "invalid", False):
            return None
        if "amplitude_dB" in self._cache:
            return self._cache["amplitude_dB"]
        amp = self._amplitude()
        db = 20 * np.log10(np.maximum(amp, EPS))
        db = np.clip(db, -120.0, 0.0)
        self._cache["amplitude_dB"] = db
        return db
    
    def _peak_amplitude(self) -> np.ndarray:
        if getattr(self, "invalid", False):
            return None
        if "peak_amplitude" in self._cache:
            return self._cache["peak_amplitude"]
        
        # Frame the signal (shape: [frame_length, num_frames])
        frames = librosa.util.frame(
            self.y,
            frame_length=self.N,
            hop_length=self.H
        )

        # Peak absolute value per frame
        peak = np.max(np.abs(frames), axis=0)

        self._cache["peak_amplitude"] = peak
        return peak
    
    def _peak_amplitude_dB(self) -> np.ndarray:
        if getattr(self, "invalid", False):
            return None
        if "peak_amplitude_dB" in self._cache:
            return self._cache["peak_amplitude_dB"]
        
        peak = self._peak_amplitude()
        peak_dB = 20.0*np.log10(np.maximum(peak, EPS))
        peak_dB = np.clip(peak_dB, -120.0, 0.0)

        self._cache["peak_amplitude_dB"] = peak_dB
        return peak_dB
    
    def _crest_factor(self) -> np.ndarray:
        if getattr(self, "invalid", False):
            return None
        if "crest_factor" in self._cache:
            return self._cache["crest_factor"]
        
        if self._is_globally_silent():
            crest = np.zeros_like(self._amplitude())
            self._cache["crest_factor"] = crest
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

        self._cache["crest_factor"] = crest
        return crest
    
    def _crest_factor_dB(self) -> np.ndarray:
        if getattr(self, "invalid", False):
            return None
        if "crest_factor_dB" in self._cache:
            return self._cache["crest_factor_dB"]
        
        crest = self._crest_factor()
        
        if self._is_globally_silent():
            cf_db = np.zeros_like(crest)
            self._cache["crest_factor_dB"] = cf_db
            return cf_db
        
        crest_dB = 20.0 * np.log10(np.maximum(crest, EPS))
        crest_dB = np.clip(crest_dB, -6.0, 40.0)

        self._cache["crest_factor_dB"] = crest_dB
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
        if "energy_envelope" in self._cache:
            return self._cache["energy_envelope"]
        # vectorized framing energy
        frames = librosa.util.frame(self.y, frame_length=self.N, hop_length=self.H).astype(float)
        E = np.sum(frames**2, axis=0)
        self._cache["energy_envelope"] = E
        return E
    
    def _energy_variance(self) -> float:
        if getattr(self, "invalid", False):
            return None
        if "energy_variance" in self._cache:
            return self._cache["energy_variance"]
        
        rms = self._amplitude()
        if rms.size == 0:
            self._cache["energy_variance"] = 0.0
            return 0.0
        
        # Normalize RMS to reduce dependence on absolute level
        q75, q25 = np.percentile(rms, [75, 25])
        iqr = q75 - q25

        ev = float(iqr/(np.median(rms) + EPS))

        self._cache["energy_variance"] = ev
        return ev
    
    def _energy_mod_rate(self) -> float:
        if getattr(self, "invalid", False):
            return None
        if "energy_mod_rate" in self._cache:
            return self._cache["energy_mod_rate"]
        
        rms = self._amplitude()
        if rms.size < 2:
            self._cache["energy_mod_rate"] = 0.0
            return 0.0
        
        # Normalize RMS to remove absolute loudness dependency
        q75, q25 = np.percentile(rms, [75, 25])
        iqr = q75 - q25

        rms_n = (rms - q25) / (iqr + EPS)

        mod_rate = float(np.mean(np.abs(np.diff(rms_n))))
        mod_rate *= self.sr/float(self.H)  # convert to Hz

        self._cache["energy_mod_rate"] = mod_rate
        return mod_rate
    
    def _energy_modulation_signal(self) -> np.ndarray:
        """
        Frame-wise absolute energy change.
        """
        if getattr(self, "invalid", False):
            return None
        if "energy_modulation_signal" in self._cache:
            return self._cache["energy_modulation_signal"]

        rms = self._amplitude()
        if rms.size < 2:
            return np.zeros_like(rms)

        rms_n = rms / (np.mean(rms) + EPS)
        mod_sig = np.abs(np.diff(rms_n))

        self._cache["energy_modulation_signal"] = mod_sig
        return mod_sig

    def _zero_crossing_rate(self) -> np.ndarray:
        if getattr(self, "invalid", False):
            return None
        if "zcr" in self._cache:
            return self._cache["zcr"]
        z = librosa.feature.zero_crossing_rate(y=self.y, frame_length=self.N, hop_length=self.H)[0]
        self._cache["zcr"] = z
        return z

    def _dynamic_range(self) -> float:
        if getattr(self, "invalid", False):
            return None
        if "dynamic_range" in self._cache:
            return self._cache["dynamic_range"]
        
        if self._is_globally_silent():
            self._cache["dynamic_range"] = 0.0
            return 0.0

        rms = self._amplitude()
        if np.all(rms == 0):
            dr = 0.0
            self._cache["dynamic_range"] = dr
            return dr
        
        # robust percentiles
        low = np.percentile(rms, 10)
        high = np.percentile(rms, 95)
        low = max(low, EPS)
        dr = 20.0 * np.log10(max(high / low, EPS))
        dr = np.clip(dr, 0.0, 60.0)
        self._cache["dynamic_range"] = float(dr)
        return float(dr)

    # --- Tempo & onset ---
    def _onset_env(self) -> np.ndarray:
        if getattr(self, "invalid", False):
            return None
        if "onset_env" in self._cache:
            return self._cache["onset_env"]

        onset = librosa.onset.onset_strength(
            y=self.y, sr=self.sr, hop_length=self.H
        )

        if onset.size == 0 or np.all(onset == 0):
            onset_n = np.zeros_like(onset)
        else:
            onset_n = onset / (np.max(onset) + EPS)

        self._cache["onset_env"] = onset_n
        return onset_n

    
    def _onset_autocorr(self) -> np.ndarray:
        if getattr(self, "invalid", False):
            return None
        if "onset_autocorr" in self._cache:
            return self._cache["onset_autocorr"]

        onset = self._onset_env()

        # Early exit for silence / no onsets
        if onset.size < 2 or np.all(onset == 0):
            ac = np.zeros_like(onset)
            self._cache["onset_autocorr"] = ac
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

        self._cache["onset_autocorr"] = ac
        return ac
    
    def _pulse_clarity(self) -> float:
        if getattr(self, "invalid", False):
            return None
        if "pulse_clarity" in self._cache:
            return self._cache["pulse_clarity"]
        
        if self._is_globally_silent():
            self._cache["pulse_clarity"] = 0.0
            return 0.0
        
        ac = self._onset_autocorr()
        if ac.size < 4:
            self._cache["pulse_clarity"] = 0.0
            return 0.0
        
        peak = np.max(ac[1:])
        mean = np.mean(ac[1:])

        clarity = float((peak - mean)/(peak + EPS))
        clarity = safe_clip01(clarity)

        self._cache["pulse_clarity"] = clarity
        return clarity

    def _tempo_var(self, window_seconds: float = 8.0) -> Tuple[float, float, float]:
        if getattr(self, "invalid", False):
            return None
        # cached key includes hop & window
        key = f"tempo_var_{self.H}_{window_seconds}"
        if key in self._cache:
            return self._cache[key]
        
        if self._is_globally_silent():
            self._cache[key] = (0.0, 0.0, 0.0)
            return self._cache[key]

        onset = self._onset_env()
        if onset.size < 2:
            self._cache[key] = (0.0, 0.0, 0.0)
            return self._cache[key]
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
            self._cache[key] = res
            return res
        tv = float(np.var(tempos))
        tm = float(np.mean(tempos))
        stab = float(max(0.0, 1.0 - (tv / (tm + EPS))))
        self._cache[key] = (tv, tm, stab)
        return self._cache[key]
    
    def _silence_ratio(self) -> float:
        if getattr(self, "invalid", False):
            return None
        if "silence_ratio" in self._cache:
            return self._cache["silence_ratio"]
        
        rms = self._amplitude()
        if rms.size == 0:
            self._cache["silence_ratio"] = 0.0
            return 0.0
        
        # Avoid degenerate threshold
        thresh = max(np.percentile(rms, 10), EPS)

        silent_frames = np.sum(rms < thresh)
        ratio = float(silent_frames) / float(rms.size)

        ratio = safe_clip01(ratio)
        self._cache["silence_ratio"] = ratio
        return ratio
    
    # Spotify-style spectral features can be added here (Full)
    def loudness_time(self) -> float:
        if getattr(self, "invalid", False):
            return None
        if "loudness_time" in self._cache:
            return self._cache["loudness_time"]
        
        rms_db = self._amplitude_dB()
        if rms_db.size == 0:
            self._cache["loudness_time"] = 0.0
            return 0.0
        
        # Exclude silence frames
        thresh = np.percentile(rms_db, 10)
        active_db = rms_db[rms_db > thresh]

        if active_db.size == 0:
            self._cache["loudness_time"] = 0.0
            return 0.0
        
        # Robust aggregation
        loudness = float(np.percentile(active_db, 95))
        loudness -= 0.1*self._dynamic_range()

        # Clip and return
        loudness = np.clip(loudness, -60.0, 0.0)
        self._cache["loudness_time"] = loudness
        return loudness
    
    def speechiness_time(self) -> float:
        if getattr(self, "invalid", False):
            return None
        if "speechiness_time" in self._cache:
            return self._cache["speechiness_time"]

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
        self._cache["speechiness_time"] = speechiness
        return speechiness
    
    def instrumentalness_time(self) -> float:
        if getattr(self, "invalid", False):
            return None
        if "instrumentalness_time" in self._cache:
            return self._cache["instrumentalness_time"]
        
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

        self._cache["instrumentalness_time"] = instrumentalness
        return instrumentalness
    
    # Spotify-style spectral features can be added here (partial)
    def energy_time(self) -> float:
        if getattr(self, "invalid", False):
            return None
        if "energy_time" in self._cache:
            return self._cache["energy_time"]
        
        # Extract features
        rms = self._amplitude()
        if rms.size == 0:
            self._cache["energy_time"] = 0.0
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
        self._cache["energy_time"] = energy

        return energy
    
    def acousticness_time(self) -> float:
        if getattr(self, "invalid", False):
            return None
        if "acousticness" in self._cache:
            return self._cache["acousticness"]
        
        rms = self._amplitude()
        if rms.size == 0:
            self._cache["acousticness"] = 0.0
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
        self._cache["acousticness"] = acousticness
        return acousticness
    
    def danceability_time(self) -> float:
        if getattr(self, "invalid", False):
            return None
        if "danceability" in self._cache:
            return self._cache["danceability"]

        rms = self._amplitude()
        if rms.size == 0:
            self._cache["danceability"] = 0.0
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
        self._cache["danceability"] = danceability
        return danceability

    
    def tempo_time(self, bpm_min: float = 40.0, bpm_max: float = 220.0) -> float:
        """
        Estimate tempo from time-domain onset autocorrelation with harmonic voting.
        Adds perceptual bias toward ~100–130 BPM.
        """

        if getattr(self, "invalid", False):
            return None

        if "tempo" in self._cache:
            return self._cache["tempo"]

        onset = self._onset_env()
        if onset is None or onset.size < 8 or np.all(onset == 0):
            self._cache["tempo"] = 0.0
            return 0.0

        ac = self._onset_autocorr()
        if ac.size < 8:
            self._cache["tempo"] = 0.0
            return 0.0

        hop_time = self.H / float(self.sr)

        lags = np.arange(1, len(ac))
        ac_lag = ac[1:]

        bpms = 60.0 / (lags * hop_time + EPS)

        mask = (bpms >= bpm_min) & (bpms <= bpm_max)
        if not np.any(mask):
            self._cache["tempo"] = 0.0
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
            self._cache["tempo"] = 0.0
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

        self._cache["tempo"] = tempo_est
        return tempo_est

    def liveness_time(self) -> float:
        if getattr(self, "invalid", False):
            return None
        if "liveness" in self._cache:
            return self._cache["liveness"]
        
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
        self._cache["liveness"] = liveness

        return liveness
    
    def time_sig_time(self) -> int:
        if getattr(self, "invalid", False):
            return None
        """
        Heuristic time-signature estimator from time-domain rhythm structure.
        Falls back to 4 when confidence is weak.
        """
        if "time_signature" in self._cache:
            return self._cache["time_signature"]

        pulse = self._pulse_clarity()
        ac = self._onset_autocorr()

        # If rhythm is weak, default to 4/4
        if pulse < 0.15 or ac.size < 10:
            self._cache["time_signature"] = 4
            return 4

        tempo = self._tempo_var()[1]
        if tempo <= 0:
            self._cache["time_signature"] = 4
            return 4

        beat_period_sec = 60.0 / tempo
        frames_per_beat = int(round(beat_period_sec * self.sr / float(self.H)))

        if frames_per_beat <= 1:
            self._cache["time_signature"] = 4
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
            self._cache["time_signature"] = 4
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
        
        self._cache["time_signature"] = int(best)
        return int(best)


    # ---------- STFT-based features (cached) ----------
class STFTFeatures(AudioSignal):
    def __init__(self, audio_path: str, N: int = 2048, H: int = 512,
                 pad_mode: str = 'constant', center: bool = True):
        super().__init__(audio_path, N=N, H=H)
        # Recompute fft freqs with final N
        self._fft_freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.N)

        # caches
        self._cache_spec: Dict[str, object] = {}
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
        cum = np.cumsum(P, axis=0)
        total = cum[-1, :]
        target = roll_percent * total
        # find first bin >= target
        bins = np.argmax(cum >= target[None, :], axis=0)
        n_bins = P.shape[0]
        freqs = self._fft_freqs[:n_bins]
        roll = freqs[bins]
        self._cache_spec[key] = roll
        return roll

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

    # ---------- High-level perceptual features ----------
    # All three of these use robust normalization and are safe to call on silence
    def speechiness(self) -> float:
        zcr = self._zero_crossing_rate()
        flux = self._onset_env()
        flat = self._spectral_flatness()
        hr = self._harmonic_ratio()
        inh = self._inharmonicity()
        slope = self._spectral_slope()
        mid_energy = self._narrowband_energy(300, 3000)

        # align lengths safely
        L = min_length(zcr, flux, flat, hr, inh, slope, mid_energy)
        if L == 0:
            return 0.0
        zcr, flux, flat = zcr[:L], flux[:L], flat[:L]
        hr = np.nan_to_num(hr[:L], nan=safe_median(hr[:L], 0.0))
        inh = np.nan_to_num(inh[:L], nan=safe_median(inh[:L], 0.0))
        slope = np.nan_to_num(slope[:L], nan=safe_median(slope[:L], 0.0))
        mid_energy = np.nan_to_num(mid_energy[:L], nan=safe_median(mid_energy[:L], 0.0))

        # robust normalize each
        zcr_n = robust_normalize(zcr)
        flux_n = robust_normalize(flux)
        flat_n = robust_normalize(flat)
        hr_n = 1.0 - robust_normalize(hr)
        inh_n = robust_normalize(inh)
        slope_inv = 1.0 - robust_normalize(slope)
        mid_n = robust_normalize(mid_energy)

        # weights (tunable)
        w_zcr = 0.20; w_flux = 0.20; w_flat = 0.20
        w_mid = 0.15; w_inh = 0.10; w_slope = 0.10; w_hr = 0.05

        frame_score = (w_zcr*zcr_n + w_flux*flux_n + w_flat*flat_n +
                       w_mid*mid_n + w_inh*inh_n + w_slope*slope_inv + w_hr*hr_n)
        score = float(np.mean(frame_score))
        return safe_clip01(score)

    def acousticness(self) -> float:
        hr = self._harmonic_ratio()
        flat = self._spectral_flatness()
        inh = self._inharmonicity()
        entropy = self._spectral_entropy()
        centroid = self._spectral_centroid()
        mid_energy = self._narrowband_energy(300, 3000)
        high_energy = self._narrowband_energy(6000, min(16000, self.sr // 2))

        L = min_length(hr, flat, inh, entropy, centroid, mid_energy, high_energy)
        if L == 0:
            return 0.0
        hr = hr[:L]; flat = flat[:L]; inh = inh[:L]; entropy = entropy[:L]
        centroid = centroid[:L]; mid_energy = mid_energy[:L]; high_energy = high_energy[:L]

        hr_n = robust_normalize(hr)
        flat_n = robust_normalize(flat)
        inh_n = robust_normalize(inh)
        ent_n = robust_normalize(entropy)
        cent_n = robust_normalize(centroid)

        band_ratio = np.divide(high_energy, (mid_energy + high_energy + EPS))
        band_n = robust_normalize(band_ratio)

        # invert features where higher means "less acoustic"
        flat_inv = 1.0 - flat_n
        inh_inv = 1.0 - inh_n
        ent_inv = 1.0 - ent_n
        cent_inv = 1.0 - cent_n
        band_inv = 1.0 - band_n

        # weights
        w_hr = 0.30; w_flat = 0.20; w_inh = 0.15
        w_ent = 0.10; w_cent = 0.10; w_band = 0.15

        frame_score = (w_hr*hr_n + w_flat*flat_inv + w_inh*inh_inv +
                       w_ent*ent_inv + w_cent*cent_inv + w_band*band_inv)
        track_score = float(np.median(frame_score))
        return safe_clip01(track_score)

    def danceability(self) -> float:
        onset = self._onset_env()
        tempo_var, tempo_mean, tempo_stab = self._tempo_var()
        bass = self._narrowband_energy(20, 250)
        centroid = self._spectral_centroid()
        flat = self._spectral_flatness()
        entropy = self._spectral_entropy()

        L = min_length(onset, bass, centroid, flat, entropy)
        if L == 0:
            return 0.0
        onset = onset[:L]; bass = bass[:L]; centroid = centroid[:L]
        flat = flat[:L]; entropy = entropy[:L]

        onset_n = robust_normalize(onset)
        bass_n = robust_normalize(bass)
        centroid_n = robust_normalize(centroid)
        flat_n = robust_normalize(flat)
        ent_n = robust_normalize(entropy)

        centroid_inv = 1.0 - centroid_n
        flat_inv = 1.0 - flat_n
        ent_inv = 1.0 - ent_n

        # pulse clarity via FFT autocorr
        ac = fft_autocorr(onset_n)
        if ac.size == 0:
            pulse_clarity = 0.0
        else:
            ac_pos = ac[len(ac)//2:]
            if np.max(ac_pos) <= 0:
                pulse_clarity = 0.0
            else:
                ac_pos = ac_pos / (np.max(ac_pos) + EPS)
                # reasonable lag window: 60-200 bpm mapped to lags
                min_lag = int(round((60.0 / 200.0) * (self.sr / float(self.H))))
                max_lag = int(round((60.0 / 60.0) * (self.sr / float(self.H))))
                min_lag = min(min_lag, len(ac_pos) - 1)
                max_lag = min(max_lag, len(ac_pos))
                if max_lag > min_lag:
                    pulse_clarity = float(np.max(ac_pos[min_lag:max_lag]))
                else:
                    pulse_clarity = 0.0

        # tempo score preference
        if tempo_mean == 0:
            tempo_score = 0.0
        else:
            bpm = tempo_mean
            if 80 <= bpm <= 130:
                tempo_score = 1.0
            elif 160 <= bpm <= 200:
                tempo_score = 0.8
            else:
                tempo_score = np.exp(-((bpm - 110.0) / 50.0)**2)

        tempo_stab = float(max(0.0, min(1.0, tempo_stab)))

        # weights
        w_onset = 0.25; w_bass = 0.20; w_pulse = 0.15
        w_cent = 0.10; w_flat = 0.10; w_ent = 0.10; w_tempo = 0.10

        frame = (w_onset*onset_n + w_bass*bass_n + w_pulse*pulse_clarity +
                 w_cent*centroid_inv + w_flat*flat_inv + w_ent*ent_inv +
                 w_tempo*tempo_stab)
        frame_score = float(np.median(frame))
        dance = 0.6 * frame_score + 0.4 * float(np.clip(tempo_score, 0.0, 1.0))
        return safe_clip01(dance)
