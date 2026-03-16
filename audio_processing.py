import os
import numpy as np
import librosa
from scipy.signal import find_peaks
from typing import Tuple, Optional, Dict

# ---------- Utilities ----------
EPS = 1e-10

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
    def __init__(self, audio_path: str = None, signal: np.ndarray = None, N: int = 2048, H: int = 512):        
        self.audio_path = audio_path
        self.N = N
        self.H = H

        if audio_path is not None:
            # load
            if not isinstance(audio_path, str):
                raise ValueError("audio_path must be a string path")
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            self.y, self.sr, invalid = safe_load(audio_path)
        elif signal is not None:
            self.y = np.asarray(signal, dtype=float)
            self.sr = 22050
            invalid = False
        else:
            raise ValueError("Must provide either 'path' or 'signal'")

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
        self.invalid = getattr(sig, 'invalid', False)
        self._cache_time = {}

        # global loudness check (too quiet to analyze)
        # gl = self._global_loudness_dB()
        # if gl is None or gl < -70.0:
        #     self.invalid = True
        #     return

        # pad short audio
        if len(self.y) < self.N:
            pad = self.N - len(self.y)
            self.y = np.pad(self.y, (0, pad), mode="constant")

        # precompute frequencies
        self._fft_freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.N)

    def _global_loudness_dB(self) -> float:
        if getattr(self, "invalid", False):
            return -80.0
        if "global_loudness_dB" in self._cache_time:
            return self._cache_time["global_loudness_dB"]
        
        rms = np.sqrt(np.mean(self.y**2)) + EPS
        if rms < 1e-10:
            self._cache_time["global_loudness_dB"] = -80.0
            return -80.0
        
        loud_db = 20.0*np.log10(rms)

        self._cache_time["global_loudness_dB"] = loud_db
        return loud_db

    # Amplitude/Loudness Features
    def _rms_envelope(self) -> np.ndarray:
        if "rms_env" in self._cache_time:
            return self._cache_time["rms_env"]
        
        num_frames = 1 + int((len(self.y) - self.N) // self.H)
        rms_env = np.zeros(num_frames, dtype=float)

        for i in range(num_frames):
            start = i*self.H
            end = start + self.N
            frame = self.y[start:end]

            rms_env[i] = float(np.sqrt(np.mean(frame**2)))

        self._cache_time["rms_env"] = rms_env
        return rms_env

    def _short_time_energy(self) -> np.ndarray:
        if "ste" in self._cache_time:
            return self._cache_time["ste"]

        rms_env = self._rms_envelope()
        ste = self.N*(rms_env**2)

        self._cache_time["ste"] = ste
        return ste
    
    def _peak_amplitude(self) -> np.ndarray:
        if "peak_amp" in self._cache_time:
            return self._cache_time["peak_amp"]
        
        num_frames = 1 + int((len(self.y) - self.N) // self.H)
        peak_amp = np.zeros(num_frames, dtype=float)

        for i in range(num_frames):
            start = i*self.H
            end = start + self.N
            frame = self.y[start:end]

            peak_amp[i] = float(np.max(np.abs(frame)))

        self._cache_time["peak_amp"] = peak_amp
        return peak_amp
    
    def _active_rms_mask(self, db_threshold: float = -60.0) -> np.ndarray:
        if "active_mask" in self._cache_time:
            return self._cache_time["active_mask"]
        
        rms_env = self._rms_envelope()
        peak_amp = self._peak_amplitude()

        rms_db = 20*np.log10(rms_env + EPS)
        peak_db = 20*np.log10(peak_amp + EPS)

        mask = (rms_db > db_threshold) | (peak_db > db_threshold)

        self._cache_time["active_mask"] = mask
        return mask
    
    def _crest_factor(self) -> np.ndarray:
        if "crest_factor" in self._cache_time:
            return self._cache_time["crest_factor"]
        
        rms_env = self._rms_envelope()
        peak_amp = self._peak_amplitude()
        mask = self._active_rms_mask(db_threshold=-60.0)

        crest = np.zeros_like(rms_env)
        crest[mask] = peak_amp[mask]/(rms_env[mask] + EPS)

        self._cache_time["crest_factor"] = crest
        return crest
    
    def _dynamic_range(self) -> float:
        if "dynamic_range" in self._cache_time:
            return self._cache_time["dynamic_range"]
        
        rms_env = self._rms_envelope()

        rms_db = 20.0*np.log10(rms_env + EPS)
        rms_db = np.clip(rms_db, -80.0, 0.0)
        dr = np.percentile(rms_db, 90) - np.percentile(rms_db, 10)

        self._cache_time["dynamic_range"] = dr
        return dr
    
    def _onset_envelope(self) -> np.ndarray:
        """Cached onset strength envelope"""
        if "onset_env_strength" in self._cache_time:
            return self._cache_time["onset_env_strength"]
        
        env = librosa.onset.onset_strength(y=self.y, sr=self.sr, hop_length=self.H)
        self._cache_time["onset_env_strength"] = env
        return env

    def _onset_frames(self) -> np.ndarray:
        """Cached onset detection frames"""
        if "onset_frames" in self._cache_time:
            return self._cache_time["onset_frames"]
        
        env = self._onset_envelope()
        frames = librosa.onset.onset_detect(
            onset_envelope=env,
            sr=self.sr, hop_length=self.H,
            backtrack=False, units='frames'
        )
        self._cache_time["onset_frames"] = frames
        return frames
    
    def _attack_time(self) -> float:
        if "attack_time" in self._cache_time:
            return self._cache_time["attack_time"]

        rms_env = self._rms_envelope()
        onsets = self._onset_frames()
        
        if len(onsets) == 0:
            self._cache_time["attack_time"] = 0.0
            return 0.0
        
        attack_times = []
        for onset_frame in onsets:
            # Define window around onset
            start = max(0, onset_frame - 5)
            end = min(len(rms_env), onset_frame + 20)

            segment = rms_env[start:end]
            if len(segment) < 3:
                continue

            peak_val = np.max(segment)
            threshold_10 = 0.1*peak_val
            threshold_90 = 0.9*peak_val

            idx_10 = np.where(segment >= threshold_10)[0]
            idx_90 = np.where(segment >= threshold_90)[0]

            if len(idx_10) > 0 and len(idx_90) > 0:
                t_10 = idx_10[0]
                t_90 = idx_90[0]
                attack_frames = t_90 - t_10
                attack_times.append(attack_frames*self.H/float(self.sr))

        if len(attack_times) == 0:
            self._cache_time["attack_time"] = 0.0
            return 0.0
        
        avg_attack_time = float(np.median(attack_times))

        self._cache_time["attack_time"] = avg_attack_time
        return avg_attack_time
    
    def _attack_slope(self) -> float:
        """Average attack slope across all onsets (dB/second)"""
        if "attack_slope" in self._cache_time:
            return self._cache_time["attack_slope"]
        
        rms_env = self._rms_envelope()
        rms_dB = 20.0 * np.log10(rms_env + EPS)
        
        onsets = self._onset_frames()
        
        if len(onsets) == 0:
            self._cache_time["attack_slope"] = 0.0
            return 0.0
        
        attack_slopes = []
        for onset_frame in onsets:
            start = max(0, onset_frame - 5)
            end = min(len(rms_env), onset_frame + 20)
            segment = rms_dB[start:end]  # ← Use dB, not linear
            
            if len(segment) < 3:
                continue
            
            peak_val = np.max(segment)
            threshold_10 = peak_val - 0.1 * (peak_val - np.min(segment))  # 10% of range
            threshold_90 = peak_val - 0.9 * (peak_val - np.min(segment))  # 90% of range
            
            idx_10 = np.where(segment >= threshold_10)[0]
            idx_90 = np.where(segment >= threshold_90)[0]
            
            if len(idx_10) > 0 and len(idx_90) > 0:
                t_10 = idx_10[0]
                t_90 = idx_90[0]
                
                if t_90 > t_10:
                    # Slope in dB/second
                    db_change = segment[t_90] - segment[t_10]
                    time_change = (t_90 - t_10) * self.H / self.sr
                    slope = db_change / (time_change + EPS)
                    attack_slopes.append(slope)
        
        if len(attack_slopes) == 0:
            self._cache_time["attack_slope"] = 0.0
            return 0.0
        
        avg_slope = float(np.median(attack_slopes))
        self._cache_time["attack_slope"] = avg_slope
        return avg_slope
    
    def _decay_slope(self) -> float:
        """Average decay slope across all onsets (dB/second)"""
        if "decay_slope" in self._cache_time:
            return self._cache_time["decay_slope"]
        
        rms_env = self._rms_envelope()
        rms_dB = 20.0 * np.log10(rms_env + EPS)
        
        onsets = self._onset_frames()
        
        if len(onsets) == 0:
            self._cache_time["decay_slope"] = 0.0
            return 0.0
        
        decay_slopes = []
        for onset_frame in onsets:
            start = onset_frame
            end = min(len(rms_env), onset_frame + 30)  # Look ahead for decay
            segment = rms_dB[start:end]
            
            if len(segment) < 5:
                continue
            
            # Find peak in this segment
            peak_idx = np.argmax(segment)
            peak_val = segment[peak_idx]
            
            # Measure decay from peak to 50% below peak
            decay_start_idx = peak_idx
            decay_threshold = peak_val - 20.0  # 20 dB below peak
            
            # Find where it crosses threshold
            after_peak = segment[peak_idx:]
            below_thresh = np.where(after_peak <= decay_threshold)[0]
            
            if len(below_thresh) > 0:
                decay_end_idx = peak_idx + below_thresh[0]
                
                if decay_end_idx > decay_start_idx:
                    db_change = segment[decay_end_idx] - segment[decay_start_idx]
                    time_change = (decay_end_idx - decay_start_idx) * self.H / self.sr
                    slope = abs(db_change) / (time_change + EPS)  # Absolute value
                    decay_slopes.append(slope)
        
        if len(decay_slopes) == 0:
            self._cache_time["decay_slope"] = 0.0
            return 0.0
        
        avg_slope = float(np.median(decay_slopes))
        self._cache_time["decay_slope"] = avg_slope
        return avg_slope
    
    def _energy_variance(self) -> float:
        if "energy_variance" in self._cache_time:
            return self._cache_time["energy_variance"]
        
        rms_env = self._rms_envelope()
        mask = self._active_rms_mask(db_threshold=-60.0)
        active = rms_env[mask]

        if active.size < 2:
            self._cache_time["energy_variance"] = 0.0
            return 0.0
        
        q75, q25 = np.percentile(active, [75 ,25])
        iqr = q75 - q25

        var = float(iqr/(np.median(active) + EPS))

        self._cache_time["energy_variance"] = var
        return var
    
    def _energy_modulation_rate(self) -> float:
        if "energy_mod_rate" in self._cache_time:
            return self._cache_time["energy_mod_rate"]
        
        rms_env = self._rms_envelope()

        if rms_env.size < 2:
            self._cache_time["energy_mod_rate"] = 0.0
            return 0.0
        
        mask = self._active_rms_mask(db_threshold=-60.0)
        active = rms_env[mask]
        silence_ratio = 1.0 - (mask.sum()/len(mask))

        if active.size < 2:
            self._cache_time["energy_mod_rate"] = 0.0
            return 0.0
        
        med = np.median(active)

        if silence_ratio > 0.1:
            q75, q25 = np.percentile(rms_env, [75, 25])
        else:
            q75, q25 = np.percentile(active, [75, 25])

        iqr = q75 - q25

        if iqr < max(1e-4, 0.01*med):
            self._cache_time["energy_mod_rate"] = 0.0
            return 0.0
        
        rms_n = np.clip((rms_env - q25) / iqr, -1.0, 3.0)
        mod_sig = np.abs(np.diff(rms_n))
        mod_rate = float(np.var(mod_sig))

        self._cache_time["energy_mod_rate"] = mod_rate
        return mod_rate
    
    # Noise/Speechiness Features
    def _zero_crossing_rate(self) -> np.ndarray:
        if "zcr" in self._cache_time:
            return self._cache_time["zcr"]
        
        num_frames = 1 + int((len(self.y) - self.N) // self.H)
        zcr = np.zeros(num_frames, dtype=float)

        for i in range(num_frames):
            start = i*self.H
            end = start + self.N

            frame = self.y[start:end]
            
            signs = np.sign(frame)
            signs[signs == 0] = 1
            crossings = np.sum(np.abs(np.diff(signs)))/2.0
            zcr[i] = float(crossings/len(frame))

        self._cache_time["zcr"] = zcr
        return zcr
    
    def _zcr_variance(self) -> float:
        if "zcr_variance" in self._cache_time:
            return self._cache_time["zcr_variance"]
        
        zcr = self._zero_crossing_rate()
        mask = self._active_rms_mask(db_threshold=-60.0)
        active_zcr = zcr[mask]

        if active_zcr.size < 2:
            self._cache_time["zcr_variance"] = 0.0
            return 0.0
        
        q75, q25 = np.percentile(active_zcr, [75 ,25])
        iqr = q75 - q25

        var = float(iqr/(np.median(active_zcr) + EPS))

        self._cache_time["zcr_variance"] = var
        return var
    
    def _voiced_ratio(self) -> float:
        if "voiced_ratio" in self._cache_time:
            return self._cache_time["voiced_ratio"]
        
        zcr = self._zero_crossing_rate()
        mask = self._active_rms_mask(db_threshold=-60.0)

        if mask.sum() < 2:
            self._cache_time["voiced_ratio"] = 0.0
            return 0.0
        
        num_frames = len(zcr)
        ac_peaks = np.zeros(num_frames, dtype=float)

        f_min = 50.0
        f_max = 400.0

        # Convert to lag range
        min_lag = int(self.sr / f_max)  # smallest lag (highest pitch)
        max_lag = int(self.sr / f_min)  # largest lag (lowest pitch)
        min_lag = max(min_lag, 2)

        for i in range(num_frames):
            if not mask[i]:
                continue

            start = i*self.H
            end = start + self.N
            frame = self.y[start:end]

            if frame.size < 3:
                continue

            frame_zm = frame - np.mean(frame)
            ac_full = np.correlate(frame_zm, frame_zm, mode='full')
            ac = ac_full[len(ac_full)//2:]

            if ac[0] <= EPS:
                ac_peaks[i] = 0.0
                continue

            ac /= ac[0]

            cur_max_lag = min(max_lag, len(ac) - 1)

            if cur_max_lag <= min_lag:
                ac_peaks[i] = 0.0
                continue

            ac_peaks[i] = float(np.max(ac[min_lag:cur_max_lag + 1]))

        strong_periodic = mask & (ac_peaks >= 0.5)
        moderate_periodic = mask & (ac_peaks >= 0.35)

        low_zcr = zcr <= 0.35

        voiced = strong_periodic | (moderate_periodic & low_zcr)

        voiced_frames = voiced.sum()
        active_frames = mask.sum()

        ratio = float(voiced_frames)/float(active_frames + EPS)

        self._cache_time["voiced_ratio"] = ratio
        return ratio
    
    def _unvoiced_ratio(self) -> float:
        return 1.0 - self._voiced_ratio()
    
    def _transient_rate(self) -> float:
        if "transient_rate" in self._cache_time:
            return self._cache_time["transient_rate"]
        
        ste = self._short_time_energy()

        # Adaptive threshold (median + factor*MAD)
        median = np.median(ste)
        mad = np.median(np.abs(ste - median))
        threshold = median + 2.0*mad

        peaks, _ = find_peaks(ste, height=threshold)

        duration = len(self.y)/float(self.sr)
        rate = float(len(peaks))/duration

        self._cache_time["transient_rate"] = rate
        return rate

    def _transient_counts(self) -> int:
        rate = self._transient_rate()
        duration = len(self.y)/float(self.sr)

        return int(round(rate*duration))
    
    # Rhythm/Beats Features
    def _onset_times(self) -> np.ndarray:
        """
        Return onset times (seconds) from peaks in onset envelope
        """
        if "onset_times" in self._cache_time:
            return self._cache_time["onset_times"]
        
        onset_env = self._onset_envelope()

        if onset_env.size == 0:
            self._cache_time["onset_times"] = np.array([], dtype=float)
            return self._cache_time["onset_times"]
        
        # Adaptive threshold: median + k*MAD
        med = np.median(onset_env)
        mad = np.median(np.abs(onset_env - med))
        thr = med + 2.0*mad

        # Find peaks above threshold
        peaks, _ = find_peaks(onset_env, height=thr)

        # Convert frame indices to time (seconds)
        times = (peaks*self.H)/float(self.sr)

        self._cache_time["onset_times"] = times.astype(float)
        return times.astype(float)
    
    def _onset_rate(self) -> float:
        """
        Average number of onsets per second
        """
        if "onset_rate" in self._cache_time:
            return self._cache_time["onset_rate"]
        
        onset_times = self._onset_times()
        if onset_times.size == 0:
            self._cache_time["onset_rate"] = 0.0
            return 0.0

        duration = len(self.y)/float(self.sr)
        if duration <= 0:
            self._cache_time["onset_rate"] = 0.0
            return 0.0
        
        rate = float(onset_times.size)/duration
        self._cache_time["onset_rate"] = rate
        return rate 
    
    def _ioi_values(self) -> np.ndarray:
        """ 
        Inter-onset intervals (seconds)
        """
        if "ioi" in self._cache_time:
            return self._cache_time["ioi"]
        
        onset_times = self._onset_times()
        if onset_times.size < 2:
            self._cache_time["ioi"] = np.array([], dtype=float)
            return self._cache_time["ioi"]
        
        ioi = np.diff(onset_times)

        # Keep only positive, non-zero intervals
        ioi = ioi[ioi > 0]
        self._cache_time["ioi"] = ioi.astype(float)
        return self._cache_time["ioi"]
    
    def _ioi_stats(self) -> tuple:
        """ 
        Return mean, std, cv of IOIs
        """
        ioi = self._ioi_values()
        if ioi.size < 2:
            return (0.0, 0.0, 0.0)

        mean_ioi = float(np.mean(ioi))
        std_ioi = float(np.std(ioi, ddof=1))
        if mean_ioi > EPS:
            cv_ioi = float(std_ioi/mean_ioi)
        else:
            cv_ioi = 0.0

        return (mean_ioi, std_ioi, cv_ioi)

    def _onset_autocorrelation(self) -> np.ndarray:
        """
        Normalized autocorrelation of onset envelope
        """
        if "ac_onset" in self._cache_time:
            return self._cache_time["ac_onset"]

        onset_env = self._onset_envelope()
        if onset_env.size == 0:
            self._cache_time["ac_onset"] = np.array([], dtype=float)
            return self._cache_time["ac_onset"]
        
        x = onset_env - np.mean(onset_env)
        ac_full = np.correlate(x, x, mode="full")
        ac = ac_full[len(ac_full)//2:].astype(float)

        if ac[0] > EPS:
            ac /= ac[0]

        self._cache_time["ac_onset"] = ac
        return ac 
    
    def _tempo_from_onset_ac(self,
                             bpm_min: float = 40.0,
                             bpm_max: float = 240.0) -> float:
        """
        Estimate global tempo (BPM) from onset autocorrelation
        """
        ac = self._onset_autocorrelation()
        if ac.size < 3:
            return 0.0

        # Map BPM range to lag range: lag*fs_frames/freq
        # onset_env is per hop -> effective frame rate = sr/H
        fs_env = self.sr/float(self.H)

        # Convert BPM bounds to lags
        f_min = bpm_min/60.0
        f_max = bpm_max/60.0

        lag_min = int(fs_env/f_max)
        lag_max = int(fs_env/f_min)

        lag_min = max(lag_min, 1)
        lag_max = min(lag_max, ac.size - 1)
        if lag_max <= lag_min:
            return 0.0
        
        # Search for max AC peak in this lag region (exclude lag 0)
        search_region = ac[lag_min:lag_max + 1]
        
        peaks, properties = find_peaks(search_region, height=0.1)

        if len(peaks) == 0:
            rel_peak_idx = int(np.argmax(search_region))
            tau_peak = lag_min + rel_peak_idx
            tempo_bpm = 60*fs_env/float(tau_peak)
            return float(tempo_bpm)

        # Convert peak indices to absolute lags
        peak_lags = lag_min + peaks
        peak_strengths = search_region[peaks]

        votes = {}
        for lag, strength in zip(peak_lags, peak_strengths):
            candidates = [lag/2.0, lag, 2.0*lag]

            for cand_lag in candidates:
                # Check if candidate is in valid range
                cand_bpm = 60*fs_env/cand_lag
                if bpm_min <= cand_bpm <= bpm_max:
                    cand_lag_int = int(round(cand_lag))
                    votes.setdefault(cand_lag_int, 0.0)
                    votes[cand_lag_int] += strength

        if not votes:
            tau_peak = peak_lags[np.argmax(peak_strengths)]
        else:
            tau_peak = max(votes, key=votes.get)

        tempo_bpm = 60*fs_env/float(tau_peak)
        return float(tempo_bpm)
    
    def _pulse_clarity_ac(self) -> float:
        """
        Pulse clarity from dominance of main AC peak over runner-up,
        gated by absolute peak strength to suppress noise floor.
        """
        if "pulse_clarity_ac" in self._cache_time:
            return self._cache_time["pulse_clarity_ac"]

        ac = self._onset_autocorrelation()
        if ac.size < 3:
            self._cache_time["pulse_clarity_ac"] = 0.0
            return 0.0

        ac_pos = ac[1:]
        if ac_pos.size == 0:
            self._cache_time["pulse_clarity_ac"] = 0.0
            return 0.0

        peaks, _ = find_peaks(ac_pos, prominence=0.01)

        if len(peaks) == 0:
            self._cache_time["pulse_clarity_ac"] = 0.0
            return 0.0

        top_peak = float(ac_pos[peaks].max())

        STRONG_FLOOR = 0.50
        WEAK_FLOOR = 0.15

        if top_peak >= STRONG_FLOOR:
            clarity = 0.60 + 0.4*(top_peak - STRONG_FLOOR)/(1.0 - STRONG_FLOOR)
        elif top_peak >= WEAK_FLOOR:
            clarity = 0.15 + 0.45*(top_peak - WEAK_FLOOR)/(STRONG_FLOOR - WEAK_FLOOR)
        else:
            clarity = top_peak/WEAK_FLOOR*0.15

        clarity = float(np.clip(clarity, 0.0, 1.0))
        self._cache_time["pulse_clarity_ac"] = clarity
        return clarity
    
    def _windowed_tempo_series(self,
                               window_sec: float = 8.0,
                               hop_sec: float = 4.0) -> np.ndarray:
        """ 
        Estimate tempo per window from onset envelope autocorrelation
        """
        onset_env = self._onset_envelope()
        if onset_env.size == 0:
            return np.array([], dtype=float)
        
        fs_env = self.sr/float(self.H)
        win_len = int(window_sec*fs_env)
        hop_len = int(hop_sec*fs_env)

        if win_len < 3:
            return np.array([], dtype=float)
        
        tempos = []
        start = 0
        
        while start < onset_env.size:
            end = min(start + win_len, onset_env.size)
            seg = onset_env[start:end]

            if seg.size < 3:
                start += hop_len
                continue

            x = seg - np.mean(seg)
            ac_full = np.correlate(x, x, mode="full")
            ac = ac_full[len(ac_full)//2:]

            if ac[0] > EPS:
                ac /= ac[0]
            else:
                start += hop_len
                continue

            # Same BPM search as global tempo
            bpm_min, bpm_max = 40.0, 240.0
            f_min = bpm_min/60.0
            f_max = bpm_max/60.0

            lag_min = int(fs_env/f_max)
            lag_max = int(fs_env/f_min)
            lag_min = max(lag_min, 1)
            lag_max = min(lag_max, ac.size - 1)
            if lag_max <= lag_min:
                start += hop_len
                continue

            search_region = ac[lag_min:lag_max + 1]
            
            peaks, properties = find_peaks(search_region, height=0.1)

            if len(peaks) == 0:
                rel_peak_idx = int(np.argmax(search_region))
                tau_peak = lag_min + rel_peak_idx
            else:
                peak_lags = lag_min + peaks
                peak_strengths = search_region[peaks]

                votes = {}
                for lag, strength in zip(peak_lags, peak_strengths):
                    candidates = [lag/2.0, lag, 2.0*lag]
                    for cand_lag in candidates:
                        cand_bpm = 60*fs_env/cand_lag
                        if bpm_min <= cand_bpm <= bpm_max:
                            cand_lag_int = int(round(cand_lag))
                            votes.setdefault(cand_lag_int, 0.0)
                            votes[cand_lag_int] += strength

                if not votes:
                    tau_peak = peak_lags[np.argmax(peak_strengths)]
                else:
                    tau_peak = max(votes, key=votes.get)

            tempo_bpm = 60.0*fs_env/float(tau_peak)
            tempos.append(tempo_bpm)
            start += hop_len

        return np.array(tempos, dtype=float)
    
    def _rhythmic_stability(self) -> dict:
        """ 
        Return tempo variance and stability metrics from windowed tempo
        """
        if "rhythmic_stability" in self._cache_time:
            return self._cache_time["rhythmic_stability"]

        tempos = self._windowed_tempo_series()
        if tempos.size < 2:
            result = {"tempo_var": 0.0, "stability_exp": 1.0, "stability_cv": 1.0}
            self._cache_time["rhythmic_stability"] = result
            return result

        mean_tempo = float(np.mean(tempos))
        var_tempo  = float(np.var(tempos))
        std_tempo  = float(np.std(tempos))

        if mean_tempo <= EPS:
            result = {"tempo_var": 0.0, "stability_exp": 0.0, "stability_cv": 0.0}
            self._cache_time["rhythmic_stability"] = result
            return result

        cv = std_tempo / mean_tempo

        # Linear trend: captures accelerando/ritardando
        x = np.arange(len(tempos), dtype=float)
        coeffs = np.polyfit(x, tempos, 1)
        slope = coeffs[0]

        normalized_slope = abs(slope) / mean_tempo
        trend_penalty = float(np.clip(normalized_slope * len(tempos), 0.0, 1.0))

        # MAD-based residual CV: robust to polyrhythm layer-switching outliers
        trend_line = slope * x + coeffs[1]
        residuals = tempos - trend_line
        residual_cv = float(np.median(np.abs(residuals)) / mean_tempo)

        # Combined: trend-free signals penalized only by residual scatter
        stability_exp = float(np.exp(-6.0 * residual_cv) * (1.0 - trend_penalty))
        stability_exp = float(np.clip(stability_exp, 0.0, 1.0))

        stability_cv = float(np.clip(1.0 - cv, 0.0, 1.0))

        result = {
            "tempo_var":     var_tempo,
            "stability_exp": stability_exp,
            "stability_cv":  stability_cv,
        }
        self._cache_time["rhythmic_stability"] = result
        return result
    
    def _beat_periodicity_entropy(self, num_bins: int = 20) -> float:
        """ 
        Entropy-based beat periodicity from IOI histogram
        """
        ioi = self._ioi_values()
        if ioi.size < 3:
            return 0.0

        # Limit IOI range to sensible beat intervals
        ioi_clipped = ioi[(ioi > 0.1) & (ioi < 2.0)]
        if ioi_clipped.size < 3:
            return 0.0
        
        hist, edges = np.histogram(ioi_clipped, bins=num_bins, density=False)
        total = np.sum(hist)
        if total == 0:
            return 0.0

        p = hist.astype(float)/float(total)
        p = p[p > 0.0]

        # Shannon entropy (bits)
        H = -np.sum(p*np.log2(p))

        H_max = np.log2(num_bins)
        if H_max <= 0:
            return 0.0

        periodicity = 1.0 - (H/H_max)
        periodicity = float(np.clip(periodicity, 0.0, 1.0))
        return periodicity 
    
    # Correlation/Structure
    def _autocorrelation(self) -> np.ndarray:
        """ 
        Normalized autocorrelation of the whole signal
        """
        if "ac_full" in self._cache_time:
            return self._cache_time["ac_full"]
        
        x = self.y.astype(float)
        if x.size == 0:
            self._cache_time["ac_full"] = np.array([], dtype=float)
            return self._cache_time["ac_full"]
        
        x = x - np.mean(x)
        ac_full = np.correlate(x, x, mode='full')
        ac = ac_full[ac_full.size//2:]

        if ac[0] > 0:
            ac = ac/ac[0]

        self._cache_time["ac_full"] = ac
        return ac 
    
    def _autocorrelation_peaks(self, min_lag: int = 1) -> dict:
        """ 
        Return locations (lags) and values of local maxima in autocorrelation 
        for lags >= min_lag
        """
        ac = self._autocorrelation()
        if ac.size <= min_lag + 2:
            return {"lags": np.array([], dtype=int),
                    "values": np.array([], dtype=float)}
        
        # Consider only lags from min_lag to end
        ac_sub = ac[min_lag:]
        peaks, _ = find_peaks(ac_sub)

        if peaks.size == 0:
            return {"lags": np.array([], dtype=int),
                    "values": np.array([], dtype=float)}
        
        lags = peaks + min_lag
        values = ac[lags]

        return {"lags": lags.astype(int),
                "values": values.astype(float)}
    
    def _lag_k_correlation(self, k: int) -> float:
        """ 
        Normalized lag-k autocorrelation coefficient
        """
        x = self.y.astype(float)
        N = x.size
        if N <= k or k < 1:
            return 0.0
        
        mu = np.mean(x)
        x_centered = x - mu

        num = np.sum(x_centered[:N - k]*x_centered[k:])
        den = np.sum(x_centered*x_centered)

        if den <= 0:
            return 0.0
        
        return float(num/den)
    
    def _lag1_lag2_correlations(self) -> dict:
        """ 
        Convenience wrapper returning lag-1 and lag-2 correlation
        """
        r1 = self._lag_k_correlation(1)
        r2 = self._lag_k_correlation(2)

        return {"lag1": r1,
                "lag2": r2}
    
    def _frames(self) -> np.ndarray:
        """ 
        Return framed signal: shape (num_frames, frame_length)
        Uses frame length self.N and hop self.H
        """
        if "frames" in self._cache_time:
            return self._cache_time["frames"]
        
        L = len(self.y)
        if L < self.N:
            self.cache_time["frames"] = np.empty((0, self.N), dtype=float)
            return self._cache_time["frames"]
        
        num_frames = 1 + (L - self.N)//self.H
        frames = np.zeros((num_frames, self.N), dtype=float)

        for i in range(num_frames):
            start = i*self.H
            end = start + self.N
            frames[i, :] = self.y[start:end]

        self._cache_time["frames"] = frames
        return frames
    
    def _self_similarity_matrix(self) -> np.ndarray:
        """ 
        Time-domain self-similarity matrix using cosine similarity between frames.
        S[k, 1] in [-1, 1], with 1 = identical (up to scaling), 0 = orthogonal
        """
        if "self_sim" in self._cache_time:
            return self._cache_time["self_sim"]
        
        frames = self._frames()
        if frames.shape[0] == 0:
            self._cache_time["self_sim"] = np.zeros((0, 0), dtype=float)
            return self._cache_time["self_sim"]
        
        # Optionally mean-center each frame
        frames = frames - np.mean(frames, axis=1, keepdims=True)

        # Compute norms
        norms = np.linalg.norm(frames, axis=1, keepdims=True)
        norms[norms == 0] = 1.0

        # Normalize frames to unit length
        frames_norm = frames/norms

        # Cosine similarity matrix: S = F*F^T
        S = frames_norm @ frames_norm.T

        self._cache_time["self_sim"] = S.astype(float)
        return self._cache_time["self_sim"]
    
    # Complexity/Texture
    def _lz_complexity(self) -> float:
        """
        Lempel-Ziv complexity (normalized) of the signal
        1) Binary quantization around median
        2) Standard LZ76 parsing
        
        Return C_norm in (0, 1], where higher = more complex)
        """
        if "lz_complexity" in self._cache_time:
            return self._cache_time["lz_complexity"]
        
        x = self.y.astype(float)
        n = len(x)  # ← Use signal length, NOT frame length
        if n < 2:
            self._cache_time["lz_complexity"] = 0.0
            return 0.0
        
        # Binary symbolic sequence
        thr = np.median(x)
        s = (x > thr).astype(int)
        seq = "".join(s.astype(str))
        
        # LZ76 parsing (unchanged)
        i = 0
        c = 1
        k = 1
        while True:
            if i + k > n:  # ← Use n, not N
                c += 1
                break
            sub = seq[i:i + k]
            if seq[:i + k - 1].find(sub) != -1:
                k += 1
            else:
                c += 1
                i += k
                k = 1
            if i + k > n:  # ← Use n, not N
                break
        
        # Normalize by signal length, not frame length
        c = float(c)
        if n > 1:
            c_norm = c / (n / np.log(n))  # ← Use n
        else:
            c_norm = 0.0
    
        c_norm = float(max(0.0, c_norm))
        self._cache_time["lz_complexity"] = c_norm
        return c_norm 
    
    def _higuchi_fd(self, k_max: int = 8) -> float:
        """
        Higuchi fractal dimension of the time series
        Higher ~ more jagged/noise-like; 1 ~ smooth, 2 ~ rough
        """
        if "higuchi_fd" in self._cache_time:
            return self._cache_time["higuchi_fd"]
        
        x = self.y.astype(float)
        n = len(x)  # ← Use signal length
        if n < 2 or k_max < 2:
            self._cache_time["higuchi_fd"] = 1.0  # ← Return 1.0, not 0.0 (smooth signal)
            return 1.0
        
        k_max = min(k_max, n - 1)
        Lk = []
        ln_k = []
        
        for k in range(1, k_max + 1):
            Lm = []
            for m in range(k):
                idxs = np.arange(m, n, k)  # ← Use n
                if idxs.size < 2:
                    continue
                
                x_m = x[idxs]
                diff = np.abs(np.diff(x_m)).sum()
                n_m = idxs.size
                
                # Higuchi length
                L_mk = (diff * (n - 1) / ((n_m - 1) * k)) / k  # ← Use n
                Lm.append(L_mk)
            
            if len(Lm) == 0:
                continue
            
            Lk.append(np.mean(Lm))
            ln_k.append(np.log(1.0 * k))
        
        Lk = np.array(Lk, dtype=float)
        ln_k = np.array(ln_k, dtype=float)
        
        if Lk.size < 2:
            self._cache_time["higuchi_fd"] = 1.0  # ← Default to smooth
            return 1.0
        
        # Guard against zero/negative Lk (happens for constant signals)
        if np.any(Lk <= 0):
            self._cache_time["higuchi_fd"] = 1.0
            return 1.0
        
        ln_Lk = np.log(Lk + EPS)
        
        # Linear fit
        A = np.vstack([ln_k, np.ones_like(ln_k)]).T
        b, a = np.linalg.lstsq(A, ln_Lk, rcond=None)[0]
        fd = -float(b)
        
        # Clamp to valid range [1, 2]
        fd = float(np.clip(fd, 1.0, 2.0))
        
        self._cache_time["higuchi_fd"] = fd
        return fd
    
    def _hjorth_parameters(self) -> dict:
        """
        Hjorth activity, mobility, and complexity for self.y

        Returns dict with keys: "activity", "mobility", "complexity"
        """
        if "hjorth" in self._cache_time:
            return self._cache_time["hjorth"]
        
        x = self.y.astype(float)
        N = len(x)
        if N < 3:
            hj = {"activity": 0.0, "mobility": 0.0, "complexity": 0.0}
            self._cache_time["hjorth"] = hj
            return hj
        
        # Activity: variance of the signal
        x_mean = np.mean(x)
        var_x = np.mean((x - x_mean)**2)

        # First derivative
        dx = np.diff(x)
        dx_mean = np.mean(dx)
        var_dx = np.mean((dx - dx_mean)**2)

        # Second derivative
        ddx = np.diff(dx)
        ddx_mean = np.mean(ddx)
        var_ddx = np.mean((ddx - ddx_mean)**2)

        # Mobility and complexity
        if var_x <= EPS or var_dx <= EPS:
            activity = float(var_x)
            mobility = 0.0
            complexity = 0.0
        else:
            activity = float(var_x)
            mobility = float(np.sqrt(var_dx/var_x))
            complexity = float(
                np.sqrt((var_ddx/var_dx)/(var_dx/var_x))
            )

        hj = {
            "activity": activity,
            "mobility": mobility,
            "complexity": complexity,
        }
        self._cache_time["hjorth"] = hj
        return hj
    
    # Silence Structure
    def _silence_threshold(self, db_threshold: float = -60.0) -> float:
        """
        Energy threshold for silence using a dB offset below max frame energy
        db_threshold is in dB relative to max STE
        """
        ste = self._short_time_energy()
        if ste.size == 0:
            return 0.0
        
        max_e = np.max(ste) + EPS

        # Convert dB relative: E_thr = max_e*10^(dB/10)
        thr = max_e*(10.0**(db_threshold/10.0))

        return thr
    
    def _silence_mask(self, db_threshold: float = -60.0) -> np.ndarray:
        """
        Boolean mask of silent frames (True = silent)
        """
        ste = self._short_time_energy()
        if ste.size == 0:
            return np.zeros(0, dtype=bool)
        
        thr = self._silence_threshold(db_threshold=db_threshold)
        silent = ste <= thr

        return silent
    
    def _silence_ratio(self, db_threshold: float = -60.0) -> float:
        """
        Fraction of frames classified as silent
        """
        ste = self._short_time_energy()
        if ste.size == 0:
            return 0.0
        
        silent = self._silence_mask(db_threshold=db_threshold)
        ratio = float(silent.sum())/float(ste.size)

        return ratio
    
    def _silence_duration(self, db_threshold: float = -60.0) -> float:
        """
        Silence duration stats (in seconds) based on contiguous runs of silent frames

        Returns dict: total, max, mean, count
        """
        ste = self._short_time_energy()
        if ste.size == 0:
            return {"total": 0.0, "max": 0.0, "mean": 0.0, "count": 0}
        
        silent = self._silence_mask(db_threshold=db_threshold)
        if not silent.any():
            return {"total": 0.0, "max": 0.0, "mean": 0.0, "count": 0}
        
        # Frame hop in seconds
        hop_sec = self.H/float(self.sr)

        # Find contiguous silent runs
        durations = []
        in_run = False
        start_idx = 0

        for k, is_silent in enumerate(silent):
            if is_silent and not in_run:
                in_run = True
                start_idx = k
            elif not is_silent and in_run:
                in_run = False
                end_idx = k - 1
                len_frames = end_idx - start_idx + 1
                durations.append(len_frames*hop_sec)

        # If ends in silence, close the last run
        if in_run:
            end_idx = len(silent) - 1
            len_frames = end_idx - start_idx + 1
            durations.append(len_frames*hop_sec)

        durations = np.array(durations, dtype=float)
        total_sil = float(durations.sum())
        max_sil = float(durations.max()) if durations.size > 0 else 0.0
        mean_sil = float(durations.mean()) if durations.size > 0 else 0.0
        count_sil = int(durations.size)

        return {
            "total": total_sil,
            "max": max_sil,
            "mean": mean_sil,
            "count": count_sil,
        }
    
    def _low_energy_frame_ratio(self, alpha: float = 0.5) -> float:
        """
        Fraction of frames whose energy is <= alpha*mean_energy
        alpha in (0, 1); alpha = 0.5 -> frames with energy <= 50% of mean energy
        """
        ste = self._short_time_energy()
        if ste.size == 0:
            return 0.0
        
        mean_e = float(np.mean(ste))
        if mean_e <= EPS:
            return 1.0
        
        thr_low = alpha*mean_e
        low = ste <= thr_low
        ratio = float(low.sum())/float(ste.size)

        return ratio

    # Spotify-based features (cached)
    def loudness_dB(self) -> float:
        """
        Global loudness in dB
        """
        if "loudness_dB" in self._cache_time:
            return self._cache_time["loudness_dB"]
        
        ldB = self._global_loudness_dB()
        if ldB is None:
            ldB = -80.0
        else:
            # Clamp to a sane range
            ldB = float(np.clip(ldB, -80.0, 0.0))

        self._cache_time["loudness_dB"] = ldB
        return ldB
    
    def loudness_norm(self) -> float:
        """
        Loudness mapped to [0, 1] using a fixed dB range
        """
        if "loudness_norm" in self._cache_time:
            return self._cache_time["loudness_norm"]
        
        ldB = self.loudness_dB()

        # Clamp to [-60, 0] then normalize
        ldB = np.clip(ldB, -60.0, 0.0)
        ln = (ldB + 60.0)/60.0
        ln = safe_clip01(ln)

        self._cache_time["loudness_norm"] = ln
        return ln
    
    def energy_partial(self) -> float:
        """
        Approximate Spotify "energy" using:
        - mean active RMS
        - dynamic range

        Returns value in [0, 1], higher = more energetic
        """
        if "energy_partial" in self._cache_time:
            return self._cache_time["energy_partial"]
        
        # Mean active RMS
        rms_env = self._rms_envelope()
        mask = self._active_rms_mask(db_threshold=-60.0)
        active_rms = rms_env[mask] if mask.any() else rms_env
        if active_rms.size == 0:
            self._cache_time["energy_partial"] = 0.0
            return 0.0
        
        mean_active_rms = float(np.mean(active_rms))
        max_rms = float(np.max(rms_env))

        # Normalize mean_active_rms by max RMS
        rms_norm = mean_active_rms / (max_rms + EPS)
        rms_norm = safe_clip01(rms_norm)

        # Dynamic range
        dr = float(self._dynamic_range())

        # Normalize DR: 0dB -> 0, 20+dB -> ~1
        dr_norm = safe_clip01(dr/20.0)

        # Energy heuristic: combine with tunable weights
        w_rms = 0.4
        w_dr = 0.6

        energy = w_rms*rms_norm + w_dr*dr_norm
        energy = safe_clip01(energy)

        ldB = self.loudness_dB()

        # Extra: damp energy for very quiet tracks
        if ldB <= -35.0:
            # At -35dB -> factor ~1, at -50dB -> factor ~0
            factor = np.clip((ldB + 50.0)/15.0, 0.0, 1.0)
            energy *= factor * 0.9

        # Live/"moderate" material with decent DR should not drop below ~0.3-0.4
        if ldB > -30.0 and dr > 6.0 and energy < 0.3:
            energy = 0.3

        # Compressed pop pattern: loud and low DR -> ensure at least moderate energy
        if ldB > -20.0 and dr < 6.0 and energy < 0.45:
            energy = 0.45

        # Loud constant-tone-like: very low DR -> cap energy lower
        if ldB > -10.0 and dr < 2.0:
            energy = min(energy, 0.40)

        # If clearly acoustic, cap energy a bit lower (acoustic guitar, etc.)
        a = self.acousticness_partial()
        if a >= 0.6:
            energy = min(energy, 0.65)

        energy = safe_clip01(energy)

        self._cache_time["energy_partial"] = energy
        return energy
    
    def speechiness(self) -> float:
        """
        Approximate speechiness using:
        - voiced ratio (fraction of frames with clear pitch)
        - zcr_variance (changing spectral character)
        - silence_ratio (pause)
        - onset_rate (articulation)
        """
        if "speechiness" in self._cache_time:
            return self._cache_time["speechiness"]
        
        # Voiced Ratio
        vr = float(self._voiced_ratio())
        ur = 1.0 - vr

        voiced_balance = 4.0*vr*ur
        voiced_balance = safe_clip01(voiced_balance)

        # ZCR Variance
        zcr_var = float(self._zcr_variance())
        zcr_var_norm = np.tanh(zcr_var)

        # Silence Ratio
        sil_ratio = float(self._silence_ratio(db_threshold=-60.0))

        # Onset Rate
        onset_rate = float(self._onset_rate())

        # Typically speech 8-15 onsets/sec -> map this band to ~[0, 1]
        onset_norm = (onset_rate - 4.0)/(16.0 - 4.0)
        onset_norm = safe_clip01(onset_norm)

        # Speechiness heuristic: combine with tunable weights
        w_vb = 0.35
        w_zcr = 0.20
        w_onset = 0.25
        w_sil = 0.20

        speechiness = (
            w_vb*voiced_balance +
            w_zcr*zcr_var_norm +
            w_onset*onset_norm +
            w_sil*(sil_ratio if sil_ratio < 0.5 else (1.0 - sil_ratio))
        )
        speechiness = safe_clip01(speechiness)

        self._cache_time["speechiness"] = speechiness
        return speechiness
    
    def acousticness_partial(self) -> float:
        """
        Approximate acousticness using:
        - dynamic range
        - transient ratio
        - silence ratio
        - energy variance
        - inverse loudness_norm
        """
        if "acousticness_partial" in self._cache_time:
            return self._cache_time["acousticness_partial"]
        
        dr = float(self._dynamic_range())
        dr_norm = safe_clip01(dr / 30.0)  # 0-30dB → 0-1

        tr = float(self._transient_rate())
        inv_tr = 1.0 - safe_clip01(tr / 15.0)

        # Silence ratio from STE
        sil_ratio = float(self._silence_ratio(db_threshold=-60.0))

        # Energy variance
        ev = float(self._energy_variance())
        ev_norm = np.tanh(ev)

        loud_norm = self.loudness_norm()
        inv_loud = 1.0 - loud_norm

        # Acousticness heuristic: combine with tunable weights
        w_dr = 0.30
        w_tr = 0.20
        w_sr = 0.20
        w_ev = 0.20
        w_inv_ld = 0.10

        acousticness = (
            w_dr*dr_norm +
            w_tr*inv_tr +
            w_sr*sil_ratio +
            w_ev*ev_norm +
            w_inv_ld*inv_loud
        )
        acousticness = safe_clip01(acousticness)

        self._cache_time["acousticness_partial"] = acousticness
        return acousticness
    
    def danceability_partial(self) -> float:
        """
        Approximate danceability using:
        - pulse clarity
        - rhythimic stability
        - BPM proximity (to 120 BPM)
        - onset rate
        - inverse silence ratio
        """
        if "danceability_partial" in self._cache_time:
            return self._cache_time["danceability_partial"]

        # Pulse Clarity
        try:
            pc = float(self._pulse_clarity_ac())
        except AttributeError:
            pc = 0.0

        # Rhythmic Stability
        try:
            rs = self._rhythmic_stability()
            stab = float(rs["stability_exp"])
        except AttributeError:
            stab = 1.0

        # BPM Proximity: how close is the tempo to 120 BPM?
        bpm = float(self._tempo_from_onset_ac())

        # Map BPM to [0, 1], peaking around 100-140 BPM
        if bpm <= 0:
            bpm_score = 0.0
        else:
            # Triangular window: 60 BPM -> 0, 100 BPM -> 1, 120 BPM -> 1, 140 BPM -> 1, 180 BPM -> 0
            if bpm < 60 or bpm > 180:
                bpm_score = 0.0
            elif bpm <= 100:
                bpm_score = (bpm - 60.0)/(100.0 - 60.0)
            elif bpm <= 140:
                bpm_score = 1.0
            else:
                bpm_score = (180.0 - bpm)/(180.0 - 140.0)

        bpm_score = safe_clip01(bpm_score)

        # Onset Rate
        onset_rate = float(self._onset_rate())

        # For dance, moderate to high onset rates, 1-8 onsets/sec, are typical
        onset_norm = (onset_rate - 1.0)/(8.0 - 1.0)
        onset_norm = safe_clip01(onset_norm)

        # Inverse Silence Ratio
        sil_ratio = float(self._silence_ratio(db_threshold=-60.0))

        inv_sil = 1.0 - sil_ratio

        # Danceability heuristic: combine with tunable weights
        w_pc = 0.40
        w_stab = 0.25
        w_bpm = 0.15
        w_onset = 0.10
        w_sil = 0.10

        danceability = (
            w_pc*pc +
            w_stab*stab +
            w_bpm*bpm_score +
            w_onset*onset_norm +
            w_sil*inv_sil
        )
        danceability = safe_clip01(danceability)
        
        # Gate on very weak pulse or stability
        if pc < 0.25 or stab < 0.5:
            danceability = min(danceability, 0.25)

        # Penalize highly acoustic, non-electronic material
        a = self.acousticness_partial()
        ldB = self.loudness_dB()
        if a >= 0.6 and ldB > -35.0:
            danceability *= 0.6

        # Strongly reduce when tempo is unstable (rubato-like)
        try:
            rs = self._rhythmic_stability()
            stab_cv = float(rs.get("stability_cv", 1.0))
        except Exception:
            stab_cv = 1.0

        # Rubato-like: sharply reduce
        if stab_cv < 0.6:
            danceability *= 0.30  # was 0.35

        # Live/unstable recordings: cap to moderate danceability
        if stab_cv < 0.7:
            danceability = min(danceability, 0.70)

        # Extra clamp for clearly rubato (very unstable)
        if stab_cv < 0.5:
            danceability = min(danceability, 0.40)

        danceability = safe_clip01(danceability)

        self._cache_time["danceability_partial"] = danceability
        return danceability
    
    def tempo_partial(self) -> float:
        """
        Direct wrapper for tempo_from_onset_ac, mapped to [0, 1] with a peak around 120 BPM
        """
        if "tempo_partial" in self._cache_time:
            return self._cache_time["tempo_partial"]
        
        bpm = float(self._tempo_from_onset_ac())
        
        self._cache_time["tempo_partial"] = bpm
        return bpm
    
    def liveness_partial(self) -> float:
        """
        Approximate liveness using:
        - inverse rhythmic stability (live performances often have more tempo variation)
        - silence ratio (live recordings may have more audience noise/silence)
        - energy variance (live recordings may have more dynamic variation)
        - transient ratio (live recordings may have more pronounced transients)
        """
        if "liveness_partial" in self._cache_time:
            return self._cache_time["liveness_partial"]
        
        # Rhythmic Stability
        try:
            rs = self._rhythmic_stability()
            stab = float(rs.get("stability_cv", 1.0))
        except Exception:
            stab = 1.0
        inv_stab = 1.0 - safe_clip01(stab)

        # Silence Ratio
        sil_ratio = float(self._silence_ratio(db_threshold=-60.0))

        # Energy Variance
        ev = float(self._energy_variance())
        ev_norm = np.tanh(ev)

        # Transient Ratio
        tr = float(self._transient_rate())
        tr_norm = safe_clip01(tr/10.0)

        # Base liveness heuristic
        w_stab = 0.60
        w_sr = 0.20
        w_ev = 0.15
        w_tr = 0.05

        liveness = (
            w_stab*inv_stab +
            w_sr*sil_ratio +
            w_ev*ev_norm +
            w_tr*tr_norm
        )

        # Extra boost for rubato-like acoustic content
        a = self.acousticness_partial()
        if a >= 0.5 and stab < 0.6:
            liveness += 0.20  # was 0.15

        # Live-like: not very quiet and unstable tempo -> ensure at least moderate liveness
        ldB = self.loudness_dB()
        if ldB > -35.0 and stab < 0.7:
            liveness = max(liveness, 0.45)  # was 0.4

        liveness = safe_clip01(liveness)
        self._cache_time["liveness_partial"] = liveness
        return liveness
    
    def instrumentalness(self) -> float:
        """
        Simple complement of speechness
        """
        if "instrumentalness" in self._cache_time:
            return self._cache_time["instrumentalness"]
        
        speech = self.speechiness()
        instrumentalness = 1.0 - speech

        instrumentalness = safe_clip01(instrumentalness)
        self._cache_time["instrumentalness"] = instrumentalness
        return instrumentalness
    
    def time_signature_partial(self) -> tuple:
        """
        Very rough guess of "beats per bar" using IOIs and onset AC

        Returns (estimated_time_signature, confidence) where:
        - estimated_time_signature: The guessed time signature (e.g., 4 for 4/4 time)
        - confidence: A measure of how confident the estimate is (between 0 and 1)
        """
        if "time_signature_partial" in self._cache_time:
            return self._cache_time["time_signature_partial"]
        
        # Use IOI mean as beat period esitmate
        mean_ioi, _, cv_ioi = self._ioi_stats()
        if mean_ioi <= 0 or cv_ioi > 0.8:
            self._cache_time["time_signature_partial"] = (0, 0.0)
            return (0, 0.0)
        
        # Onset AC
        ac = self._onset_autocorrelation()
        if ac.size < 3:
            self._cache_time["time_signature_partial"] = (0, 0.0)
            return (0, 0.0)
        
        fs_env = self.sr/float(self.H)

        # Convert 2, 3, 4 beats per bar into candidate bar-period lags
        candidates = [2, 3, 4]
        best_m = 0
        best_conf = 0.0

        for m in candidates:
            bar_period_sec = m*mean_ioi
            tau = int(round(bar_period_sec*fs_env))
            if tau <= 0 or tau >= ac.size:
                continue
            val = float(ac[tau])
            if val > best_conf:
                best_conf = val
                best_m = m

        # Normalize confidence by max nonzero lag
        if ac[1:].size > 0:
            max_ac = float(np.max(ac[1:]))
        else:
            max_ac = 0.0

        if max_ac > 0:
            conf = best_conf/max_ac
        else:
            conf = 0.0

        conf = safe_clip01(conf)

        self._cache_time["time_signature_partial"] = (best_m, conf)
        return (best_m, conf)

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
            self._cache_spec["loudness_stft"] = -60.0
            return -60.0
             
        # Mean power across freq per frame
        frame_pow = np.mean(P, axis=0) + EPS

        # Convert power -> amplitude
        frame_rms = np.sqrt(frame_pow)

        ref_rms = np.max(frame_rms) + EPS

        # Convert to dBFS-like scale
        loudness_frames = 20.0*np.log10(frame_rms/(ref_rms + EPS))

        # Aggregate
        loudness = float(np.mean(loudness_frames))
        loudness = float(np.clip(loudness, -60.0, 0.0))

        self._cache_spec["loudness_stft"] = loudness
        return loudness
    
    def energy_stft(self) -> float:
        if getattr(self, "invalid", False):
            return 0.0
        
        if "energy_stft" in self._cache_spec:
            return self._cache_spec["energy_stft"]
        
        # Extract features
        loud = self.loudness_stft()
        E = self._stft_energy()
        flux = self._onset_env()
        flat = self._spectral_flatness()
        ratio = self._band_ratio()

        # Normalize energy to 0-1
        E_log = np.log10(E + EPS)
        E_min = np.percentile(E_log, 10)
        E_max = np.percentile(E_log, 90)
        E_norm = np.clip((E_log - E_min)/(E_max - E_min + EPS), 0.0, 1.0)

        # Normalize Band Ratio
        ratio_norm = np.tanh(ratio)

        flux_n = robust_normalize(flux)
        loud_n = np.clip((loud + 20)/20, 0.0, 1.0)

        # Weights
        w_loud = 0.30
        w_E = 0.30
        w_flux = 0.20
        w_flat = 0.10
        w_ratio = 0.10

        energy_frame = (
            w_loud*loud_n +
            w_E*E_norm +
            w_flux*flux_n +
            w_flat*flat +
            w_ratio*ratio_norm
        )

        # Smooth slightly 
        smooth = np.convolve(energy_frame, np.ones(5)/5.0, mode='same')

        # Final Scalar
        energy = float(np.median(smooth))

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

        onset = self._onset_env()

        if onset.size < 16 or np.all(onset == 0):
            self._cache_spec["danceability_stft"] = 0.1
            return 0.1

        # ---- 1) RHYTHM & TEMPO FOUNDATION ----
        rhythm_conf = self._rhythm_confidence()
        if rhythm_conf < 0.08:
            rhythm_conf = 0.08

        tempo = self.tempo_stft()

        groove_center = 122.0
        groove_width = 32.0
        groove_score = np.exp(-0.5 * ((tempo - groove_center) / groove_width) ** 2)

        # ---- 2) PULSE CLARITY ----
        ac = np.correlate(onset, onset, mode="full")
        ac = ac[ac.size // 2 + 1 :]

        if np.max(ac) > 0:
            ac = ac / (np.max(ac) + 1e-9)

        main_peak = np.max(ac)
        mean_rest = np.mean(ac)
        pulse_clarity = np.clip((main_peak - mean_rest) / (main_peak + EPS), 0.0, 1.0)
        pulse_clarity = np.sqrt(pulse_clarity)

        # ---- 3) PERIODICITY STABILITY ----
        lags = np.arange(1, len(ac) + 1)
        hop_time = self.H / float(self.sr)
        bpms = 60.0 / (lags * hop_time + 1e-9)

        mask = (bpms >= 70) & (bpms <= 160)
        if np.any(mask):
            periodicity = float(np.mean(ac[mask]))
        else:
            periodicity = float(np.mean(ac))

        periodicity = np.clip(periodicity, 0.0, 1.0)

        # ---- 4) SPECTRAL CONTROL ----
        flat = self._spectral_flatness()
        band = self._band_ratio()

        flat_score = 1.0 - np.clip(flat/1.0, 0.0, 1.0)
        band_score = 1.0 - np.clip(np.tanh(band), 0.0, 1.0)

        # ---- 5) ENERGY CONSISTENCY ----
        energy = self._stft_energy()
        E_mean = np.mean(energy) + 1e-9
        E_std = np.std(energy)
        consistency = 1.0 - np.clip(E_std/(2.0*E_mean), 0.0, 1.0)

        # ---- 6) COMBINE ----
        w_groove = 0.28
        w_pulse = 0.20
        w_period = 0.18
        w_flat = 0.10
        w_band = 0.10
        w_cons = 0.14

        dance = (
            w_groove * groove_score +
            w_pulse * pulse_clarity +
            w_period * periodicity +
            w_flat * flat_score +
            w_band * band_score +
            w_cons * consistency
        )

        # ---- 7) CONFIDENCE MODULATION (fixed) ----
        dance = float(np.mean(dance))                   # already scalar now
        dance = dance * (0.5 + 0.5 * rhythm_conf)
        dance = 0.25 + 0.75*dance
        dance = float(np.clip(dance, 0.0, 1.0))

        self._cache_spec["danceability_stft"] = dance
        return dance

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
    
    def tempo_stft(self,
               bpm_min: float = 40.0,
               bpm_max: float = 220.0) -> float:

        if getattr(self, "invalid", False):
            return 0.0

        if "tempo_stft" in self._cache_spec:
            return self._cache_spec["tempo_stft"]

        # 1. Compute STFT
        S_abs = self._amp_spectrogram()
        S_dB = librosa.amplitude_to_db(S_abs, ref=np.max)

        # 2. Keep only up to ~150Hz (core tempo region)
        low_mask = self._fft_freqs <= 150
        S_low = S_dB[low_mask, :]

        # 3. Onset Envelope from low band only
        onset_env = librosa.onset.onset_strength(
            S=S_low,
            sr=self.sr,
            hop_length=self.H
        )

        if onset_env is None or len(onset_env) < 8 or np.all(onset_env == 0):
            self._cache_spec["tempo_stft"] = 0.0
            return 0.0
        
        onset_env = onset_env - np.mean(onset_env)
        ac = np.correlate(onset_env, onset_env, mode="full")
        ac = ac[len(ac)//2:]
        ac = ac/(np.max(ac) + EPS)

        if len(ac) < 8:
            self._cache_spec["tempo_stft"] = 0.0
            return 0.0
        
        # Convert lags to BPM grid
        hop_time = self.H/float(self.sr)

        lags = np.arange(1, len(ac))
        ac_lag = ac[1:]

        bpms = 60.0/(lags*hop_time + EPS)

        mask = (bpms >= bpm_min) & (bpms <= bpm_max)
        if not np.any(mask):
            self._cache_spec["tempo_stft"] = 0.0
            return 0.0
        
        bpms = bpms[mask]
        strengths = ac_lag[mask]

        # normalize strengths
        strengths = strengths - np.min(strengths)
        strengths = strengths/(np.max(strengths) + EPS)

        votes = []
        for bpm, s in zip(bpms, strengths):
            candidates = [bpm/2.0, bpm, bpm*2.0]
            for c in candidates:
                if bpm_min <= c <= bpm_max:
                    votes.append((c, s))

        if not votes:
            self._cache_spec["tempo_stft"] = 0.0
            return 0.0
        
        # Perceptual weighting
        def perceptual_weight(bpm):
            return np.exp(-((bpm - 115.0)**2)/(2*(30.0**2)))
        
        bins = {}
        for bpm, s in votes:
            b = int(round(bpm))
            w = s*perceptual_weight(b)

            bins.setdefault(b, 0.0)
            bins[b] += w 

        best_bpm = float(max(bins, key=bins.get))

        # Neighbor Smoothing
        neighbor_vals = []
        weights = []

        for k, v in bins.items():
            w = 1.0/(1.0 + abs(k - best_bpm))
            neighbor_vals.append(k*w*v)
            weights.append(w*v)

        tempo_est = float(
            np.sum(neighbor_vals)/(np.sum(weights) + EPS)
        )

        # Soft Guardrail
        tempo_est = float(np.clip(tempo_est, 60.0, 180.0))

        self._cache_spec["tempo_stft"] = tempo_est
        return tempo_est
    
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
        w_flat = 0.20
        w_ent  = 0.25
        w_inh  = 0.15
        w_roll = 0.15
        w_dyn  = 0.25
        w_ir   = 0.15

        frame_score = (w_flat * flat_n +
                    w_ent  * entr_n +
                    w_inh  * inh_n +
                    w_roll * roll_n)

        track_score = (0.6 * float(np.median(frame_score)) +
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
        speech_like = 0.5*robust_normalize(pitch_var) + \
                      0.5*robust_normalize(onset_diff)
        
        speech_gate = np.clip(1.0 - 0.6*speech_like, 0.2, 1.0)

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
        )*(1.0 - hr_n)
        I_noise = np.clip(I_noise, 0.0, 1.0)

        # ---------- Fuse ----------
        inst_struct_score = np.percentile(I_struct*speech_gate, 80)
        inst_noise_score  = np.percentile(I_noise*speech_gate, 70)

        instrumentalness = 0.6*inst_struct_score + 0.4*inst_noise_score 
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

            downbeat = proto[0]
            others = np.mean(proto[1:]) + EPS
            accent_ratio = downbeat/others

            clarity = np.clip((accent_ratio - 1.0)/2.0, 0.0, 1.0)
            clarity *= np.clip(np.std(proto)/0.1, 0.0, 1.0)

            penalty = 0.9 if m in (5, 7) else 1.0
            score = clarity*penalty

            if score > best_score:
                best_score = score
                best_m = m

        if best_score < 0.25:
            best_m = 4

        self._cache_spec["time_signature_stft"] = int(best_m)
        return int(best_m)