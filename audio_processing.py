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
                            bpm_min: float = 70.0,    # ✅ Tightened from 40
                            bpm_max: float = 170.0) -> float:  # ✅ Tightened from 240
        """
        Estimate global tempo (BPM) from onset autocorrelation
        """
        ac = self._onset_autocorrelation()
        if ac.size < 3:
            return 0.0

        # Map BPM range to lag range
        fs_env = self.sr / float(self.H)

        # Convert BPM bounds to lags
        f_min = bpm_min / 60.0
        f_max = bpm_max / 60.0

        lag_min = int(fs_env / f_max)
        lag_max = int(fs_env / f_min)

        lag_min = max(lag_min, 1)
        lag_max = min(lag_max, ac.size - 1)
        if lag_max <= lag_min:
            return 0.0
        
        # Search for AC peaks in this lag region
        search_region = ac[lag_min:lag_max + 1]
        
        peaks, properties = find_peaks(search_region, height=0.1)

        if len(peaks) == 0:
            rel_peak_idx = int(np.argmax(search_region))
            tau_peak = lag_min + rel_peak_idx
            tempo_bpm = 60 * fs_env / float(tau_peak)
            return float(tempo_bpm)

        # Convert peak indices to absolute lags
        peak_lags = lag_min + peaks
        peak_strengths = search_region[peaks]

        votes = {}
        for lag, strength in zip(peak_lags, peak_strengths):
            # Weighted candidates: prefer actual peak
            candidates_weights = [
                (lag,     strength * 5.0),   # ✅ Actual peak (highest weight)
                (lag/2.0, strength * 0.2),   # Half-time (low weight)
                (lag*2.0, strength * 0.2),   # Double-time (low weight)
            ]

            for cand_lag, cand_weight in candidates_weights:
                cand_bpm = 60.0 * fs_env / cand_lag
                # ✅ This check now filters out-of-range half/double candidates!
                if bpm_min <= cand_bpm <= bpm_max:
                    cand_lag_int = int(round(cand_lag))
                    votes.setdefault(cand_lag_int, 0.0)
                    votes[cand_lag_int] += cand_weight

        if not votes:
            tau_peak = peak_lags[np.argmax(peak_strengths)]
        else:
            tau_peak = max(votes, key=votes.get)

        tempo_bpm = 60.0 * fs_env / float(tau_peak)

        # Post-processing: Octave correction
        if 60 <= tempo_bpm < 75:
            # Suspiciously slow - try doubling
            doubled = tempo_bpm*2
            if 90 <= doubled <= 180:
                tempo_bpm = doubled
        elif 165 < tempo_bpm <= 180:
            # Suspiciously fast - try halving
            halved = tempo_bpm/2
            if 80 <= halved <= 100:
                tempo_bpm = halved

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
        if pc < 0.15:  # More lenient threshold
            danceability = min(danceability, 0.35)  # Higher ceiling

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

        # ✅ ADD DEBUG (temporarily):
        # print(f"[DEBUG Liveness] inv_stab={inv_stab:.3f}, sil={sil_ratio:.3f}, ev={ev_norm:.3f}, tr={tr_norm:.3f}")
        
        liveness = safe_clip01(liveness)
        
        # ✅ ADD: Cap at 0.35 for studio recordings (based on real data)
        if liveness > 0.35:
            liveness = 0.35  # Studio recordings shouldn't exceed this
            
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
        
        # Get beat from tempo
        tempo_bpm = self._tempo_from_onset_ac()
        if tempo_bpm <= 0:
            return (4, 0.15)
        
        beat_period_sec = 60.0/tempo_bpm

        # Check AC for bar-level patterns
        ac = self._onset_autocorrelation()
        if ac.size < 3:
            return (4, 0.15)
        
        fs_env = self.sr/float(self.H)

        # Search for bar patterns with multi-harmonic voting
        candidates = [3, 4]
        votes = {}

        # Adaptive window based on tempo
        window = max(int(round(0.15*fs_env)), 5)

        for m in candidates:
            bar_period_sec = m*beat_period_sec
            tau = int(round(bar_period_sec*fs_env))

            if tau <= 0 or tau >= ac.size:
                continue

            # Check peak at bar length
            start = max(1, tau - window)
            end = min(ac.size, tau + window + 1)
            local_region = ac[start:end]

            if local_region.size == 0:
                continue

            peak_val = float(np.max(local_region))

            # Also check harmonics (2x, 3x bar length) for reinforcement
            harmonic_bonus = 0.0
            for harmonic in [2, 3]:
                tau_h = int(round(harmonic*bar_period_sec*fs_env))
                if tau_h < ac.size:
                    start_h = max(1, tau_h - window)
                    end_h = min(ac.size, tau_h + window + 1)
                    if end_h > start_h:
                        harmonic_peak = float(np.max(ac[start_h:end_h]))
                        harmonic_bonus += harmonic_peak*0.2

            total_strength = peak_val + harmonic_bonus
            votes[m] = total_strength

        if not votes:
            return (4, 0.30)
        
        # Find best candidate
        best_m = max(votes, key=votes.get)
        best_conf = votes[best_m]

        # Normalize confidence
        max_ac = float(np.max(ac[1:])) if ac[1:].size > 0 else 0.0
        conf = best_conf/max_ac if max_ac > 0 else 0.0
        conf = safe_clip01(conf)

        # Strong 3/4 penalty
        if best_m == 3:
            conf *= 0.60
            if conf < 0.50:
                return (4, 0.30)
            
        # Lower default threshold
        if conf < 0.25:
            return (4, 0.30)
        
        result = (best_m, conf)
        self._cache_time["time_signature_partial"] = result
        return result

    # ---------- STFT-based features (cached) ----------
class FrequencyFeatures():
    def __init__(self, sig: AudioSignal):
        self.y = sig.y
        self.sr = sig.sr
        self.N = sig.N
        self.H = sig.H

        if len(self.y) < self.N:
            pad = self.N - len(self.y)
            self.y = np.pad(self.y, (0, pad), mode="constant")

        # Run STFT on signal
        self.X = librosa.stft(
            self.y,
            n_fft=self.N,
            hop_length=self.H,
            win_length=self.N,
            window="hann",
            center=True
        )

        self.freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.N)

        self._cache_freq = {}

    # Amplitude/Frequency
    def _magnitude_spectrum(self):
        if "mag" in self._cache_freq:
            return self._cache_freq["mag"]

        mag = np.abs(self.X)

        self._cache_freq["mag"] = mag
        return mag

    def _power_spectrum(self):
        if "pow" in self._cache_freq:
            return self._cache_freq["pow"]
        
        mag = self._magnitude_spectrum()
        power = mag**2

        self._cache_freq["pow"] = power
        return power
    
    def _db_spectrum(self, ref=None, power: bool=False):
        """
        dB spectrum from magnitude or power
        If power=False: 20*log10(|X|)
        if power=True: 10*log10(|X|^2)
        """
        key = f"db_{'pow' if power else 'map'}"
        if key in self._cache_freq:
            return self._cache_freq[key]

        if power:
            S = self._power_spectrum()
            if ref is None:
                ref = np.max(S) + EPS
            db = 10.0*np.log10(np.maximum(S, EPS)/ref)
        else:
            S = self._magnitude_spectrum()
            if ref is None:
                ref = np.max(S) + EPS
            db = 20.0*np.log10(np.maximum(S, EPS)/ref)

        self._cache_freq[key] = db
        return db
    
    def _frame_energy(self):
        if "frame_energy" in self._cache_freq:
            return self._cache_freq["frame_energy"]

        power = self._power_spectrum()
        frame_energy = np.sum(power, axis=0)

        self._cache_freq["frame_energy"] = frame_energy
        return frame_energy
    
    def _frame_energy_db(self):
        if "frame_energy_db" in self._cache_freq:
            return self._cache_freq["frame_energy_db"]

        e = self._frame_energy()
        e_db = 10.0*np.log10(e + EPS)

        self._cache_freq["frame_energy_db"] = e_db
        return e_db

    def _dynamic_range(self):
        if "dynamic_range" in self._cache_freq:
            return self._cache_freq["dynamic_range"]

        e_db = self._frame_energy_db()
        if e_db.size == 0:
            dr = 0.0
        else:
            dr = float(np.percentile(e_db, 90) - np.percentile(e_db, 10))

        self._cache_freq["dynamic_range"] = dr
        return dr

    def _band_energy(self, bands):
        """
        bands: list of (low_hz, high_hz) pairs
        Returns:
            band_energy: shape (n_bands, n_frames)
        """
        key = f"band_energy_{tuple(bands)}"
        if key in self._cache_freq:
            return self._cache_freq[key]
        
        power = self._power_spectrum()
        
        band_energy = np.zeros((len(bands), power.shape[1]), dtype=float)

        for i, (f_lo, f_hi) in enumerate(bands):
            mask = (self.freqs >= f_lo) & (self.freqs < f_hi)
            if np.any(mask):
                band_energy[i] = np.sum(power[mask, :], axis=0)
            else:
                band_energy[i] = 0.0

        self._cache_freq[key] = band_energy
        return band_energy
    
    def _band_energy_ratio(self, bands):
        """
        Relative energy in each band, normalized by total frame energy
        """
        key = f"band_energy_ratio_{tuple(bands)}"
        if key in self._cache_freq:
            return self._cache_freq[key]

        band_energy = self._band_energy(bands)
        total_energy = self._frame_energy()[None, :] + EPS
        ratio = band_energy/total_energy

        self._cache_freq[key] = ratio
        return ratio
    
    # Spectral Shape
    def _spectral_centroid(self, use_power=True):
        """
        Spectral Centroid per frame (Hz)
        """
        key = f"spectral_centroid_{'pow' if use_power else 'mag'}"
        if key in self._cache_freq:
            return self._cache_freq[key]
        
        S = self._power_spectrum() if use_power else self._magnitude_spectrum()
        freqs = self.freqs[..., None] # (K, 1)

        S_sum = np.sum(S, axis=0, keepdims=True) + EPS
        p = S/S_sum # Normalize

        centroid = np.sum(freqs*p, axis=0)

        self._cache_freq[key] = centroid
        return centroid

    def _spectral_bandwidth(self, use_power=True):
        """
        Spectral Bandwidth per frame (Hz), as STD Dev around centroid
        """
        key = f"spectral_bandwidth_{'pow' if use_power else 'mag'}"
        if key in self._cache_freq:
            return self._cache_freq[key]

        S = self._power_spectrum() if use_power else self._magnitude_spectrum()
        freqs = self.freqs[..., None]

        S_sum = np.sum(S, axis=0, keepdims=True) + EPS
        p = S/S_sum

        centroid = self._spectral_centroid(use_power=use_power)[None, :]
        var = np.sum(((freqs - centroid)**2)*p, axis=0)
        bw = np.sqrt(np.maximum(var, 0.0))

        self._cache_freq[key] = bw
        return bw
    
    def _spectral_rolloff(self, roll_percent=0.85, use_power=True):
        """
        Spectral rolloff per frame (Hz)
        roll_percent in (0, 1), e.g. 0.85 or 0.95
        """
        key = f"spectral_roll_{roll_percent}_{'pow' if use_power else 'mag'}"
        if key in self._cache_freq:
            return self._cache_freq[key]

        S = self._power_spectrum() if use_power else self._magnitude_spectrum()
        freqs = self.freqs
        K, T = S.shape

        rolloff_freq = np.zeros(T, dtype=float)
        thresh = float(roll_percent)

        for t in range(T):
            spec = S[:, t]
            total = np.sum(spec)
            if total <= 0.0:
                rolloff_freq[t] = 0.0
                continue
            cumsum = np.cumsum(spec)
            idx = np.searchsorted(cumsum, thresh*total)
            if idx >= K:
                idx = K - 1
            rolloff_freq[t] = freqs[idx]

        self._cache_freq[key] = rolloff_freq
        return rolloff_freq

    def _spectral_slope(self, use_power=True, log_amp=False):
        """
        Spectral Slope per frame, from linear regression of S over frequency

        If log_amp=True, regress on log(S); otherwise on S directly
        """
        key = f"spectral_slope_{'pow' if use_power else 'mag'}_{'log' if log_amp else 'lin'}"
        if key in self._cache_freq:
            return self._cache_freq[key]

        S = self._power_spectrum() if use_power else self._magnitude_spectrum()
        freqs = self.freqs
        K, T = S.shape

        x = freqs.astype(float)
        x_mean = np.mean(x)
        x_centered = x - x_mean
        denom = np.sum(x_centered**2) + EPS

        slopes = np.zeros(T, dtype=float)
        for t in range(T):
            y = S[:, t]
            if log_amp:
                y = np.log(y + EPS)

            y_mean = np.mean(y)
            y_centered = y - y_mean
            num = np.sum(x_centered*y_centered)
            slopes[t] = num/denom

        self._cache_freq[key] = slopes
        return slopes

    def _spectral_skewness(self, use_power=True):
        """
        Spectral skewness per frame (standardized third central moment)
        """
        key = f"spectral_skewness_{'pow' if use_power else 'mag'}"
        if key in self._cache_freq:
            return self._cache_freq[key]
        
        S = self._power_spectrum() if use_power else self._magnitude_spectrum()
        freqs = self.freqs[..., None]

        S_sum = np.sum(S, axis=0, keepdims=True) + EPS
        p = S/S_sum

        centroid = self._spectral_centroid(use_power=use_power)[None, :]

        diffs = freqs - centroid
        mu2 = np.sum((diffs**2)*p, axis=0) + EPS
        mu3 = np.sum((diffs**3)*p, axis=0)

        skew = mu3/(mu2**1.5)

        self._cache_freq[key] = skew
        return skew

    def _spectral_kurtosis(self, use_power=True, excess=True):
        """
        Spectral kurtosis per frame (standardized fourth central moment)
        If excess=True subtract 3 (Gaussian baseline)
        """
        key = f"spectral_kurtosis_{'pow' if use_power else 'mag'}_{'excess' if excess else 'raw'}"
        if key in self._cache_freq:
            return self._cache_freq[key]

        S = self._power_spectrum() if use_power else self._magnitude_spectrum()
        freqs = self.freqs[..., None]

        S_sum = np.sum(S, axis=0, keepdims=True) + EPS
        p = S/S_sum

        centroid = self._spectral_centroid(use_power=use_power)[None, :]

        diffs = freqs - centroid
        mu2 = np.sum((diffs**2)*p, axis=0) + EPS
        mu4 = np.sum((diffs**4)*p, axis=0)

        kurt = mu4/(mu2**2)
        if excess:
            kurt = kurt - 3.0

        self._cache_freq[key] = kurt
        return kurt

    # Harmonic/Timbre
    def _fundamental_freq_estimate(self, fmin=50.0, fmax=2000.0):
        """
        Very rough f0 estimate per frame using spectral peaks and spacing
        This is simplistic and mainly to support harmonic features.
        Returns array of shape (n_frames,) in Hz, with 0.0 when unknown.
        """
        key = f"f0_estimate_{fmin}_{fmax}"
        if key in self._cache_freq:
            return self._cache_freq[key]
        
        mag = self._magnitude_spectrum()
        freqs = self.freqs
        K, T = mag.shape

        f0 = np.zeros(T, dtype=float)

        for t in range(T):
            spec = mag[:, t]

            # Find prominent peaks
            peak_idx, _ = find_peaks(spec, height=np.max(spec)*0.2)
            if peak_idx.size < 2:
                f0[t] = 0.0
                continue

            peak_freqs = freqs[peak_idx]

            # Restrict to plausible f0 range
            peak_freqs = peak_freqs[(peak_freqs >= fmin) & (peak_freqs <= fmax)]
            if peak_freqs.size == 0:
                f0[t] = 0.0
                continue

            # Pick lowest prominent peak as crude f0
            f0[t] = float(np.min(peak_freqs))

        self._cache_freq[key] = f0
        return f0
    
    def _harmonic_bin_indices(self, f0, fmax=None):
        """
        Given f0 (Hz) per frame, return list of arrays of bin indices
        corresponding to harmonics for each frame.
        """
        mag = self._magnitude_spectrum()
        freqs = self.freqs
        K, T = mag.shape
        if fmax is None:
            fmax = freqs[-1]

        harmonic_bins_per_frame = []

        for t in range(T):
            f0_t = f0[t]
            if f0_t <= 0.0:
                harmonic_bins_per_frame.append(np.array([], dtype=int))
                continue

            h = 1
            bins = []
            while True:
                fh = h*f0_t
                if fh > fmax:
                    break
                freq_resolution = self.sr/self.N
                k = int(np.round(fh/freq_resolution))
                if 0 <= k < K:
                    bins.append(k)
                h += 1

            harmonic_bins_per_frame.append(np.array(bins, dtype=int))

        return harmonic_bins_per_frame

    def _harmonic_ratio(self):
        """
        Harmonic ratio per frame: harmonic energy / total energy
        """
        if "harmonic_ratio" in self._cache_freq:
            return self._cache_freq["harmonic_ratio"]
        
        mag = self._magnitude_spectrum()
        f0 = self._fundamental_freq_estimate()
        harmonic_bins_per_frame = self._harmonic_bin_indices(f0)

        K, T = mag.shape
        hr = np.zeros(T, dtype=float)

        for t in range(T):
            spec = mag[:, t]
            total = np.sum(spec) + EPS
            hb = harmonic_bins_per_frame[t]
            if hb.size == 0:
                hr[t] = 0.0
                continue
            harmonic_energy = np.sum(spec[hb])
            hr[t] = harmonic_energy/total

        self._cache_freq["harmonic_ratio"] = hr
        return hr

    def _inharmonicity(self):
        """
        Inharmonicity per frame: deviation of harmonic peaks from h*f0
        Returns a normalized scalar per frame
        """
        if "inharmonicity" in self._cache_freq:
            return self._cache_freq["inharmonicity"]
        
        mag = self._magnitude_spectrum()
        freqs = self.freqs
        f0 = self._fundamental_freq_estimate()
        K, T = mag.shape

        inh = np.zeros(T, dtype=float)

        for t in range(T):
            f0_t = f0[t]
            if f0_t <= 0.0:
                inh[t] = 0.0
                continue

            spec = mag[:, t]

            # Find peaks
            peak_idx, _ = find_peaks(spec, height=np.max(spec)*0.2)
            if peak_idx.size < 2:
                inh[t] = 0.0
                continue

            peak_freqs = freqs[peak_idx]
            H = peak_freqs.size

            num = 0.0
            den = 0.0

            for i, f_p in enumerate(peak_freqs, start=1):
                h = i
                ideal = h*f0_t
                num += (h**2)*(f_p - ideal)**2
                den += (h**2)*(f0_t**2)

            inh[t] = num/(den + EPS)

        self._cache_freq["inharmonicity"] = inh
        return inh

    def _spectral_peaks(self, height_factor=0.2, max_peaks=20):
        """
        Simple peak extraction per frame
        Returns:
            peak_freqs: list of 1D arrays (Hz)
            peak_mags: list of 1D arrays
        """
        key = f"spectral_peaks_{height_factor}_{max_peaks}"
        if key in self._cache_freq:
            return self._cache_freq[key]
        
        mag = self._magnitude_spectrum()
        freqs = self.freqs
        K, T = mag.shape

        peak_freqs = []
        peak_mags = []

        for t in range(T):
            spec = mag[:, t]
            if spec.max() <= 0.0:
                peak_freqs.append(np.array([], dtype=float))
                peak_mags.append(np.array([], dtype=float))

                continue

            thresh = spec.max()*float(height_factor)
            idx, props = find_peaks(spec, height=thresh)
            if idx.size > max_peaks:
                # Keep top max_peaks by height
                heights = props["peak_heights"]
                order = np.argsort(heights)[::-1][:max_peaks]
                idx = idx[order]

            peak_freqs.append(freqs[idx])
            peak_mags.append(spec[idx])

        self._cache_freq[key] = (peak_freqs, peak_mags)
        return (peak_freqs, peak_mags)
    
    def _hnr(self, f0_hz=None, max_harmonics=20, tol_hz=0.5, use_power=True):
        """
        HNR per frame 
        f0_hz: array-like (T,), fundamental frequency; if None, returns zeros
        """
        key = f"hnr_{'pow' if use_power else 'mag'}"
        if key in self._cache_freq:
            return self._cache_freq[key]
        
        S = self._power_spectrum() if use_power else self._magnitude_spectrum()
        freqs = self.freqs
        K, T = S.shape

        if f0_hz is None:
            return np.zeros(T, dtype=float)
        
        f0_hz = np.asarray(f0_hz, dtype=float)
        if f0_hz.shape[0] != T:
            raise ValueError("f0_hz must have length equal to number of frames")
        
        hnr_db = np.zeros(T, dtype=float)

        for t in range(T):
            f0 = f0_hz[t]
            spec = S[:, t]
            total = np.sum(spec)
            if f0 <= 0.0 or total <= 0.0:
                hnr_db[t] = 0.0
                continue

            harmonic_mask = np.zeros_like(spec, dtype=bool)
            for h in range(1, max_harmonics + 1):
                target = h*f0
                if target > freqs[-1] + tol_hz:
                    break
                mask = (freqs >= target - tol_hz) & (freqs <= target + tol_hz)
                harmonic_mask |= mask

            E_h = np.sum(spec[harmonic_mask])
            E_n = np.sum(spec[~harmonic_mask])

            if E_n <= 0.0:
                # All energy is harmonic (or almost)
                hnr_db[t] = 60.0 # arbitrary high cap
            else:
                hnr_db[t] = 10*np.log10(E_h/(E_n + EPS))

        if f0_hz is None:
            self._cache_freq[key] = hnr_db

        return hnr_db
    
    def _spectral_envelope_bands(self, bands):
        """
        Coarse spectral envelope via band energies

        bands: list of (low_hz, high_hz) pairs
        Returns:
            envelope: shape (len(bands), T) with energy per band per frame
        """
        return self._band_energy(bands)
    
    def _spectral_envelope_normalized(self, bands):
        """
        Relative band-energy envelope (per-frame normalized)
        """
        band_energy = self._spectral_envelope_bands(bands)
        total = np.sum(band_energy, axis=0, keepdims=True) + EPS

        return band_energy/total
    
    # Noise
    def _spectral_flatness(self, use_power=True):
        """
        Spectral flatness per frame
        High values -> noise-like; low values -> tonal/peaky
        """
        key = f"spectral_flatness_{'pow' if use_power else 'mag'}"
        if key in self._cache_freq:
            return self._cache_freq[key]
        
        S = self._power_spectrum() if use_power else self._magnitude_spectrum()
        S = np.maximum(S, EPS)

        gm = np.exp(np.mean(np.log(S), axis=0))
        am = np.mean(S, axis=0) + EPS
        flatness = gm/am
        flatness = np.clip(flatness, 0.0, 1.0)

        self._cache_freq[key] = flatness
        return flatness
    
    def _spectral_entropy(self, use_power=True, normalize=True):
        """
        Spectral entropy per frame
        High values -> more uniform spectrum (noise-like)
        """
        key = f"spectral_entropy_{'pow' if use_power else 'mag'}"
        if key in self._cache_freq:
            return self._cache_freq[key]
        
        S = self._power_spectrum() if use_power else self._magnitude_spectrum()
        S_sum = np.sum(S, axis=0, keepdims=True) + EPS
        p = S/S_sum
        p = np.maximum(p, EPS)

        H = -np.sum(p*np.log(p), axis=0)

        if normalize:
            H = H/np.log(S.shape[0] + EPS)

        H = np.clip(H, 0.0, 1.0)

        self._cache_freq[key] = H
        return H
    
    def _spectral_flux(self, use_power=True, normalize=True, half_wave_rectify=False):
        """
        Spectral flux per frame
        Measures frame-to-frame change in spectrum
        """
        key = f"spectral_flux_{'pow' if use_power else 'mag'}_{'norm' if normalize else 'raw'}_{'hwr' if half_wave_rectify else 'full'}"
        if key in self._cache_freq:
            return self._cache_freq[key]

        S = self._power_spectrum() if use_power else self._magnitude_spectrum()

        if normalize:
            norms = np.linalg.norm(S, axis=0, keepdims=True) + EPS
            S = S/norms

        diff = S[:, 1:] - S[:, :-1]

        if half_wave_rectify:
            diff = np.maximum(diff, 0.0)

        flux = np.sqrt(np.sum(diff**2, axis=0))
        flux = np.concatenate([[0.0], flux])

        self._cache_freq[key] = flux
        return flux

    def _band_ratios(self, bands, relative=True):
        """
        Band energies ratios per frame

        bands: list of (low_hz, high_hz) tuples

        If relative = True:
            returns each band's energy divided by total energy
        If relative = False:
            returns absolute band energies
        """
        key = f"band_ratios_{tuple(bands)}_{'rel' if relative else 'abs'}"
        if key in self._cache_freq:
            return self._cache_freq[key]

        band_energy = self._band_energy(bands)

        if relative:
            total = self._frame_energy()[None, :] + EPS
            ratios = band_energy/total
        else:
            ratios = band_energy

        self._cache_freq[key] = ratios
        return ratios
    
    def _low_high_band_ratio(self, low_band, high_band):
        """
        Ratio of low band energy to high band energy per frame
        """
        key = f"low_high_band_ratio_{low_band}_{high_band}"
        if key in self._cache_freq:
            return self._cache_freq[key]
        
        bands = [low_band, high_band]
        e = self._band_energy(bands)
        ratio = e[0]/(e[1] + EPS)

        self._cache_freq[key] = ratio
        return ratio
    
    # Rhythm/Transient
    def _pulse_clarity_ac(self, use_power=True, normalize=True, half_wave_rectify=True, min_lag=1, max_lag=None):
        """
        Pulse clarity via autocorrelation of onset envelope
        Returns a per-frame-like scalar summary over track
            clarity = best best nonzero autocorrelation peak/zero-lag autocorr
        """
        key = f"pulse_clarity_autocorr_{'pow' if use_power else 'mag'}_{'norm' if normalize else 'raw'}_{'hwr' if half_wave_rectify else 'full'}"
        if key in self._cache_freq:
            return self._cache_freq[key]
        
        flux = self._spectral_flux(
            use_power=use_power,
            normalize=normalize,
            half_wave_rectify=half_wave_rectify
        )

        if flux.size == 0 or np.all(flux <= 0):
            self._cache_freq[key] = 0.0
            return 0.0

        flux = flux - np.mean(flux)
        ac = np.correlate(flux, flux, mode="full")
        ac = ac[ac.size//2:]

        if ac.size < 2:
            self._cache_freq[key] = 0.0
            return 0.0
        
        zero_lag = ac[0] + EPS
        if max_lag is None:
            max_lag = ac.size - 1

        lo = max(1, int(min_lag))
        hi = min(int(max_lag), ac.size - 1)
        if hi <= lo:
            self._cache_freq[key] = 0.0
            return 0.0

        best_peak = np.max(ac[lo:hi + 1])
        clarity = best_peak/zero_lag
        clarity = safe_clip01(clarity)

        self._cache_freq[key] = clarity
        return clarity

    def _beat_periodicity(self, use_power=True, normalize=True, half_wave_rectify=True, min_lag=1, max_lag=None):
        """
        Beat periodicity from autocorrelation peak regularity of onset envelope
        Higher = more regular rhythmic repitition.
        """
        key = f"beat_periodicity_{'pow' if use_power else 'mag'}_{'norm' if normalize else 'raw'}_{'hwr' if half_wave_rectify else 'full'}"
        if key in self._cache_freq:
            return self._cache_freq[key]
        
        flux = self._spectral_flux(
            use_power=use_power,
            normalize=normalize,
            half_wave_rectify=half_wave_rectify
        )

        if flux.size < 3:
            self._cache_freq[key] = 0.0
            return 0.0
        
        flux = flux - np.mean(flux)
        ac = np.correlate(flux, flux, mode="full")
        ac = ac[ac.size//2:]

        if ac.size < 2:
            self._cache_freq[key] = 0.0
            return 0.0

        if max_lag is None:
            max_lag = ac.size - 1

        lo = max(1, int(min_lag))
        hi = min(int(max_lag), ac.size - 1)
        if hi <= lo:
            self._cache_freq[key] = 0.0
            return 0.0
        
        region = ac[lo:hi + 1]
        if region.size == 0 or np.all(region <= 0):
            self._cache_freq[key] = 0.0
            return 0.0

        peak = np.max(region)
        mean_region = np.mean(np.abs(region)) + EPS
        periodicity = peak/mean_region
        periodicity = float(np.clip(periodicity / (1.0 + periodicity), 0.0, 1.0))

        self._cache_freq[key] = periodicity
        return periodicity
    
    def _transient_counts(self, use_power=True, normalize=True, half_wave_rectify=True, threshold=None):
        """
        Count onset-envelope peaks above threshold
        If threshold is None, uses a robust adaptive threshold
        Returns:
            counts: integer count of transient-like peaks
        """
        key = f"transient_counts_{'pow' if use_power else 'mag'}_{'norm' if normalize else 'raw'}_{'hwr' if half_wave_rectify else 'full'}"
        if key in self._cache_freq:
            return self._cache_freq[key]
        
        flux = self._spectral_flux(
            use_power=use_power,
            normalize=normalize,
            half_wave_rectify=half_wave_rectify
        )

        if flux.size < 3:
            self._cache_freq[key] = 0
            return 0
        
        # Adaptive threshold
        if threshold is None:
            med = np.median(flux)
            mad = np.median(np.abs(flux - med)) + EPS
            threshold = med + 4*mad  # Stricter than before
        
        # Use scipy's find_peaks for robust detection
        from scipy.signal import find_peaks
        
        peaks, _ = find_peaks(
            flux,
            height=threshold,        # Above threshold
            prominence=threshold*0.2, # Must stand out
            distance=3               # At least 3 frames apart
        )

        count = int(len(peaks))
        self._cache_freq[key] = count
        return count
    
    def _transient_rate(self, use_power=True, normalize=True, half_wave_rectify=True, threshold=None):
        """
        Transient count normalized by track duration in seconds
        """
        key = f"transient_rate_{'pow' if use_power else 'mag'}_{'norm' if normalize else 'raw'}_{'hwr' if half_wave_rectify else 'full'}"
        if key in self._cache_freq:
            return self._cache_freq[key]

        count = self._transient_counts(
            use_power=use_power,
            normalize=normalize,
            half_wave_rectify=half_wave_rectify,
            threshold=threshold
        )

        n_frames = max(1, self._spectral_flux(
            use_power=use_power,
            normalize=normalize,
            half_wave_rectify=half_wave_rectify
        ).size)

        duration_sec = (n_frames*self.H)/float(self.sr)
        rate = count/max(duration_sec, EPS)

        self._cache_freq[key] = rate
        return rate
    
    def _percussive_spectral_slope(self, use_power=True, log_amp=True):
        """
        Average spectral slope of onset-rich frames
        Uses only frequency-domain quantities
        """
        key = f"percussive_spectral_slope_{'pow' if use_power else 'mag'}_{'log' if log_amp else 'lin'}"
        if key in self._cache_freq:
            return self._cache_freq[key]

        S = self._power_spectrum() if use_power else self._magnitude_spectrum()
        freqs = self.freqs
        K, T = S.shape

        flux = self._spectral_flux(
            use_power=use_power,
            normalize=True,
            half_wave_rectify=True
        )

        if flux.size != T:
            flux = np.resize(flux, T)

        # Select the transient-rich frames adaptibility
        thr = np.median(flux) + np.std(flux)
        idx_frames = np.where(flux > thr)[0]

        if idx_frames.size == 0:
            self._cache_freq[key] = 0.0
            return 0.0

        x = freqs
        x_mean = np.mean(x)
        x_centered = x - x_mean
        denom = np.sum(x_centered**2) + EPS

        slopes = []
        for t in idx_frames:
            y = S[:, t]
            if log_amp:
                y = np.log(y + EPS)
            y_mean = np.mean(y)
            y_centered = y - y_mean
            slope = np.sum(x_centered*y_centered)/denom
            slopes.append(slope)

        slope_avg = float(np.mean(slopes)) if slopes else 0.0

        self._cache_freq[key] = slope_avg
        return slope_avg
    
    # Phase
    def _phase(self):
        """
        Unwrapped STFT phase in radians, shape (K, T)
        """
        key = "phase"
        if key in self._cache_freq:
            return self._cache_freq[key]
        
        X = self.X
        phi = np.unwrap(np.angle(X), axis=0)

        self._cache_freq[key] = phi
        return phi
    
    def _group_delay(self):
        """
        Group delay approximation:
            tau_g(w) = -d(phi)/dw
        Return shape (K, T) array of group delay values in seconds
        """
        key = "group_delay"
        if key in self._cache_freq:
            return self._cache_freq[key]
        
        phi = self._phase()
        freqs = self.freqs.astype(float)

        if freqs.size < 3:
            gd = np.zeros_like(phi)
            self._cache_freq[key] = gd
            return gd

        dphi_df = np.gradient(phi, freqs, axis=0)
        gd = -dphi_df
        gd = np.nan_to_num(gd, nan=0.0, posinf=0.0, neginf=0.0)

        self._cache_freq[key] = gd
        return gd
    
    def _instantaneous_freq(self):
        """
        Instantaneous frequency estimate from phase increments across time
        Rreturns shape (K, T) array of instantaneous frequency in Hz
        """
        key = "instantaneous_frequency"
        if key in self._cache_freq:
            return self._cache_freq[key]

        X = self.X

        phi_1 = np.angle(X[:, 0:-1])/(2*np.pi)
        phi_2 = np.angle(X[:, 1:])/(2*np.pi)

        K = X.shape[0]
        ind_k = np.arange(0, K).reshape(-1, 1)

        # Bin offset
        delta_phi = phi_2 - phi_1 - ind_k*self.H/self.N
        delta_phi = np.mod(delta_phi + 0.5, 1.0) - 0.5
        kappa = (self.N/self.H)*delta_phi

        # Instantaneous frequency
        inst_freq = (ind_k + kappa)*self.sr/self.N
        inst_freq = np.nan_to_num(inst_freq, nan=0.0, posinf=0.0, neginf=0.0)
        inst_freq = np.hstack((np.copy(inst_freq[:, 0]).reshape(-1, 1), inst_freq))

        self._cache_freq[key] = inst_freq
        return inst_freq
    
    def _phase_congruency(self, use_power=True):
        """
        Phase congruency per frame
        Measures alignment of phase vectors across frequency bins
        Returns shape (T,) array of values in [0, 1], higher = more aligned/tonal
        """
        key = f"phase_congruency_{'pow' if use_power else 'mag'}"
        if key in self._cache_freq:
            return self._cache_freq[key]
        
        X = self._power_spectrum() if use_power else self._magnitude_spectrum()
        phi = self._phase()

        weights = np.maximum(X, EPS)
        vec = np.sum(weights*np.exp(1j*phi), axis=0)
        denom = np.sum(weights, axis=0) + EPS
        pc = np.abs(vec)/denom
        pc = np.clip(pc, 0.0, 1.0)

        self._cache_freq[key] = pc
        return pc
    
    def _phase_coherence_time(self):
        """
        Phase coherence over time for each frequency bin
        Measures consistency of phase increments across frames
        Returns shape (K,) array of values in [0, 1], higher = more coherent/tonal
        """
        key = "phase_coherence_time"
        if key in self._cache_freq:
            return self._cache_freq[key]
        
        phi = self._phase()
        K, T = phi.shape

        if T < 2:
            out = np.zeros(K)
            self._cache_freq[key] = out
            return out

        delta_phi = np.diff(phi, axis=1)
        coh = np.abs(np.mean(np.exp(1j*delta_phi), axis=1))
        coh = np.clip(coh, 0.0, 1.0)

        self._cache_freq[key] = coh
        return coh

    def _phase_coherence_channels(self, phi_a, phi_b):
        """
        Phase coherence between two phase sequences
        Useful if you have two channels or two phase trajectories

        phi_a, phi_b: shape (K, T) arrays of unwrapped phase
        Returns shape (T,) array of values in [0, 1], higher = more coherent
        """
        phi_a = np.asarray(phi_a, dtype=float)
        phi_b = np.asarray(phi_b, dtype=float)

        if phi_a.shape != phi_b.shape or phi_a.size == 0:
            return 0.0
        
        diff = phi_a - phi_b
        coh = np.abs(np.mean(np.exp(1j*diff)))
        coh = float(np.clip(coh, 0.0, 1.0))

        return coh
    
    # Sub-band Features
    def _default_sub_bands(self, n_bands=8, fmin=0.0, fmax=None):
        """
        Uniform frequency sub-bands
        Returns list of (low_hz, high_hz, center_hz) tuples
        """
        if fmax is None:
            fmax = self.sr/2.0

        edges = np.linspace(fmin, fmax, n_bands + 1)
        bands = []
        for i in range(n_bands):
            lo = float(edges[i])
            hi = float(edges[i + 1])
            center = (lo + hi)/2.0
            bands.append((lo, hi, center))

        return bands
    
    def _sub_band_energy(self, n_bands=8, use_power=True, fmin=0.0, fmax=None):
        """
        Returns absolute energy per sub-band per frame
        Shape: (n_bands, n_frames)
        """
        key = f"sub_band_energy_{n_bands}_{'pow' if use_power else 'mag'}_{fmin}_{fmax}"
        if key in self._cache_freq:
            return self._cache_freq[key]
        
        bands = self._default_sub_bands(n_bands=n_bands, fmin=fmin, fmax=fmax)
        S = self._power_spectrum() if use_power else self._magnitude_spectrum()
        freqs = self.freqs

        band_energy = np.zeros((n_bands, S.shape[1]), dtype=float)

        for b, (lo, hi, _) in enumerate(bands):
            if b == n_bands - 1:
                mask = (freqs >= lo) & (freqs <= hi)
            else:
                mask = (freqs >= lo) & (freqs < hi)

            if np.any(mask):
                band_energy[b] = np.sum(S[mask, :], axis=0)

        self._cache_freq[key] = band_energy
        return band_energy
    
    def _sub_band_energy_ratios(self, n_bands=8, use_power=True, fmin=0.0, fmax=None):
        """
        Relative energy in each sub-band
        Shape: (n_bands, n_frames)
        """
        key = f"sub_band_energy_ratios_{n_bands}_{'pow' if use_power else 'mag'}_{fmin}_{fmax}"
        if key in self._cache_freq:
            return self._cache_freq[key]

        band_energy = self._sub_band_energy(n_bands=n_bands, use_power=use_power, fmin=fmin, fmax=fmax)
        total = np.sum(band_energy, axis=0, keepdims=True) + EPS
        ratios = band_energy/total

        self._cache_freq[key] = ratios
        return ratios
    
    def _sub_band_entropy(self, n_bands=8, use_power=True, fmin=0.0, fmax=None, normalize=True):
        """
        Entropy over sub-band energy distribution per frame
        Shape: (n_frames,) with values in [0, 1], higher = more uniform distribution
        """
        key = f"sub_band_entropy_{n_bands}_{'pow' if use_power else 'mag'}_{fmin}_{fmax}_{'norm' if normalize else 'raw'}"
        if key in self._cache_freq:
            return self._cache_freq[key]

        ratios = self._sub_band_energy_ratios(n_bands=n_bands, use_power=use_power, fmin=fmin, fmax=fmax)
        p = np.maximum(ratios, EPS)
        H = -np.sum(p*np.log(p), axis=0)

        if normalize:
            H = H/np.log(n_bands + EPS)

        H = np.clip(H, 0.0, 1.0)

        self._cache_freq[key] = H
        return H
    
    def _sub_band_centroid(self, n_bands=8, use_power=True, fmin=0.0, fmax=None):
        """
        Centroid over sub-band energies per frame
        Shape: (n_frames,) in Hz
        """
        key = f"sub_band_centroid_{n_bands}_{'pow' if use_power else 'mag'}_{fmin}_{fmax}"
        if key in self._cache_freq:
            return self._cache_freq[key]
        
        bands = self._default_sub_bands(n_bands=n_bands, fmin=fmin, fmax=fmax)
        centers = np.array([c for _, _, c in bands], dtype=float)

        ratios = self._sub_band_energy_ratios(n_bands=n_bands, use_power=use_power, fmin=fmin, fmax=fmax)
        centroid = np.sum(centers[:, None]*ratios, axis=0)

        self._cache_freq[key] = centroid
        return centroid
    
    def _sub_band_flatness(self, n_bands=8, use_power=True, fmin=0.0, fmax=None):
        """
        Flatness of sub-band energies per frame
        Shape: (n_frames,) with values in [0, 1], higher = more uniform distribution
        """
        key = f"sub_band_flatness_{n_bands}_{'pow' if use_power else 'mag'}_{fmin}_{fmax}"
        if key in self._cache_freq:
            return self._cache_freq[key]

        band_energy = self._sub_band_energy(n_bands=n_bands, use_power=use_power, fmin=fmin, fmax=fmax)
        band_energy = np.maximum(band_energy, EPS)

        gm = np.exp(np.mean(np.log(band_energy), axis=0))
        am = np.mean(band_energy, axis=0) + EPS
        flatness = gm/am
        flatness = np.clip(flatness, 0.0, 1.0)

        self._cache_freq[key] = flatness
        return flatness

    def _sub_band_ratio(self, band_a, band_b, n_bands=8, use_power=True, fmin=0.0, fmax=None):
        """
        Ratio of one sub-band to another per frame
        band_a, band_b: indices of sub-bands to compare (0-based)
        Returns shape (n_frames,) array of ratios
        """
        key = f"sub_band_ratio_{band_a}_{band_b}_{n_bands}_{'pow' if use_power else 'mag'}_{fmin}_{fmax}"
        if key in self._cache_freq:
            return self._cache_freq[key]
        
        band_energy = self._sub_band_energy(n_bands=n_bands, use_power=use_power, fmin=fmin, fmax=fmax)

        if band_a < 0 or band_a >= n_bands or band_b < 0 or band_b >= n_bands:
            ratio = np.zeros(band_energy.shape[1], dtype=float)
        else:
            ratio = band_energy[band_a]/(band_energy[band_b] + EPS)

        self._cache_freq[key] = ratio
        return ratio
    
    def _sub_band_low_high_ratio(self, split_band=4, n_bands=8, use_power=True, fmin=0.0, fmax=None):
        """
        Low-band energy / high-band energy ratio per frame
        split_band: index of sub-band to split low vs high (0-based)
        Returns shape (n_frames,) array of ratios 
        """
        key = f"sub_band_low_high_ratio_{split_band}_{n_bands}_{'pow' if use_power else 'mag'}_{fmin}_{fmax}"
        if key in self._cache_freq:
            return self._cache_freq[key]

        band_energy = self._sub_band_energy(n_bands=n_bands, use_power=use_power, fmin=fmin, fmax=fmax)

        split_band = int(np.clip(split_band, 0, n_bands - 1))
        low = np.sum(band_energy[:split_band, :], axis=0)
        high = np.sum(band_energy[split_band:, :], axis=0)
        ratio = low/(high + EPS)

        self._cache_freq[key] = ratio
        return ratio