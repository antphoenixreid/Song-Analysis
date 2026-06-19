import os
import numpy as np
import librosa
from scipy.signal import find_peaks
from scipy.ndimage import maximum_filter
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

        key = "global_loudness_dB"
        if key in self._cache_time:
            return self._cache_time[key]
        
        rms = np.sqrt(np.mean(self.y**2)) + EPS
        if rms < 1e-10:
            self._cache_time[key] = -80.0
            return -80.0
        
        loud_db = 20.0*np.log10(rms)

        self._cache_time[key] = loud_db
        return loud_db

    # Amplitude/Loudness Features
    def _rms_envelope(self) -> np.ndarray:
        key = "rms_env"
        if key in self._cache_time:
            return self._cache_time[key]
        
        num_frames = 1 + int((len(self.y) - self.N) // self.H)
        rms_env = np.zeros(num_frames, dtype=float)

        for i in range(num_frames):
            start = i*self.H
            end = start + self.N
            frame = self.y[start:end]

            rms_env[i] = float(np.sqrt(np.mean(frame**2)))

        self._cache_time[key] = rms_env
        return rms_env

    def _short_time_energy(self) -> np.ndarray:
        key = "ste"
        if key in self._cache_time:
            return self._cache_time[key]

        rms_env = self._rms_envelope()
        ste = self.N*(rms_env**2)

        self._cache_time[key] = ste
        return ste
    
    def _peak_amplitude(self) -> np.ndarray:
        key = "peak_amp"
        if key in self._cache_time:
            return self._cache_time[key]
        
        num_frames = 1 + int((len(self.y) - self.N) // self.H)
        peak_amp = np.zeros(num_frames, dtype=float)

        for i in range(num_frames):
            start = i*self.H
            end = start + self.N
            frame = self.y[start:end]

            peak_amp[i] = float(np.max(np.abs(frame)))

        self._cache_time[key] = peak_amp
        return peak_amp
    
    def _active_rms_mask(self, db_threshold: float = -60.0) -> np.ndarray:
        key = f"active_mask_{db_threshold}"
        if key in self._cache_time:
            return self._cache_time[key]
        
        rms_env = self._rms_envelope()
        peak_amp = self._peak_amplitude()

        rms_db = 20*np.log10(rms_env + EPS)
        peak_db = 20*np.log10(peak_amp + EPS)

        mask = (rms_db > db_threshold) | (peak_db > db_threshold)

        self._cache_time[key] = mask
        return mask
    
    def _crest_factor(self, db_threshold: float = -60.0) -> np.ndarray:
        key = f"crest_factor_{db_threshold}"
        if key in self._cache_time:
            return self._cache_time[key]
        
        rms_env = self._rms_envelope()
        peak_amp = self._peak_amplitude()
        mask = self._active_rms_mask(db_threshold=db_threshold)

        crest = np.zeros_like(rms_env)
        crest[mask] = peak_amp[mask]/(rms_env[mask] + EPS)

        self._cache_time[key] = crest
        return crest
    
    def _dynamic_range(self) -> float:
        key = "dynamic_range"
        if key in self._cache_time:
            return self._cache_time[key]
        
        rms_env = self._rms_envelope()

        rms_db = 20.0*np.log10(rms_env + EPS)
        rms_db = np.clip(rms_db, -80.0, 0.0)
        dr = np.percentile(rms_db, 90) - np.percentile(rms_db, 10)

        self._cache_time[key] = dr
        return dr
    
    def _onset_envelope(self, 
                    aggregate=np.median,
                    n_mels=128,
                    lag=1,
                    max_size=1,
                    detrend=False,
                    center=True):
        """Cached onset strength envelope"""
        key = f"onset_env_strength_{aggregate}_{n_mels}_{lag}_{max_size}_{detrend}_{center}"
        if key in self._cache_time:
            return self._cache_time[key]
        
        onset_env = librosa.onset.onset_strength(
            y=self.y,
            sr=self.sr,
            hop_length=self.H,
            n_fft=self.N,
            aggregate=aggregate,
            n_mels=n_mels,
            lag=lag,
            max_size=max_size,
            detrend=detrend,
            center=center
        )

        onset_env = np.asarray(onset_env, dtype=float).ravel()

        frame_rate = self.sr/float(self.H)
        times = np.arange(len(onset_env))/frame_rate

        result = {
            'onset_env': onset_env,
            'times': times,
            'frame_rate': frame_rate
        }

        self._cache_time[key] = result
        return result

    def _onset_frames(self) -> np.ndarray:
        """Cached onset detection frames"""
        key = "onset_frames"
        if key in self._cache_time:
            return self._cache_time[key]
        
        env = self._onset_envelope()['onset_env']
        frames = librosa.onset.onset_detect(
            onset_envelope=env,
            sr=self.sr, hop_length=self.H,
            backtrack=False, units='frames'
        )
        self._cache_time[key] = frames
        return frames
    
    def _attack_time(self) -> float:
        key = "attack_time"
        if key in self._cache_time:
            return self._cache_time[key]

        rms_env = self._rms_envelope()
        onsets = self._onset_frames()
        
        if len(onsets) == 0:
            self._cache_time[key] = 0.0
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
            self._cache_time[key] = 0.0
            return 0.0
        
        avg_attack_time = float(np.median(attack_times))

        self._cache_time[key] = avg_attack_time
        return avg_attack_time
    
    def _attack_slope(self) -> float:
        """Average attack slope across all onsets (dB/second)"""
        key = "attack_slope"
        if key in self._cache_time:
            return self._cache_time[key]
        
        rms_env = self._rms_envelope()
        rms_dB = 20.0 * np.log10(rms_env + EPS)
        
        onsets = self._onset_frames()
        
        if len(onsets) == 0:
            self._cache_time[key] = 0.0
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
            self._cache_time[key] = 0.0
            return 0.0
        
        avg_slope = float(np.median(attack_slopes))
        self._cache_time[key] = avg_slope
        return avg_slope
    
    def _decay_slope(self) -> float:
        """Average decay slope across all onsets (dB/second)"""
        key = "decay_slope"
        if key in self._cache_time:
            return self._cache_time[key]
        
        rms_env = self._rms_envelope()
        rms_dB = 20.0 * np.log10(rms_env + EPS)
        
        onsets = self._onset_frames()
        
        if len(onsets) == 0:
            self._cache_time[key] = 0.0
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
            self._cache_time[key] = 0.0
            return 0.0
        
        avg_slope = float(np.median(decay_slopes))
        self._cache_time[key] = avg_slope
        return avg_slope
    
    def _energy_variance(self) -> float:
        key = "energy_variance"
        if key in self._cache_time:
            return self._cache_time[key]
        
        rms_env = self._rms_envelope()
        mask = self._active_rms_mask(db_threshold=-60.0)
        active = rms_env[mask]

        if active.size < 2:
            self._cache_time[key] = 0.0
            return 0.0
        
        q75, q25 = np.percentile(active, [75 ,25])
        iqr = q75 - q25

        var = float(iqr/(np.median(active) + EPS))

        self._cache_time[key] = var
        return var
    
    def _energy_modulation_rate(self, db_threshold: float = -60.0) -> float:
        key = f"energy_mod_rate_{db_threshold}"
        if key in self._cache_time:
            return self._cache_time[key]
        
        rms_env = self._rms_envelope()

        if rms_env.size < 2:
            self._cache_time[key] = 0.0
            return 0.0
        
        mask = self._active_rms_mask(db_threshold=db_threshold)
        active = rms_env[mask]
        silence_ratio = 1.0 - (mask.sum()/len(mask))

        if active.size < 2:
            self._cache_time[key] = 0.0
            return 0.0
        
        med = np.median(active)

        if silence_ratio > 0.1:
            q75, q25 = np.percentile(rms_env, [75, 25])
        else:
            q75, q25 = np.percentile(active, [75, 25])

        iqr = q75 - q25

        if iqr < max(1e-4, 0.01*med):
            self._cache_time[key] = 0.0
            return 0.0
        
        rms_n = np.clip((rms_env - q25) / iqr, -1.0, 3.0)
        mod_sig = np.abs(np.diff(rms_n))
        mod_rate = float(np.var(mod_sig))

        self._cache_time[key] = mod_rate
        return mod_rate
    
    # Noise/Speechiness Features
    def _zero_crossing_rate(self) -> np.ndarray:
        key = "zcr"
        if key in self._cache_time:
            return self._cache_time[key]
        
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

        self._cache_time[key] = zcr
        return zcr
    
    def _zcr_variance(self) -> float:
        key = "zcr_variance"
        if key in self._cache_time:
            return self._cache_time[key]
        
        zcr = self._zero_crossing_rate()
        mask = self._active_rms_mask(db_threshold=-60.0)
        active_zcr = zcr[mask]

        if active_zcr.size < 2:
            self._cache_time[key] = 0.0
            return 0.0
        
        q75, q25 = np.percentile(active_zcr, [75 ,25])
        iqr = q75 - q25

        var = float(iqr/(np.median(active_zcr) + EPS))

        self._cache_time[key] = var
        return var
    
    def _voiced_ratio(self, db_threshold: float = -60.0) -> float:
        key = f"voiced_ratio_{db_threshold}"
        if key in self._cache_time:
            return self._cache_time[key]
        
        zcr = self._zero_crossing_rate()
        mask = self._active_rms_mask(db_threshold=db_threshold)

        if mask.sum() < 2:
            self._cache_time[key] = 0.0
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

        self._cache_time[key] = ratio
        return ratio
    
    def _unvoiced_ratio(self) -> float:
        return 1.0 - self._voiced_ratio()
    
    def _transient_rate(self) -> float:
        key = "transient_rate"
        if key in self._cache_time:
            return self._cache_time[key]
        
        ste = self._short_time_energy()

        # Adaptive threshold (median + factor*MAD)
        median = np.median(ste)
        mad = np.median(np.abs(ste - median))
        threshold = median + 2.0*mad

        peaks, _ = find_peaks(ste, height=threshold)

        duration = len(self.y)/float(self.sr)
        rate = float(len(peaks))/duration

        self._cache_time[key] = rate
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
        key = "onset_times"
        if key in self._cache_time:
            return self._cache_time[key]
        
        onset_env = self._onset_envelope()['onset_env']

        if onset_env.size == 0:
            self._cache_time[key] = np.array([], dtype=float)
            return self._cache_time[key]
        
        # Adaptive threshold: median + k*MAD
        med = np.median(onset_env)
        mad = np.median(np.abs(onset_env - med))
        thr = med + 2.0*mad

        # Find peaks above threshold
        peaks, _ = find_peaks(onset_env, height=thr)

        # Convert frame indices to time (seconds)
        times = (peaks*self.H)/float(self.sr)

        self._cache_time[key] = times.astype(float)
        return times.astype(float)
    
    def _onset_rate(self) -> float:
        """
        Average number of onsets per second
        """
        key = "onset_rate"
        if key in self._cache_time:
            return self._cache_time["onset_rate"]
        
        onset_times = self._onset_times()
        if onset_times.size == 0:
            self._cache_time[key] = 0.0
            return 0.0

        duration = len(self.y)/float(self.sr)
        if duration <= 0:
            self._cache_time[key] = 0.0
            return 0.0
        
        rate = float(onset_times.size)/duration
        self._cache_time[key] = rate
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

    def _onset_autocorrelation(self, max_lag=None, normalize=True) -> np.ndarray:
        """
        Normalized autocorrelation of onset envelope
        """
        key = f"ac_onset_{max_lag}_{normalize}"
        if key in self._cache_time:
            return self._cache_time[key]

        onset = self._onset_envelope()
        onset_env = onset['onset_env']
        if onset_env.size < 2:
            ac = np.array([1.0, 0.0], dtype=float)
            self._cache_time[key] = ac
            return ac

        # Determine max lag
        if max_lag is None:
            # Default: up to ~10 seconds worth of lags
            # At 43 Hx frame rate, this is 430 frames
            frame_rate = onset['frame_rate']
            max_lag = min(len(onset_env) - 1, int(frame_rate*10.0))
        else:
            max_lag = min(max_lag, len(onset_env) - 1)

        env_centered = onset_env - np.mean(onset_env)

        # Compute autocorrelation 
        ac = np.correlate(env_centered, env_centered, mode='full')

        center = len(ac)//2
        ac = ac[center:center + max_lag + 1]

        # Normalize
        if normalize and ac[0] > 0:
            ac /= ac[0]
        else:
            ac = np.ones(1, dtype=float)

        self._cache_time[key] = ac.astype(float)
        return ac.astype(float)
    
    def _parabolic_interpolation(self, f: np.ndarray, x: int) -> float:
        """
        Sub-frame peak estimation to improve BPM precision.
        """
        if x <= 0 or x >= len(f) - 1:
            return float(x)
        
        a, b, c = f[x - 1], f[x], f[x + 1]
        denom = a - 2*b + c

        if abs(denom) < 1e-6:
            return float(x)

        return float(x - 0.5*(a - c)/denom)
    
    def _tempo_from_onset_ac(self,
                            bpm_min=40.0,
                            bpm_max=240.0) -> float:
        """
        Estimate global tempo (BPM) from onset autocorrelation
        
        Strategy:
        1. Compute autocorrelation of onset envelope
        2. Find peaks in the valid BPM range
        3. Check octave relationships (2x, 3x, 1/2, 1/3, etc.)
        4. Select best candidate using strength + preference weighting
        
        Parameters
        ----------
        bpm_min : float
            Minimum tempo to consider (default: 40 BPM)
        bpm_max : float
            Maximum tempo to consider (default: 240 BPM)
        
        Returns
        -------
        float
            Estimated tempo in BPM
        """
        key = f"tempo_from_onset_ac_{bpm_min}_{bpm_max}"
        if key in self._cache_time:
            return self._cache_time[key]
        
        # Get autocorrelation
        ac = self._onset_autocorrelation()
        onset = self._onset_envelope()
        frame_rate = onset.get('frame_rate', self.sr/float(self.H))

        if ac.size < 2:
            self._cache_time[key] = 0.0
            return 0.0

        # Convert lags to BPM
        lags = np.arange(len(ac))
        bpm = np.zeros_like(lags,dtype=float)
        bpm[1:] = 60.0*frame_rate/lags[1:]

        # Filter to valid range
        mask = (bpm >= bpm_min) & (bpm <= bpm_max) & np.isfinite(bpm)
        if not np.any(mask):
            self._cache_time[key] = 0.0
            return 0.0
        
        # Find strongest peak in valid range
        valid_ac = ac[mask]
        valid_bpm = bpm[mask]
        valid_lags = lags[mask]

        peak_idx = np.argmax(valid_ac)
        detected_bpm = valid_bpm[peak_idx]
        detected_lag = valid_lags[peak_idx]
        detected_strength = valid_ac[peak_idx]

        # ===================================================================
        # OCTAVE CORRECTION WITH AGGRESSIVE WEIGHTING
        # ===================================================================
        candidates = []

        # Add initial detection
        candidates.append({
            'bpm': detected_bpm,
            'lag': detected_lag,
            'strength': detected_strength,
            'label': 'initial'
        })

        # Helper: find peak near a target BPM with wider search window
        def find_peak_near(target_bpm, label, search_radius=3):
            """
            Find actual peak near target BPM
            """
            if not (bpm_min <= target_bpm <= bpm_max):
                return
            
            target_lag = 60.0*frame_rate/target_bpm
            lag_start = max(1, int(np.round(target_lag)) - search_radius)
            lag_end = min(len(ac), int(np.round(target_lag)) + search_radius + 1)

            if lag_start >= lag_end:
                return

            # Find maximum in search window
            search_window = ac[lag_start:lag_end]
            local_max_idx = np.argmax(search_window)
            actual_lag = lag_start + local_max_idx
            actual_strength = ac[actual_lag]
            actual_bpm = 60*frame_rate/actual_lag

            candidates.append({
                'bpm': actual_bpm,
                'lag': actual_lag,
                'strength': actual_strength,
                'label': label 
            })

        # Check ALL octave relationships
        find_peak_near(detected_bpm * 2.0, '2x')
        find_peak_near(detected_bpm * 3.0, '3x')
        find_peak_near(detected_bpm * 4.0, '4x')
        find_peak_near(detected_bpm / 2.0, '1/2')
        find_peak_near(detected_bpm / 3.0, '1/3')
        find_peak_near(detected_bpm / 4.0, '1/4')
        find_peak_near(detected_bpm * 3.0 / 2.0, '3/2')
        find_peak_near(detected_bpm * 2.0 / 3.0, '2/3')

        # ===================================================================
        # SELECTION: MUCH STRONGER PREFERENCE FOR TYPICAL RANGE
        # ===================================================================
        max_strength = max(c['strength'] for c in candidates)

        # Only consider candidates >= 50% of max strength
        threshold = max_strength*0.5
        strong_candidates = [c for c in candidates if c['strength'] >= threshold]

        if not strong_candidates:
            strong_candidates = candidates

        best_score = -1
        best_bpm = detected_bpm

        for cand in strong_candidates:
            score = cand['strength']

            # ===============================================================
            # VERY AGGRESSIVE PREFERENCE FOR TYPICAL RANGE
            # ===============================================================
            if 80 <= cand['bpm'] <= 160:
                score *= 2.5 # 150% boost
            elif 60 <= cand['bpm'] <= 180:
                score *= 1.3 # 30% boost

            # Heavy penalty for extremes
            if cand['strength'] < max_strength*0.95:
                if cand['bpm'] < 60:
                    score *= 0.6 # Stronger penalty
                if cand['bpm'] > 200:
                    score *= 0.6 # Stronger penalty

            if score > best_score:
                best_score = score
                best_bpm = cand['bpm']

        self._cache_time[key] = float(best_bpm)
        return float(best_bpm)
    
    def _pulse_clarity_ac(self) -> float:
        """
        Pulse clarity from dominance of main AC peak over runner-up,
        gated by absolute peak strength to suppress noise floor.
        """
        key = "pulse_clarity_ac"
        if key in self._cache_time:
            return self._cache_time[key]

        ac = self._onset_autocorrelation()
        if ac.size < 3:
            self._cache_time[key] = 0.0
            return 0.0

        ac_pos = ac[1:]
        if ac_pos.size == 0:
            self._cache_time[key] = 0.0
            return 0.0

        peaks, _ = find_peaks(ac_pos, prominence=0.01)

        if len(peaks) == 0:
            self._cache_time[key] = 0.0
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
        self._cache_time[key] = clarity
        return clarity
    
    def _windowed_tempo_series(self,
                               window_sec: float = 8.0,
                               hop_sec: float = 4.0) -> np.ndarray:
        """ 
        Estimate tempo per window from onset envelope autocorrelation
        """
        onset_env = self._onset_envelope()['onset_env']
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
        key = f"higuchi_fd_{k_max}"
        if key in self._cache_time:
            return self._cache_time[key]
        
        x = self.y.astype(float)
        n = len(x)  # ← Use signal length
        if n < 2 or k_max < 2:
            self._cache_time[key] = 1.0  # ← Return 1.0, not 0.0 (smooth signal)
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
            self._cache_time[key] = 1.0  # ← Default to smooth
            return 1.0
        
        # Guard against zero/negative Lk (happens for constant signals)
        if np.any(Lk <= 0):
            self._cache_time[key] = 1.0
            return 1.0
        
        ln_Lk = np.log(Lk + EPS)
        
        # Linear fit
        A = np.vstack([ln_k, np.ones_like(ln_k)]).T
        b, a = np.linalg.lstsq(A, ln_Lk, rcond=None)[0]
        fd = -float(b)
        
        # Clamp to valid range [1, 2]
        fd = float(np.clip(fd, 1.0, 2.0))
        
        self._cache_time[key] = fd
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
    
    # Spotify-based features (Time)
    def _spotify_loudness(self, active_only: bool = True) -> float:
        key = f"spotify_loudness_{active_only}"
        if key in self._cache_time:
            return self._cache_time[key]
        
        if getattr(self, "invalid", False):
            self._cache_time[key] = -80.0
            return -80.0

        if active_only:
            rms = self._rms_envelope()
            mask = self._active_rms_mask(db_threshold=-60.0)
            if mask.size == 0 or not np.any(mask):
                val = -80.0
            else:
                rms_active = np.sqrt(np.mean(rms[mask]**2)) + EPS
                val = 20.0*np.log10(rms_active)
        else:
            rms = np.sqrt(np.mean(self.y**2)) + EPS
            val = 20.0*np.log10(rms)

        val = float(max(val, -80.0))

        self._cache_time[key] = val
        return val
    
    def _spotify_energy(self, active_only: bool  = True) -> float:
        key = f"spotify_energy_{active_only}"
        if key in self._cache_time:
            return self._cache_time[key]
        
        if getattr(self, "invalid", False):
            self._cache_time[key] = 0.0
            return 0.0
        
        ste = self._short_time_energy()
        if ste.size == 0:
            self._cache_time[key] = 0.0
            return 0.0
        
        if active_only:
            mask = self._active_rms_mask(db_threshold=-60.0)
            if mask.size == 0 or not np.any(mask):
                avg_ste = 0.0
            else:
                avg_ste = float(np.mean(ste[mask]))
        else:
            avg_ste = float(np.mean(ste))

        # Normalize intensity to a roughly 0-1 scale using sigmoid-like squash
        intensity_score = np.tanh(avg_ste/(np.median(ste) + EPS))

        # 2. Rhythmic Activity Component (Onset Rate & Pulse Clarity)
        # High energy tracks are usually "busier" with more transients.
        onset_rate = self._onset_rate()
        # Normalize onset rate (e.g., 0 to 12 onsets/sec mapped to 0-1)
        activity_score = np.clip(onset_rate / 10.0, 0.0, 1.0)
        
        # 3. Complexity & Noise Component (ZCR & Hjorth Complexity)
        # Distorted or high-frequency heavy signals (high energy) have higher ZCR.
        zcr = np.mean(self._zero_crossing_rate())
        zcr_score = np.clip(zcr * 5.0, 0.0, 1.0) 

        # 4. Temporal Dynamics (Attack Slopes & Peak Amplitude)
        # "Punchy" music has steeper attack slopes and higher crest factors.
        crest_factor_values = self._crest_factor()
        if np.ndim(crest_factor_values) > 0:
            active_mask = self._active_rms_mask(db_threshold=-60.0)
            active_crest = crest_factor_values[active_mask] if active_mask.size else crest_factor_values
            crest_factor = float(np.mean(active_crest)) if active_crest.size else 0.0
        else:
            crest_factor = float(crest_factor_values)

        dynamic_score = np.clip((crest_factor - 1.0) / 10.0, 0.0, 1.0)

        # ===================================================================
        # WEIGHTED FUSION
        # ===================================================================
        # Weights prioritize Intensity and Activity as the primary drivers.
        weights = {
            'intensity': 0.40,
            'activity':  0.30,
            'zcr':       0.15,
            'dynamics':  0.15
        }

        energy_val = (
            (intensity_score * weights['intensity']) +
            (activity_score  * weights['activity']) +
            (zcr_score       * weights['zcr']) +
            (dynamic_score   * weights['dynamics'])
        )

        final_energy = float(np.clip(energy_val, 0.0, 1.0))

        self._cache_time[key] = final_energy
        return final_energy
    
    def _spotify_speechiness(self) -> float:
        """
        Estimate speechiness by analyzing Zero-Crossing Rate (ZCR) stability,
        spectral complexity, and the ratio of voiced to unvoiced segments.
        """
        key = "spotify_speechiness"
        if key in self._cache_time:
            return self._cache_time[key]
        
        if getattr(self, "invalid", False):
            self._cache_time[key] = 0.0
            return 0.0
        
        # 1. ZCR Variance (Speech Indicator)
        # Speech has highly varible ZCR compared to the consistent cycles of music
        zcr = self._zero_crossing_rate()
        zcr_var = np.var(zcr) if zcr.size > 0 else 0.0
        # Normalize: Speech usually has higher variance than stable musical tones.
        zcr_score = safe_clip01(zcr_var*100.0)

        # 2. Vocal/Unvoiced Ratio
        # This directly targets the phonetic components of speech
        vocal_ratio = self._unvoiced_ratio()
        v_u_score = safe_clip01(vocal_ratio)

        # 3. Spectral Entropy/Complexity
        # Speech is often more "complex" (less predictable) than harmonic music.
        # We can use Hjorth Complexity or Fractal Dimension as a proxy
        complexity = self._hjorth_parameters()["complexity"]
        complexity_score = safe_clip01((complexity - 1.0)/5.0)

        # 4. Rhythmic Stability Penalty
        # Music is periodic; speech is not. High beat periodicity lowers speechiness.
        periodicity = self._beat_periodicity_entropy()
        rhythm_penalty = 1.0 - safe_clip01(periodicity)

        # ===================================================================
        # WEIGHTED FUSION
        # ===================================================================
        # Vocal components and ZCR variance are the strongest indicators.
        weights = {
            'vocal': 0.45,
            'zcr_var': 0.25,
            'complexity': 0.15,
            'non_rhythmic': 0.15
        }

        speech_val = (
            (v_u_score*weights['vocal']) + 
            (zcr_score*weights['zcr_var']) + 
            (complexity_score*weights['complexity']) +
            (rhythm_penalty*weights['non_rhythmic'])
        )

        # Spotify Thresholds:
        # > 0.66: Entirely spoken
        # 0.33 - 0.66: Mix of speech and music (Rap)
        # < 0.33: Mostly music
        val = safe_clip01(speech_val)

        self._cache_time[key] = val
        return val

    def _spotify_acousticness(self) -> float:
        key = "spotify_acousticness"
        if key in self._cache_time:
            return self._cache_time[key]
        
        if getattr(self, "invalid", False):
            self._cache_time[key] = 0.0
            return 0.0
        
        rms = self._rms_envelope()
        if rms.size < 2:
            self._cache_time[key] = 0.0
            return 0.0
        
        diffs = np.abs(np.diff(rms))
        smoothness = 1.0 - np.mean(diffs)/(np.mean(rms) + EPS)

        attack_time = self._attack_time()
        attack_slope = self._attack_slope()
        decay_slope = self._decay_slope()
        silence = self._silence_ratio(db_threshold=-60.0)

        attack_score = safe_clip01(attack_time/(attack_time + 0.05))
        attack_slope_score = safe_clip01(1.0/(1.0 + abs(attack_slope)/50.0))
        decay_score = safe_clip01(1.0/(1.0 + abs(decay_slope)/50.0))
        smooth_score = safe_clip01(smoothness)
        silence_score = safe_clip01(silence)

        w_at = 0.20
        w_as = 0.20
        w_d = 0.20
        w_sm = 0.30
        w_ss = 0.10

        val = w_sm*smooth_score + w_at*attack_score + w_as*attack_slope_score + w_d*decay_score + w_ss*silence_score
        val = safe_clip01(val)

        self._cache_time[key] = val
        return val

    def _spotify_danceability(self) -> float:
        key = "spotify_danceability"
        if key in self._cache_time:
            return self._cache_time[key]
        
        if getattr(self, "invalid", False):
            self._cache_time[key] = 0.0
            return 0.0
        
        pulse = self._pulse_clarity_ac()
        stability = self._rhythmic_stability()
        periodicity = self._beat_periodicity_entropy()
        onset_rate = self._onset_rate()
        tempo = self._tempo_from_onset_ac()

        tempo_score = safe_clip01(1.0 - abs(tempo - 120.0)/120.0)
        onset_score = safe_clip01(onset_rate/(onset_rate + 5.0))
        pulse_score = safe_clip01(pulse)
        periodicity_score = safe_clip01(periodicity)
        stability_score = safe_clip01(0.5*stability.get("stability_exp", 0.0) + 0.5*stability.get("stability_cv", 0.0))

        w_temp = 0.15
        w_onset = 0.10
        w_pulse = 0.30
        w_per = 0.20
        w_stab = 0.25

        val = w_pulse*pulse_score + w_stab*stability_score + w_per*periodicity_score + w_temp*tempo_score + w_onset*onset_score
        val = safe_clip01(val)

        self._cache_time[key] = val
        return val
    
    def _spotify_tempo(self) -> float:
        key = "spotify_tempo"
        if key in self._cache_time:
            return self._cache_time[key]

        if getattr(self, "invalid", False):
            self._cache_time[key] = 0.0
            return 0.0

        # Get primary AC estimate
        tempo = self._tempo_from_onset_ac()
        
        # Sanity check: Use Onset Rate (transients per second)
        # 90 BPM is 1.5 beats/sec. 180 BPM is 3 beats/sec.
        # If onsets/sec is very high, but tempo is low, double it.
        onsets_per_sec = self._onset_rate()
        
        if tempo < 90.0 and onsets_per_sec > 3.5:
            # Likely an octave error (e.g., your 90 BPM vs 143 BPM error)
            tempo *= 2.0
        elif tempo > 160.0 and onsets_per_sec < 2.0:
            # Likely a double-time error
            tempo /= 2.0

        final_val = float(np.clip(tempo, 40.0, 240.0))
        self._cache_time[key] = final_val
        return final_val

    def _spotify_liveness(self) -> float:
        key = "spotify_liveness"
        if key in self._cache_time:
            return self._cache_time[key]

        if getattr(self, "invalid", False):
            self._cache_time[key] = 0.0
            return 0.0
        
        trans_rate = self._transient_rate()
        attack_time = self._attack_time()
        attack_slope = self._attack_slope()
        decay_slope = self._decay_slope()
        tempo_var = self._rhythmic_stability().get("tempo_var", 0.0)
        zcr_var = self._zcr_variance()

        trans_score = safe_clip01(trans_rate/(trans_rate + 3.0))
        attack_score = safe_clip01(1.0 - attack_time/(attack_time + 0.15))
        attack_slope_score = safe_clip01(abs(attack_slope)/(abs(attack_slope) + 20.0))
        decay_score = safe_clip01(abs(decay_slope)/(abs(decay_slope) + 20.0))
        tempo_var_score = safe_clip01(tempo_var/(tempo_var + 10.0))
        zcr_score = safe_clip01(zcr_var/(1.0 + zcr_var))

        w_t = 0.25
        w_attk = 0.20
        w_atks = 0.20
        w_d = 0.15
        w_tv = 0.10
        w_zcr = 0.10

        val = w_t*trans_score + w_attk*attack_score + w_atks*attack_slope_score + w_d*decay_score + w_tv*tempo_var_score + w_zcr*zcr_score
        val = safe_clip01(val)

        self._cache_time[key] = val
        return val

    def _spotify_time_signature(self) -> int:
        key = "spotify_time_signature"
        if key in self._cache_time:
            return self._cache_time[key]

        if getattr(self, "invalid", False):
            self._cache_time[key] = 4
            return 4
        
        periodicity = self._beat_periodicity_entropy()
        ac = self._onset_autocorrelation()
        if ac.size < 3:
            self._cache_time[key] = 4
            return 4
        
        fs_env = self.sr/float(self.H)
        candidate_meters = [3, 4]
        scores = {}

        for meter in candidate_meters:
            if meter == 3:
                lag_targets = [int(round(fs_env*0.5)), int(round(fs_env*1.0)), int(round(fs_env*1.5))]
            else:
                lag_targets = [int(round(fs_env*0.5)), int(round(fs_env*1.0)), int(round(fs_env*2.0))]

            vals = []
            for lag in lag_targets:
                if 1 <= lag < ac.size:
                    vals.append(ac[lag])
            scores[meter] = float(float(np.mean(vals)) if vals else 0.0)

        meter = max(scores, key=scores.get)
        if periodicity < 0.15:
            meter = 4

        self._cache_time[key] = meter
        return meter
    
    def spotify_audio_features(self, weights=None) -> dict:
        loudness = self._spotify_loudness(active_only=True)
        energy = self._spotify_energy(active_only=True)
        speechiness = self._spotify_speechiness()
        acousticness = self._spotify_acousticness()
        danceability = self._spotify_danceability()
        tempo = self._spotify_tempo()
        liveness = self._spotify_liveness()
        time_signature = self._spotify_time_signature()

        loudness_score = safe_clip01((loudness + 60.0)/60.0)
        energy_score = safe_clip01(energy/(energy + 0.01))
        tempo_score = safe_clip01(tempo/200.0)

        vals = np.array([
            loudness_score,
            energy_score,
            speechiness,
            acousticness,
            danceability,
            tempo_score,
            liveness,
            1.0 if time_signature == 4 else 0.5 if time_signature == 3 else 0.0
        ], dtype=float)

        if weights is None:
            weights = np.array([0.15, 0.15, 0.12, 0.12, 0.18, 0.14, 0.08, 0.06], dtype=float)

        fused = float(np.sum(vals*weights)/(np.sum(weights) + EPS))

        return {
            "loudness_db": loudness,
            "energy": energy,
            "speechiness": speechiness,
            "acousticness": acousticness,
            "danceability": danceability,
            "tempo_bpm": tempo,
            "liveness": liveness,
            "time_signature": time_signature,
            "spotify_fused": safe_clip01(fused)
        }

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

    def _band_energy(self, bands, use_power=True):
        """
        bands: list of (low_hz, high_hz) pairs
        Returns:
            band_energy: shape (n_bands, n_frames)
        """
        key = f"band_energy_{tuple(bands)}_{use_power}"
        if key in self._cache_freq:
            return self._cache_freq[key]

        S = self._power_spectrum() if use_power else self._magnitude_spectrum()
        out = np.zeros((len(bands), S.shape[1]), dtype=float)

        for i, (f_lo, f_hi) in enumerate(bands):
            mask = self._safe_band_mask(f_lo, f_hi)
            if np.any(mask):
                out[i] = np.sum(S[mask, :], axis=0)

        self._cache_freq[key] = out
        return out
    
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

    def _band_ratios(self, bands, use_power=True, relative=True):
        key = f"band_ratios_{tuple(bands)}_{use_power}_{relative}"
        if key in self._cache_freq:
            return self._cache_freq[key]

        B = self._band_energy(bands, use_power=use_power)
        if B.size == 0:
            self._cache_freq[key] = np.zeros((len(bands), 0), dtype=float)
            return self._cache_freq[key]

        if relative:
            total = np.sum(self._power_spectrum() if use_power else self._magnitude_spectrum(), axis=0) + EPS
            R = B / total[None, :]
        else:
            R = B

        self._cache_freq[key] = R
        return R
    
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
    
    # Spotify-based Frequency Features
    def _safe_band_mask(self, f_lo, f_hi):
        return (self.freqs >= f_lo) & (self.freqs < f_hi)
    
    def _sub_band_energy_ratios(self, n_bands=12, use_power=True, fmin=50.0, fmax=None):
        key = f"sub_band_energy_ratios_{n_bands}_{'pow' if use_power else 'mag'}_{fmin}_{fmax}"
        if key in self._cache_freq:
            return self._cache_freq[key]

        if fmax is None:
            fmax = self.sr/2.0

        edges = np.linspace(fmin, fmax, n_bands + 1)
        bands = [(edges[i], edges[i + 1]) for i in range (n_bands)]
        R = self._band_ratios(bands, use_power=use_power, relative=True)

        self._cache_freq[key] = R
        return R
    
    def _pitch_class_profile(self, use_power=True):
        """
        Mean energy pooled into 12 pitch classes from the STFT bins.
        """
        key = f"pitch_class_profile_{'pow' if use_power else 'mag'}"
        if key in self._cache_freq:
            return self._cache_freq[key]

        S = self._power_spectrum() if use_power else self._magnitude_spectrum()
        profile = np.zeros(12, dtype=float)

        if S.size > 0:
            for k, f in enumerate(self.freqs):
                if f <= 0.0:
                    continue
                midi = 12.0 * np.log2(f / 440.0) + 69.0
                pc = int(np.round(midi)) % 12
                profile[pc] += float(np.mean(S[k, :]))

        profile /= np.sum(profile) + EPS

        self._cache_freq[key] = profile
        return profile

    def _mean_pitch_class_profile(self, n_bands=12, use_power=True):
        key = f"mean_pitch_class_profile_{n_bands}_{'pow' if use_power else 'mag'}"
        if key in self._cache_freq:
            return self._cache_freq[key]

        profile = self._pitch_class_profile(use_power=use_power)

        self._cache_freq[key] = profile
        return profile

    def _freq_key_templates(self):
        key = "freq_key_templates"
        if key in self._cache_freq:
            return self._cache_freq[key]

        major_profile = np.array([
            6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
            2.52, 5.19, 2.39, 3.66, 2.29, 2.88
        ], dtype=float)
        minor_profile = np.array([
            6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
            2.54, 4.75, 3.98, 2.69, 3.34, 3.17
        ], dtype=float)
        major_profile /= np.sum(major_profile)
        minor_profile /= np.sum(minor_profile)

        templates = np.zeros((24, 12), dtype=float)
        for i in range(12):
            templates[i] = np.roll(major_profile, i)
        for i in range(12):
            templates[12 + i] = np.roll(minor_profile, i)

        self._cache_freq[key] = templates
        return templates

    def _estimate_key_mode_freq(self):
        key = "estimate_key_mode_freq"
        if key in self._cache_freq:
            return self._cache_freq[key]

        profile = self._pitch_class_profile(use_power=True)
        templates = self._freq_key_templates()

        Pn = profile / (np.linalg.norm(profile) + EPS)
        Tn = templates / (np.linalg.norm(templates, axis=1, keepdims=True) + EPS)
        scores = Tn @ Pn

        key_idx = int(np.argmax(scores))
        result = {
            "tonic": key_idx % 12,
            "mode": "major" if key_idx < 12 else "minor",
            "key_idx": key_idx,
            "score": float(scores[key_idx]),
        }

        self._cache_freq[key] = result
        return result

    def _frequency_weights(self):
        f = np.asarray(self.freqs, dtype=float)
        if f.size == 0:
            return np.array([], dtype=float)

        return 1.0 + 0.5*(f/(f.max() + EPS))

    def _loudness_freq_db(self):
        key = "loudness_freq_db"
        if key in self._cache_freq:
            return self._cache_freq[key]

        S = self._power_spectrum()
        if S.size == 0:
            val = -80.0
        else:
            P = float(np.mean(S))
            val = float(10.0*np.log10(P + EPS))
            val = float(max(val, -80.0))

        self._cache_freq[key] = val
        return val

    def _loudness_freq_active_db(self):
        key = "loudness_freq_active_db"
        if key in self._cache_freq:
            return self._cache_freq[key]
        
        S = self._power_spectrum()
        if S.size == 0:
            val = -80.0
        else:
            frame_energy = self._frame_energy()
            if frame_energy.size == 0:
                val = -80.0
            else:
                thr = np.median(frame_energy) + np.std(frame_energy)
                acive = frame_energy >= thr
                if not np.any(active):
                    active = frame_energy > 0
                if not np.any(active):
                    val = -80.0
                else:
                    val = float(10.0*np.log10(np.mean(frame_energy[active]) + EPS))
                    val = float(max(val, -80.0))

        self._cache_freq[key] = val
        return val

    def _energy_freq(self, weighted=False):
        key = f"energy_freq_{'weighted' if weighted else 'unweighted'}"
        if key in self._cache_freq:
            return self._cache_freq[key]

        S = self._power_spectrum()
        if S.size == 0:
            self._cache_freq[key] = 0.0
            return 0.0

        fe_db = self._frame_energy_db()
        p10_db = float(np.percentile(fe_db, 10)) if fe_db.size else -80.0
        p50_db = float(np.percentile(fe_db, 50)) if fe_db.size else -80.0
        p90_db = float(np.percentile(fe_db, 90)) if fe_db.size else -80.0

        if weighted:
            w = self._frequency_weights()
            frame_level = float(np.mean(np.sum(S * w[:, None], axis=0)))
            level_db = 10.0 * np.log10(frame_level + EPS)
            w_level, w_act, w_crest, w_trans = 0.35, 0.40, 0.10, 0.15
        else:
            level_db = 0.55 * p50_db + 0.45 * p90_db
            w_level, w_act, w_crest, w_trans = 0.15, 0.45, 0.20, 0.20

        level_score = float(np.clip((level_db + 40.0) / 50.0, 0.0, 1.0))
        crest_score = float(np.clip((p90_db - p10_db) / 40.0, 0.0, 1.0))

        flux = self._spectral_flux(
            use_power=True, normalize=True, half_wave_rectify=True
        )
        if flux.size > 0:
            flux_med = float(np.median(flux)) + EPS
            flux_mean = float(np.mean(flux))
            flux_p95 = float(np.percentile(flux, 95))
            activity_score = float(
                np.clip(0.5 * flux_mean / flux_med + 0.5 * flux_p95 / flux_med, 0.0, 1.0)
            )
        else:
            activity_score = 0.0

        transient = self._transient_rate(
            use_power=True, normalize=True, half_wave_rectify=True
        )
        transient_score = float(np.clip(transient / (transient + 3.0), 0.0, 1.0))

        energy_normalized = float(
            np.clip(
                w_level * level_score
                + w_act * activity_score
                + w_crest * crest_score
                + w_trans * transient_score,
                0.0,
                1.0,
            )
        )

        self._cache_freq[key] = energy_normalized
        return energy_normalized

    def _speechiness_freq(self):
        key = "speechiness_freq"
        if key in self._cache_freq:
            return self._cache_freq[key]

        flat = self._spectral_flatness(use_power=True)
        entr = self._spectral_entropy(use_power=True, normalize=True)
        flux = self._spectral_flux(use_power=True, normalize=True, half_wave_rectify=True)
        mid_ratio = self._band_ratios([(300.0, 3000.0)], relative=True)[0]
        
        z = 0.35*np.mean(flat) + 0.25*np.mean(entr) + 0.25*np.mean(np.clip(flux/(np.mean(flux) + EPS), 0.0, 1.0)) + 0.15*np.mean(mid_ratio)
        val = safe_clip01(z)

        self._cache_freq[key] = val
        return val

    def _acousticness_freq(self):
        key = "acousticness_freq"
        if key in self._cache_freq:
            return self._cache_freq[key]

        flat = self._spectral_flatness(use_power=True)
        cent = self._spectral_centroid(use_power=True)
        roll = self._spectral_rolloff(roll_percent=0.85, use_power=True)
        slope = self._spectral_slope(use_power=True, log_amp=True)
        harm = self._harmonic_ratio()
        high_ratio = self._band_ratios([(6000.0, self.sr/2.0)], relative=True)[0]

        c_score = 1.0 - np.mean(np.clip(cent/(self.sr/2.0), 0.0, 1.0))
        r_score = 1.0 - np.mean(np.clip(roll/(self.sr/2.0), 0.0, 1.0))
        s_score = 1.0 - np.mean(np.clip(np.abs(slope)/(np.abs(slope).max() + EPS), 0.0, 1.0))
        h_score = np.mean(np.clip(harm, 0.0, 1.0))
        high_score = 1.0 - np.mean(np.clip(high_ratio, 0.0, 1.0))

        z = 0.25*np.mean(flat) + 0.20*c_score + 0.20*r_score + 0.15*s_score + 0.15*h_score + 0.05*high_score
        val = safe_clip01(z)

        self._cache_freq[key] = val
        return val

    def _danceability_freq(self):
        key = "danceability_freq"
        if key in self._cache_freq:
            return self._cache_freq[key]

        flux = self._spectral_flux(use_power=True, normalize=True, half_wave_rectify=True)
        if flux.size < 10:
            self._cache_freq[key] = 0.0
            return 0.0
        
        # Detect rhythmic periodicity using autocorrelation
        flux_centered = flux - np.mean(flux)
        ac = np.correlate(flux_centered, flux_centered, mode="full")
        ac = ac[ac.size//2:]

        if ac[0] > 0:
            ac = ac/ac[0]

        # Look for peaks in the autocorrelation to find periodicity
        frame_rate = self.sr/float(self.H)
        lag_min = int(frame_rate*60/160)  # Max 160 BPM
        lag_max = int(frame_rate*60/80)   # Min 80 BPM

        if lag_min < lag_max < len(ac):
            periodicity = float(np.max(ac[lag_min:lag_max]))
        else:
            periodicity = 0.0

        # Flux energy (percussive has high flux)
        flux_energy = float(np.mean(flux))
        flux_norm = np.tanh(flux_energy*3.0)

        # Combine
        danceability = 0.7*periodicity + 0.3*flux_norm
        danceability = float(np.clip(danceability, 0.0, 1.0))

        self._cache_freq[key] = danceability
        return danceability
    
    def _valence_freq(self):
        key = "valence_freq"
        if key in self._cache_freq:
            return self._cache_freq[key]
        
        flat = self._spectral_flatness(use_power=True)
        cent = self._spectral_centroid(use_power=True)
        skew = self._spectral_skewness(use_power=True)
        kurt = self._spectral_kurtosis(use_power=True, excess=True)
        low_high = self._low_high_band_ratio((0.0, 3000.0), (3000.0, self.sr/2.0))

        bright = np.mean(np.clip(cent/(self.sr/2.0), 0.0, 1.0))
        skew_score = float(np.clip(0.5 + 0.25*np.tanh(np.mean(skew)), 0.0, 1.0))
        kurt_score = float(np.clip(1.0 - 0.25*np.tanh(np.mean(np.abs(kurt))), 0.0, 1.0))

        tonal = float(np.mean(np.clip(self._harmonic_ratio(), 0.0, 1.0)))
        z = (
            0.28 * bright
            + 0.24 * (1.0 - np.mean(flat))
            + 0.18 * skew_score
            + 0.10 * kurt_score
            + 0.10 * np.mean(np.clip(low_high / (low_high + 1.0), 0.0, 1.0))
            + 0.10 * tonal
        )

        mode = self._mode_freq()
        mode_adj = 0.03 if mode == "major" else -0.03

        val = safe_clip01(z + mode_adj)

        self._cache_freq[key] = val
        return val

    def _flux_implied_bpm(self, bpm_min=40.0, bpm_max=240.0):
        """
        Estimate tempo from median inter-onset interval in the spectral-flux envelope.
        """
        key = f"flux_implied_bpm_{bpm_min}_{bpm_max}"
        if key in self._cache_freq:
            return self._cache_freq[key]

        flux = self._spectral_flux(
            use_power=True, normalize=True, half_wave_rectify=True
        )
        if flux.size < 3:
            val = 0.0
        else:
            fs_env = self.sr / float(self.H)
            med = np.median(flux)
            mad = np.median(np.abs(flux - med)) + EPS
            peaks, _ = find_peaks(
                flux,
                height=med + 1.5 * mad,
                distance=max(1, int(fs_env * 60.0 / bpm_max)),
            )
            if peaks.size < 2:
                val = 0.0
            else:
                ibi = np.diff(peaks) / fs_env
                ibi = ibi[(ibi >= 60.0 / bpm_max) & (ibi <= 60.0 / bpm_min)]
                if ibi.size == 0:
                    val = 0.0
                else:
                    fast_ibi = float(np.percentile(ibi, 25))
                    ibi_bpm = float(np.clip(60.0 / fast_ibi, bpm_min, bpm_max))

                    lo = max(1, int(round(fs_env * 0.25)))
                    ac_seg = np.correlate(
                        flux - np.mean(flux), flux - np.mean(flux), mode="full"
                    )
                    ac_seg = ac_seg[ac_seg.size // 2 :]
                    hi = min(int(round(fs_env * 1.0)), ac_seg.size - 1)
                    if hi > lo and ac_seg.size > hi and ac_seg[0] > 0:
                        ac_seg = ac_seg / ac_seg[0]
                        seg = ac_seg[lo : hi + 1]
                        seg_peaks, _ = find_peaks(seg, height=0.25 * float(np.max(seg)))
                        if seg_peaks.size:
                            ac_bpm = float(
                                np.clip(60.0 * fs_env / float(lo + int(seg_peaks[0])), bpm_min, bpm_max)
                            )
                        else:
                            ac_bpm = float(
                                np.clip(
                                    60.0 * fs_env / float(lo + int(np.argmax(seg))),
                                    bpm_min,
                                    bpm_max,
                                )
                            )
                    else:
                        ac_bpm = ibi_bpm

                    if ibi_bpm > 0.0 and ac_bpm > 0.0:
                        lo_bpm = min(ibi_bpm, ac_bpm)
                        hi_bpm = max(ibi_bpm, ac_bpm)
                        val = hi_bpm if hi_bpm > 1.45 * lo_bpm else 0.5 * (ibi_bpm + ac_bpm)
                    else:
                        val = max(ibi_bpm, ac_bpm)

                    val = float(np.clip(val, bpm_min, bpm_max))

        self._cache_freq[key] = val
        return val
    
    def _tempo_freq(self):
        key = "tempo_freq"
        if key in self._cache_freq:
            return self._cache_freq[key]

        flux = self._spectral_flux(
            use_power=True, normalize=True, half_wave_rectify=True
        )
        if flux.size < 3:
            self._cache_freq[key] = 0.0
            return 0.0

        x = flux - np.mean(flux)
        ac = np.correlate(x, x, mode="full")
        ac = ac[ac.size // 2 :]
        if ac.size < 3 or ac[0] <= 0:
            self._cache_freq[key] = 0.0
            return 0.0

        ac = ac / ac[0]

        fs_env = self.sr / float(self.H)
        bpm_min, bpm_max = 40.0, 240.0
        lag_min = max(1, int(round(fs_env / (bpm_max / 60.0))))
        lag_max = min(ac.size - 1, int(round(fs_env / (bpm_min / 60.0))))
        if lag_max <= lag_min:
            self._cache_freq[key] = 0.0
            return 0.0

        region = ac[lag_min : lag_max + 1]
        peak_height = 0.25 * float(np.max(region))
        min_distance = max(1, int(round(fs_env * 60.0 / bpm_max)))
        peaks, _ = find_peaks(region, height=peak_height, distance=min_distance)

        if peaks.size:
            tau = lag_min + int(peaks[0])
        else:
            tau = lag_min + int(np.argmax(region))

        def _ac_at_lag(lag):
            lag = int(lag)
            if lag <= 0 or lag >= len(ac):
                return 0.0
            tol = max(2, int(round(lag * 0.08)))
            s = max(0, lag - tol)
            e = min(len(ac), lag + tol + 1)
            return float(np.max(ac[s:e])) if s < e else 0.0

        bpm = float(60.0 * fs_env / float(tau + EPS))

        # Upgrade to faster octave when half-period correlation is stronger
        half_tau = tau // 2
        if half_tau >= lag_min and _ac_at_lag(half_tau) > _ac_at_lag(tau):
            bpm = float(60.0 * fs_env / float(half_tau + EPS))

        val = float(np.clip(bpm, bpm_min, bpm_max))
        self._cache_freq[key] = val
        return val
    
    def _liveness_freq(self):
        key = "liveness_freq"
        if key in self._cache_freq:
            return self._cache_freq[key]
        
        flux = self._spectral_flux(use_power=True, normalize=True, half_wave_rectify=True)
        transient = self._transient_rate(use_power=True, normalize=True, half_wave_rectify=True)
        high_noise = np.mean(self._band_ratios([(4000.0, self.sr/2.0)], relative=True)[0])
        flat = np.mean(self._spectral_flatness(use_power=True))
        slope = np.mean(np.abs(self._spectral_slope(use_power=True, log_amp=True)))

        z = 0.30*float(np.clip(transient/(transient + 5.0), 0.0, 1.0)) + 0.20*float(np.clip(np.mean(flux)/(np.mean(flux) + 1.0), 0.0, 1.0)) + 0.20*high_noise + 0.15*flat + 0.15*float(np.clip(slope/(slope + 10.0), 0.0, 1.0))
        val = safe_clip01(z)

        self._cache_freq[key] = val
        return val
    
    def _instrumentalness_freq(self):
        key = "instrumentalness_freq"
        if key in self._cache_freq:
            return self._cache_freq[key]

        flat = np.mean(self._spectral_flatness(use_power=True))
        hnr = self._harmonic_ratio()
        hnr_mean = float(np.mean(hnr)) if np.size(hnr) else 0.0
        inh = self._inharmonicity()
        inh_mean = float(np.mean(inh)) if np.size(inh) else 0.0
        voiced_band = np.mean(self._band_ratios([(300.0, 3400.0)], relative=True)[0])

        z = 0.35*(1.0 - flat) + 0.25*hnr_mean + 0.20*(1.0 - float(np.clip(inh_mean/(inh_mean + 1.0), 0.0, 1.0))) + 0.20*(1.0 - voiced_band)
        val = safe_clip01(z)

        self._cache_freq[key] = val
        return val

    def _key_freq(self):
        key = "key_freq"
        if key in self._cache_freq:
            return self._cache_freq[key]

        tonic = self._estimate_key_mode_freq()["tonic"]

        self._cache_freq[key] = tonic
        return tonic

    def _mode_freq(self):
        key = "mode_freq"
        if key in self._cache_freq:
            return self._cache_freq[key]

        mode = self._estimate_key_mode_freq()["mode"]

        self._cache_freq[key] = mode
        return mode
    
    def _time_signature_freq(self):
        key = "time_signature_freq"
        if key in self._cache_freq:
            return self._cache_freq[key]

        periodicity = self._beat_periodicity(use_power=True, normalize=True, half_wave_rectify=True)
    
        # If no clear beat, default to 4/4
        if periodicity < 0.15:
            self._cache_freq[key] = 4
            return 4
        
        flux = self._spectral_flux(use_power=True, normalize=True, half_wave_rectify=True)
        if flux.size < 10:
            self._cache_freq[key] = 4
            return 4
        
        # Autocorrelation of flux
        ac = np.correlate(flux - np.mean(flux), flux - np.mean(flux), mode="full")
        ac = ac[ac.size // 2:]
        
        if ac[0] > 0:
            ac = ac / ac[0]
        
        fs_env = self.sr / float(self.H)
        
        # For 3/4 vs 4/4 discrimination:
        # 3/4 has emphasis every 3 beats
        # 4/4 has emphasis every 2 or 4 beats
        
        # Check at different beat multiples
        # Assuming typical tempo ~120 BPM = 2 beats/sec
        
        # 3-beat pattern (3/4): check at 1.5 beat intervals
        lag_3beat = int(round(fs_env * 0.75))  # 3/4 of a second at 120 BPM
        score3 = ac[lag_3beat] if lag_3beat < ac.size else 0.0
        
        # 4-beat pattern (4/4): check at 2 beat intervals (backbeat)
        lag_4beat = int(round(fs_env * 1.0))  # 1 second at 120 BPM
        score4 = ac[lag_4beat] if lag_4beat < ac.size else 0.0
        
        # Also check at measure level
        lag_3measure = int(round(fs_env * 1.5))  # Full 3/4 measure
        score3_measure = ac[lag_3measure] if lag_3measure < ac.size else 0.0
        
        lag_4measure = int(round(fs_env * 2.0))  # Full 4/4 measure
        score4_measure = ac[lag_4measure] if lag_4measure < ac.size else 0.0
        
        # Combine scores
        total_score3 = score3 + score3_measure
        total_score4 = score4 + score4_measure
        
        # Need clear preference for 3/4 (since 4/4 is more common)
        val = 3 if total_score3 > total_score4 * 1.1 else 4
        
        self._cache_freq[key] = val
        return val

    def spotify_audio_features(self, weights=None):
        loudness = self._loudness_freq_db()
        energy = self._energy_freq(weighted=True)
        speechiness = self._speechiness_freq()
        acousticness = self._acousticness_freq()
        danceability = self._danceability_freq()
        valence = self._valence_freq()
        tempo = self._tempo_freq()
        liveness = self._liveness_freq()
        instrumentalness = self._instrumentalness_freq()
        key = self._key_freq()
        mode = self._mode_freq()
        time_signature = self._time_signature_freq()

        loudness_score = safe_clip01((loudness + 80.0) / 80.0)
        energy_score = safe_clip01(energy / (energy + 1.0))
        tempo_score = safe_clip01(tempo / 240.0)
        key_score = safe_clip01(key / 11.0)
        mode_score = 1.0 if mode == "major" else 0.0
        time_sig_score = 1.0 if time_signature == 4 else 0.5 if time_signature == 3 else 0.0

        vals = np.array([
            loudness_score,
            energy_score,
            speechiness,
            acousticness,
            danceability,
            valence,
            tempo_score,
            liveness,
            instrumentalness,
            key_score,
            mode_score,
            time_sig_score,
        ], dtype=float)

        if weights is None:
            weights = np.array([0.12, 0.12, 0.08, 0.10, 0.12, 0.08, 0.08, 0.10, 0.12, 0.06, 0.05, 0.07], dtype=float)

        fused = float(np.sum(vals * weights) / (np.sum(weights) + EPS))

        return {
            "loudness_db": loudness,
            "energy": energy,
            "speechiness": speechiness,
            "acousticness": acousticness,
            "danceability": danceability,
            "valence": valence,
            "tempo_bpm": tempo,
            "liveness": liveness,
            "instrumentalness": instrumentalness,
            "key": key,
            "mode": mode,
            "time_signature": time_signature,
            "spotify_fused": float(np.clip(fused, 0.0, 1.0)),
        }
    
# Chromagram class
class ChromagramFeatures():
    def __init__(self, sig: AudioSignal):
        self.sig = sig
        self.y = sig.y
        self.sr = sig.sr
        self.N = sig.N
        self.H = sig.H

        self.X = librosa.stft(
            self.y,
            n_fft=self.N,
            hop_length=self.H,
            win_length=self.N,
            window="hann",
            center=True
        )
        self.X_mag = np.abs(self.X)
        self.freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.N)

        self._cache_chroma = {}

    @staticmethod
    def midi_to_hz(midi, pitch_ref=69, freq_ref=440.0):
        return (2.0**((midi - pitch_ref)/12.0))*freq_ref
    
    @staticmethod
    def hz_to_midi(hz, pitch_ref=69, freq_ref=440.0):
        return 12.0*np.log2(hz/freq_ref) + pitch_ref
    
    def _pitch_bin_edges(self, midi, pitch_ref=69, freq_ref=440.0):
        lower = self.midi_to_hz(midi - 0.5, pitch_ref=pitch_ref, freq_ref=freq_ref)
        upper = self.midi_to_hz(midi + 0.5, pitch_ref=pitch_ref, freq_ref=freq_ref)
        return lower, upper
    
    def _pool_pitch(self, midi):
        lower, upper = self._pitch_bin_edges(midi)
        mask = (self.freqs >= lower) & (self.freqs < upper)
        
        return np.where(mask)[0]
    
    def _compute_spec_log_freq(self, n_pitches=128):
        Y_LF = np.zeros((n_pitches, self.X_mag.shape[1]), dtype=float)
        for p in range(n_pitches):
            k = self._pool_pitch(p)
            if k.size > 0:
                Y_LF[p] = self.X_mag[k, :].sum(axis=0)

        return Y_LF, np.arange(n_pitches)
    
    def _compute_chromagram(self, X_LF):
        chroma = np.zeros((12, X_LF.shape[1]), dtype=float)
        p = np.arange(X_LF.shape[0])

        for c in range(12):
            mask = (p%12) == c
            chroma[c, :] = X_LF[mask, :].sum(axis=0)

        return chroma
    
    def _chroma(self):
        key = "chroma"
        if key in self._cache_chroma:
            return self._cache_chroma[key]
        
        Y_LF, F_coef_pitch = self._compute_spec_log_freq()
        C = self._compute_chromagram(Y_LF)
        self._cache_chroma[key] = C
        self._cache_chroma["chroma_freqs"] = F_coef_pitch
        self._cache_chroma["Y_LF"] = Y_LF
        
        return C
    
    def _chroma_db(self):
        key = "chroma_db"
        if key in self._cache_chroma:
            return self._cache_chroma[key]
        
        C = self._chroma()
        C_db = 10.0*np.log10(C + EPS)

        self._cache_chroma[key] = C_db
        return C_db
    
    def _normalize_chroma(self, C):
        S = np.sum(C, axis=0, keepdims=True) + EPS
        return C/S
    
    def _chroma_profile(self, normalize=True, use_db=False):
        C = self._chroma_db() if use_db else self._chroma()
        if normalize:
            C = self._normalize_chroma(C)

        return C

    def _mean_chroma(self, normalize=True, use_db=False):
        key = f"mean_chroma_{'norm' if normalize else 'raw'}_{'db' if use_db else 'lin'}"
        if key in self._cache_chroma:
            return self._cache_chroma[key]

        C = self._chroma_profile(normalize=normalize, use_db=use_db)
        mean_chroma = np.mean(C, axis=1)

        self._cache_chroma[key] = mean_chroma
        return mean_chroma
    
    def _chord_templates(self):
        """
        Generate chord templates for common chord types
        """
        key = "chord_templates"
        if key in self._cache_chroma:
            return self._cache_chroma[key]
        
        # Define chord types (intervals from root)
        chord_types = {
            'maj': [0, 4, 7],           # Major triad
            'min': [0, 3, 7],           # Minor triad
            '7': [0, 4, 7, 10],         # Dominant 7th
            'maj7': [0, 4, 7, 11],      # Major 7th
            'min7': [0, 3, 7, 10],      # Minor 7th
            'dim': [0, 3, 6],           # Diminished
            'aug': [0, 4, 8],           # Augmented
            'sus4': [0, 5, 7],          # Suspended 4th
            'sus2': [0, 2, 7],          # Suspended 2nd
        }
        
        templates = []
        labels = []
        roots = []
        qualities = []
        
        # Generate templates for all 12 roots × all chord types
        for root in range(12):
            for quality, intervals in chord_types.items():
                # Create chroma vector
                chroma = np.zeros(12, dtype=float)
                for interval in intervals:
                    chroma[(root + interval) % 12] = 1.0
                
                # Normalize
                chroma = chroma / (np.sum(chroma) + EPS)
                
                templates.append(chroma)
                labels.append(f"{root}:{quality}")
                roots.append(root)
                qualities.append(quality)
        
        templates = np.array(templates, dtype=float)
        
        result = {
            'templates': templates,
            'labels': labels,
            'roots': roots,
            'qualities': qualities
        }
        
        self._cache_chroma[key] = result
        return result
    
    def _pitch_class_profile(self, normalize=True, use_db=False):
        return self._chroma_profile(normalize=normalize, use_db=use_db)
    
    def _pitch_class_deviation(self, normalize=True, use_db=False):
        key = f"pitch_class_deviation_{'norm' if normalize else 'raw'}_{'db' if use_db else 'lin'}"
        if key in self._cache_chroma:
            return self._cache_chroma[key]

        P = self._pitch_class_profile(normalize=normalize, use_db=use_db)
        c_star = np.argmax(P, axis=0)
        idx = np.arange(12)[:, None]
        dist = np.minimum(np.abs(idx - c_star[None, :]), 12 - np.abs(idx - c_star[None, :]))
        dev = np.sum((dist**2)*P, axis=0)

        self._cache_chroma[key] = dev
        return dev
    
    def _chroma_centroid(self, normalize=True, use_db=False):
        """
        Circular centroid of chroma distribution per frame

        Formula:
            x_t = sum(c*p_c)/sum(p_c) for c in [0..11]
            y_t = sum(sin(2*pi*c/12)*p_c)/sum(p_c)
            mu_t = atan2(y_t, x_t) in radians, then converted to [0, 12) in pitch classes
        """
        key = f"chroma_centroid_{'norm' if normalize else 'raw'}_{'db' if use_db else 'lin'}"
        if key in self._cache_chroma:
            return self._cache_chroma[key]

        P = self._chroma_profile(normalize=normalize, use_db=use_db)
        theta = 2.0*np.pi*np.arange(12)/12.0
        x = np.sum(P*np.cos(theta)[:, None], axis=0)
        y = np.sum(P*np.sin(theta)[:, None], axis=0)
        mu = np.arctan2(y, x)

        self._cache_chroma[key] = mu
        return mu
    
    def _chroma_spread(self, normalize=True, use_db=False):
        """
        Circular spread of chroma distribution per frame

        Formula:
            mu_t = chroma centroid in radians
            spread_t = sqrt(sum(p_c*(theta_c - mu_t)^2)/sum(p_c)) where theta_c is angle of chroma bin c
        """
        key = f"chroma_spread_{'norm' if normalize else 'raw'}_{'db' if use_db else 'lin'}"
        if key in self._cache_chroma:
            return self._cache_chroma[key]

        P = self._chroma_profile(normalize=normalize, use_db=use_db)
        mu = self._chroma_centroid(normalize=normalize, use_db=use_db)

        # Map mu to fractional chroma [0, 12)
        centroid_idx = (12.0*mu/(2.0*np.pi))%12.0

        idx = np.arange(12)[:, None]
        dist = np.minimum(np.abs(idx - centroid_idx[None, :]), 12 - np.abs(idx - centroid_idx[None, :]))
        spread = np.sqrt(np.sum((dist**2)*P, axis=0))

        self._cache_chroma[key] = spread
        return spread

    def _chroma_skewness(self, normalize=True, use_db=False):
        """
        Circular skewness of chroma distribution per frame

        Formula:
            mu_t = chroma centroid in radians
            skew_t = sum(p_c*sin(2*(theta_c - mu_t)))/sum(p_c)
        """
        key = f"chroma_skewness_{'norm' if normalize else 'raw'}_{'db' if use_db else 'lin'}"
        if key in self._cache_chroma:
            return self._cache_chroma[key]

        P = self._chroma_profile(normalize=normalize, use_db=use_db)
        mu = self._chroma_centroid(normalize=normalize, use_db=use_db)
        centroid_idx = (12.0*mu/(2.0*np.pi))%12.0

        idx = np.arange(12)[:, None]
        dist = np.minimum(np.abs(idx - centroid_idx[None, :]), 12 - np.abs(idx - centroid_idx[None, :]))

        m2 = np.sum((dist**2)*P, axis=0)
        m3 = np.sum((dist**3)*P, axis=0)

        skew = m3/(np.power(m2, 1.5) + EPS)

        
        self._cache_chroma[key] = skew
        return skew
    
    def _chroma_kurtosis(self, normalize=True, use_db=False, excess=True):
        """
        Circular kurtosis proxy around the chroma centroid per frame

        Formula:
            mu_t = chroma centroid in radians
            kurt_t = sum(p_c*(theta_c - mu_t)^4)/sum(p_c) / (sum(p_c*(theta_c - mu_t)^2)/sum(p_c))^2
        """
        key = f"chroma_kurtosis_{'norm' if normalize else 'raw'}_{'db' if use_db else 'lin'}_{'excess' if excess else 'raw'}"
        if key in self._cache_chroma:
            return self._cache_chroma[key]

        P = self._chroma_profile(normalize=normalize, use_db=use_db)
        mu = self._chroma_centroid(normalize=normalize, use_db=use_db)
        centroid_idx = (12.0*mu/(2.0*np.pi))%12.0

        idx = np.arange(12)[:, None]
        dist = np.minimum(np.abs(idx - centroid_idx[None, :]), 12 - np.abs(idx - centroid_idx[None, :]))

        m2 = np.sum((dist**2)*P, axis=0)
        m4 = np.sum((dist**4)*P, axis=0)

        kurt = m4/(np.square(m2) + EPS)

        if excess:
            kurt = kurt - 3.0

        self._cache_chroma[key] = kurt
        return kurt
    
    def _chroma_template(self):
        """
        Returns simple major/minor chord templates for chroma matching
        Output shape: (24, 12) array where rows correspond to major/minor templates for each root
        Order of rows: C major, C minor, C# major, C# minor, ..., B major, B minor
        """
        key = "chroma_template"
        if key in self._cache_chroma:
            return self._cache_chroma[key]

        # Krumhansl-Schmuckler major profile (C major)
        major_profile = np.array([
            6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 
            2.52, 5.19, 2.39, 3.66, 2.29, 2.88
        ])
        
        # Krumhansl-Schmuckler minor profile (C minor)
        minor_profile = np.array([
            6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
            2.54, 4.75, 3.98, 2.69, 3.34, 3.17
        ])
        
        # Normalize profiles
        major_profile = major_profile / np.sum(major_profile)
        minor_profile = minor_profile / np.sum(minor_profile)
        
        # Create templates for all 24 keys by circular rotation
        templates = np.zeros((24, 12), dtype=float)
        
        # Major keys (0-11)
        for i in range(12):
            templates[i] = np.roll(major_profile, i)
        
        # Minor keys (12-23)
        for i in range(12):
            templates[12 + i] = np.roll(minor_profile, i)
        
        self._cache_chroma[key] = templates
        return templates
    
    def _template_labels(self):
        names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        labels = [f"{n}:maj" for n in names] + [f"{n}:min" for n in names]

        return labels
    
    # Tonality/Chord
    def _tonal_clarity(self, normalize=True, use_db=False, method="cosine"):
        """
        A simple tonal clarity measure based on the best key estimation score normalized by the mean score
        """
        key = f"tonal_clarity_{'norm' if normalize else 'raw'}_{'db' if use_db else 'lin'}_{method}"
        if key in self._cache_chroma:
            return self._cache_chroma[key]
        
        result = self._key_estimation(normalize=normalize, use_db=use_db, method=method)
        scores = result["scores"]
        s_sorted = np.sort(scores)
        best = float(s_sorted[-1])
        second = float(s_sorted[-2]) if len(s_sorted) > 1 else 0.0

        margin = best - second
        clarity = margin/(best + EPS)

        result = {
            "tonal_clarity": clarity,
            "best_score": best,
            "margin": margin
        }

        self._cache_chroma[key] = result
        return result
    
    def _harmonic_entropy(self, normalize=True, use_db=False, method="cosine", beta=1.0):
        """
        Entropy of the key estimation scores across all templates, normalized by log(num_templates)
        Higher values indicate a more ambiguous tonal center, while lower values indicate a clearer key
        """
        key = f"harmonic_entropy_{'norm' if normalize else 'raw'}_{'db' if use_db else 'lin'}_{method}_beta{beta}"
        if key in self._cache_chroma:
            return self._cache_chroma[key]
        
        result = self._key_estimation(normalize=normalize, use_db=use_db, method=method)
        scores = result["scores"]

        z = beta*(scores - np.max(scores))
        p = np.exp(z)
        p = p/(np.sum(p) + EPS)

        H = -np.sum(p*np.log(p + EPS))
        H_norm = H/np.log(len(scores) + EPS)

        result = {
            "harmonic_entropy": H_norm,
            "raw_entropy": H,
            "probabilities": p
        }

        self._cache_chroma[key] = result
        return result
    
    def _consonance_dissonance(self, normalize=True, use_db=False, method="cosine"):
        """
        Consonance: average score of the best matching major/minor template
        Dissonance: average score of the non-best templates (1 - consonance)
        """
        key = f"consonance_dissonance_{'norm' if normalize else 'raw'}_{'db' if use_db else 'lin'}_{method}"
        if key in self._cache_chroma:
            return self._cache_chroma[key]
        
        result = self._tonal_clarity(normalize=normalize, use_db=use_db, method=method)
        best = result["best_score"]

        consonance = safe_clip01(best)
        dissonance = float(1.0 - consonance)

        result = {
            "consonance": consonance,
            "dissonance": dissonance
        }

        self._cache_chroma[key] = result
        return result
    
    # Harmonic Features
    def _chord_detection(self, normalize=True, use_db=False, method="cosine"):
        """
        Detect chord per frame

        Returns:
            chord_idx: shape (T,)
            chord_labels: list[str]
            chord_scores: shape (T,)
            all_scores: shape (24, T)
        """
        key = f"chord_detection_{'norm' if normalize else 'raw'}_{'db' if use_db else 'lin'}_{method}"
        if key in self._cache_chroma:
            return self._cache_chroma[key]
        
        C = self._chroma_profile(normalize=normalize, use_db=use_db)
        templates = self._chord_templates()
        TPL = templates['templates']
        labels = templates['labels']
        roots = np.asarray(templates['roots'])
        qualities = np.asarray(templates['qualities'])

        if method == "cosine":
            Cn = C/(np.linalg.norm(C, axis=0, keepdims=True) + EPS)
            Tn = TPL/(np.linalg.norm(TPL, axis=1, keepdims=True) + EPS)
            scores = Tn @ Cn
        elif method == "dot":
            scores = TPL @ C
        else:
            raise ValueError("method must be 'cosine' or 'dot'")
        
        chord_idx = np.argmax(scores, axis=0)
        best_scores = scores[chord_idx, np.arange(scores.shape[1])]
        chord_labels = [labels[i] for i in chord_idx]

        result = {
            "chord_idx": chord_idx,
            "chord_labels": chord_labels,
            "scores": scores,
            "best_scores": best_scores,
            "roots": roots[chord_idx],
            "qualities": qualities[chord_idx]
        }

        self._cache_chroma[key] = result
        return result
    
    def _chord_progression_mapping(self, normalize=True, use_db=False, method="cosine"):
        """
        Sequence of chord labels plus transition counts/probabilities
        """
        key = f"chord_progression_{'norm' if normalize else 'raw'}_{'db' if use_db else 'lin'}_{method}"
        if key in self._cache_chroma:
            return self._cache_chroma[key]
        
        det = self._chord_detection(normalize=normalize, use_db=use_db, method=method)
        chord_labels = det["chord_labels"]

        if len(chord_labels) == 0:
            result = {
                "labels": [],
                "transition_counts": np.zeros((0, 0), dtype=float),
                "transition_probs": np.zeros((0, 0), dtype=float)
            }
            self._cache_chroma[key] = result
            return result
        
        uniq = sorted(set(chord_labels))
        idx_map = {lab: i for i, lab in enumerate(uniq)}
        n = len(uniq)

        counts = np.zeros((n, n), dtype=float)
        for a, b in zip(chord_labels[:-1], chord_labels[1:]):
            counts[idx_map[a], idx_map[b]] += 1.0

        probs = counts/(np.sum(counts, axis=1, keepdims=True) + EPS)

        result = {
            "labels": uniq,
            "transition_counts": counts,
            "transition_probs": probs
        }

        self._cache_chroma[key] = result
        return result
    
    def _harmonic_rhythm(self, normalize=True, use_db=False, method="cosine"):
        """
        Chord-change rae and average chord duration
        """
        key = f"harmonic_rhythm_{'norm' if normalize else 'raw'}_{'db' if use_db else 'lin'}_{method}"
        if key in self._cache_chroma:
            return self._cache_chroma[key]
        
        det = self._chord_detection(normalize=normalize, use_db=use_db, method=method)
        labels = det["chord_labels"]

        if len(labels) < 2:
            result = {
                "change_rate": 0.0,
                "avg_duration_sec": 0.0,
                "durations_sec": np.array({}, dtype=float),
                "num_changes": 0
            }

            self._cache_chroma[key] = result
            return result
        
        changes = np.where(np.array(labels[1:]) != np.array(labels[:-1]))[0] + 1
        boundaries = np.concatenate([[0], changes, [len(labels)]])
        frame_durations = np.diff(boundaries)

        sec_per_frame = self.H/float(self.sr)
        durations_sec = frame_durations*sec_per_frame

        num_changes = int(len(boundaries) - 2)
        total_sec = len(labels)*sec_per_frame
        change_rate = num_changes/(total_sec + EPS)
        avg_duration_sec = float(np.mean(durations_sec)) if durations_sec.size > 0 else 0.0

        result = {
            "change_rate": change_rate,
            "avg_duration_sec": avg_duration_sec,
            "durations_sec": durations_sec,
            "num_changes": num_changes,
            "boundaries": boundaries
        }

        self._cache_chroma[key] = result
        return result
    
    def _root_motion_analysis(self, normalize=True, use_db=False, method="cosine"):
        """
        Analyze root movement between successive detected chords
        Returns root interval histogram and average circular motion
        """
        key = f"root_motion_{'norm' if normalize else 'raw'}_{'db' if use_db else 'lin'}_{method}"
        if key in self._cache_chroma:
            return self._cache_chroma[key]
        
        det = self._chord_detection(normalize=normalize, use_db=use_db, method=method)
        roots = np.asarray(det["roots"], dtype=int)

        if roots.size < 2:
            result = {
                "intervals": np.array([], dtype=int),
                "histogram": np.zeros(12, dtype=float),
                "avg_motion": 0.0
            }

            self._cache_chroma[key] = result
            return result
        
        intervals = (roots[1:] - roots[:-1])%12
        hist = np.bincount(intervals, minlength=12).astype(float)
        hist = hist/(np.sum(hist) + EPS)

        circular_motion = np.minimum(intervals, 12 - intervals)
        avg_motion = float(np.mean(circular_motion))

        result = {
            "intervals": intervals,
            "histogram": hist,
            "avg_motion": avg_motion
        }

        self._cache_chroma[key] = result
        return result
    
    def _tonal_stability_index(self, normalize=True, use_db=False, method="cosine"):
        """
        Tonal stability index with proper scaling
        """
        key = f"tonal_stability_{'norm' if normalize else 'raw'}_{'db' if use_db else 'lin'}_{method}"
        if key in self._cache_chroma:
            return self._cache_chroma[key]
        
        det = self._chord_detection(normalize=normalize, use_db=use_db, method=method)
        best_scores = np.asarray(det["best_scores"], dtype=float)
        labels = np.asarray(det["chord_labels"])

        if best_scores.size == 0:
            self._cache_chroma[key] = 0.0
            return 0.0
        
        # Component 1: Confidence
        mean_score = float(np.mean(best_scores))
        confidence = (mean_score - 0.55) / 0.43
        confidence = float(np.clip(confidence, 0.0, 1.0))
        
        # Component 2: Persistence (exponential scaling)
        if labels.size < 2:
            persistence_raw = 1.0
        else:
            persistence_raw = float(np.mean(labels[1:] == labels[:-1]))
        
        if persistence_raw < 0.5:
            persistence = 0.0
        else:
            normalized = (persistence_raw - 0.5) / 0.5
            persistence = float(normalized ** 3)
        
        # Component 3: Transition diversity (FIXED)
        prog = self._chord_progression_mapping(normalize=normalize, use_db=use_db, method=method)
        counts = prog["transition_counts"]
        
        if counts.size == 0:
            predictability = 1.0
        else:
            # Count unique transitions
            num_transitions = int(np.sum(counts > 0))
            num_chords = counts.shape[0]
            
            # Normalize by number of chords
            # Few chords with few transitions = high predictability
            # Many chords with many transitions = low predictability
            transition_density = num_transitions / (num_chords + EPS)
            predictability = 1.0 / (1.0 + transition_density)
            predictability = float(np.clip(predictability, 0.0, 1.0))
        
        # Combined TSI
        tsi = 0.5 * persistence + 0.3 * confidence + 0.2 * predictability
        tsi = float(np.clip(tsi, 0.0, 1.0))
        
        self._cache_chroma[key] = tsi
        return tsi
    
    # Temporal Chroma Features
    def _chroma_autocorrelation(self, lag_max=64, normalize=True, use_db=False):
        """
        Autocorrelation of chroma over time

        Returns:
            acf: shape (12, lag_max + 1)
            acf_mean: shape (lag_max + 1,)
        """
        key = f"chroma_autocorr_{lag_max}_{'norm' if normalize else 'raw'}_{'db' if use_db else 'lin'}"
        if key in self._cache_chroma:
            return self._cache_chroma[key]
        
        P = self._chroma_profile(normalize=normalize, use_db=use_db)
        T = P.shape[1]
        lag_max = int(min(lag_max, max(0, T - 1)))

        acf = np.zeros((12, lag_max + 1), dtype=float)
        for c in range(12):
            x = P[c] - np.mean(P[c])
            den = np.sum(x*x) + EPS
            for tau in range(lag_max + 1):
                if tau == 0:
                    acf[c, tau] = 1.0
                else:
                    acf[c, tau] = np.sum(x[:-tau]*x[tau:])/den

        acf_mean = np.mean(acf, axis=0)

        result = {
            "acf": acf,
            "acf_mean": acf_mean
        }

        self._cache_chroma[key] = result
        return result
    
    def _chroma_variability(self, normalize=True, use_db=False):
        """
        Variability of each chroma bin over time
        """
        key = f"chroma_variability_{'norm' if normalize else 'raw'}_{'db' if use_db else 'lin'}"
        if key in self._cache_chroma:
            return self._cache_chroma[key]
        
        P = self._chroma_profile(normalize=normalize, use_db=use_db)
        mu = np.mean(P, axis=1, keepdims=True)
        sigma = np.sqrt(np.mean((P - mu)**2, axis=1))

        result = {
            "per_bin_std": sigma,
            "mean_variability": float(np.mean(sigma)),
            "median_variability": float(np.median(sigma))
        }

        self._cache_chroma[key] = result
        return result
        
    def _chroma_smoothness(self, normalize=True, use_db=False, metric="l2"):
        """
        Smoothness based on frame-to-frame chroma differences
        """
        key = f"chroma_smoothness_{'norm' if normalize else 'raw'}_{'db' if use_db else 'lin'}_{metric}"
        if key in self._cache_chroma:
            return self._cache_chroma[key]
        
        P = self._chroma_profile(normalize=normalize, use_db=use_db)

        if P.shape[1] < 2:
            result = {
                "smoothness": 1.0,
                "mean_diff": 0.0,
                "diffs": np.array([], dtype=float)
            }

            self._cache_chroma[key] = result
            return result
        
        diffs = np.diff(P, axis=1)

        if metric == "l1":
            d = np.sum(np.abs(diffs), axis=0)
            d /= 2.0
            smooth = 1.0 - float(np.mean(d))
        elif metric == "l2":
            d = np.linalg.norm(diffs, axis=0)
            smooth = float(np.exp(-np.mean(d)))
        else:
            raise ValueError("metric must 'l1' or 'l2'")
        
        result = {
            "smoothness": safe_clip01(smooth),
            "mean_diff": float(np.mean(d)),
            "diffs": d
        }

        self._cache_chroma[key] = result
        return result
    
    def _dominant_pitch_track(self, normalize=True, use_db=False):
        """
        Dominant chroma bin per frame
        """
        key = f"dominant_pitch_track_{'norm' if normalize else 'raw'}_{'db' if use_db else 'lin'}"
        if key in self._cache_chroma:
            return self._cache_chroma[key]
        
        P = self._chroma_profile(normalize=normalize, use_db=use_db)
        dom = np.argmax(P, axis=0)

        self._cache_chroma[key] = dom
        return dom
    
    def _tuning_deviation_detection(self, top_k=20):
        """
        Estimate cents drift from equal temperament using dominant spectral peaks

        Returns:
            cents_per_peak: (top_k,)
            track_cents: scalar median drift
        """
        key = f"tuning_deviation_{top_k}"
        if key in self._cache_chroma:
            return self._cache_chroma[key]
        
        mag = self.X_mag
        if mag.size == 0:
            result = {
                "cents_per_peak": np.array([], dtype=float),
                "track_cents": 0.0
            }
            self._cache_chroma[key] = result
            return result
        
        peak_bins = np.argsort(np.mean(mag, axis=1))[-int(top_k):]
        peak_freqs = self.freqs[peak_bins]
        peak_freqs = peak_freqs[peak_freqs > 0]

        if peak_freqs.size == 0:
            result = {
                "cents_per_peak": np.array([], dtype=float),
                "track_cents": 0.0
            }
            self._cache_chroma[key] = result
            return result
        
        midi = self.hz_to_midi(peak_freqs)
        nearest = np.round(midi)
        cents = 100.0*(midi - nearest)

        result = {
            "cents_per_peak": cents,
            "track_cents": float(np.median(cents)),
            "mean_abs_cents": float(np.mean(np.abs(cents)))
        }

        self._cache_chroma[key] = result
        return result

    # Spotify-based Features + Helpers
    def _mean_chroma_profile(self, normalize=True, use_db=False):
        key = f"mean_chroma_profile_{'norm' if normalize else 'raw'}_{'db' if use_db else 'lin'}"
        if key in self._cache_chroma:
            return self._cache_chroma[key]
        
        P = self._chroma_profile(normalize=normalize, use_db=use_db)
        prof = np.mean(P, axis=1) if P.size else np.zeros(12, dtype=float)

        self._cache_chroma[key] = prof
        return prof
    
    def _chroma_entropy(self, normalize=True, use_db=False):
        key = f"chroma_entropy_{'norm' if normalize else 'raw'}_{'db' if use_db else 'lin'}"
        if key in self._cache_chroma:
            return self._cache_chroma[key]
        
        P = self._chroma_profile(normalize=normalize, use_db=use_db)
        if P.size == 0:
            self._cache_chroma[key] = 0.0
            return 0.0
        
        p = P/(np.sum(P, axis=0, keepdims=True) + EPS)
        H = -np.sum(p*np.log2(p + EPS), axis=0)
        val = float(np.mean(H)/np.log2(12))
        val = float(np.clip(val, 0.0, 1.0))

        self._cache_chroma[key] = val
        return val

    def _chroma_flux_mean(self, normalize=True, use_db=False):
        key = f"chroma_flux_mean_{'norm' if normalize else 'raw'}_{'db' if use_db else 'lin'}"
        if key in self._cache_chroma:
            return self._cache_chroma[key]

        P = self._chroma_profile(normalize=normalize, use_db=use_db)
        if P.shape[1] < 2:
            self._cache_chroma[key] = 0.0
            return 0.0

        d = np.diff(P, axis=1)
        val = float(np.mean(np.linalg.norm(d, axis=0)))

        self._cache_chroma[key] = val
        return val
    
    def _chroma_flux_variance(self, normalize=True, use_db=False):
        key = f"chroma_flux_variance_{'norm' if normalize else 'raw'}_{'db' if use_db else 'lin'}"
        if key in self._cache_chroma:
            return self._cache_chroma[key]

        P = self._chroma_profile(normalize=normalize, use_db=use_db)
        if P.shape[1] < 2:
            self._cache_chroma[key] = 0.0
            return 0.0

        d = np.linalg.norm(np.diff(P, axis=1), axis=0)
        val = float(np.var(d))

        self._cache_chroma[key] = val
        return val

    def _harmonic_template_fit(self, normalize=True, use_db=False, method="cosine"):
        key = f"harmonic_template_fit_{'norm' if normalize else 'raw'}_{'db' if use_db else 'lin'}_{method}"
        if key in self._cache_chroma:
            return self._cache_chroma[key]

        P = self._mean_chroma_profile(normalize=normalize, use_db=use_db)
        T = self._chroma_template()["templates"][:, :12]

        if method == "cosine":
            Pn = P/(np.linalg.norm(P) + EPS)
            Tn = T/(np.linalg.norm(T, axis=1, keepdims=True) + EPS)
            scores = Tn @ Pn
        elif method == "dot":
            scores = T @ P
        else:
            raise ValueError("method must be 'cosine' or 'dot'")

        best = float(np.max(scores)) if scores.size else 0.0
        result = {
            "best_score": best,
            "scores": scores,
            "best_idx": int(np.argmax(scores)) if scores.size else 0
        }

        self._cache_chroma[key] = result
        return result

    def _tonal_stability(self, normalize=True, use_db=False):
        key = f"tonal_stability_{'norm' if normalize else 'raw'}_{'db' if use_db else 'lin'}"
        if key in self._cache_chroma:
            return self._cache_chroma[key]

        smooth = self._chroma_smoothness(normalize=normalize, use_db=use_db, metric="l2")["smoothness"]
        clarity = self._tonal_clarity(normalize=normalize, use_db=use_db, method="cosine")["tonal_clarity"]

        val = float(np.clip(0.5*smooth + 0.5*clarity, 0.0, 1.0))

        self._cache_chroma[key] = val
        return val
    
    def _chroma_repetition(self, normalize=True, use_db=False):
        key = f"chroma_repetition_{'norm' if normalize else 'raw'}_{'db' if use_db else 'lin'}"
        if key in self._cache_chroma:
            return self._cache_chroma[key]

        ac = self._chroma_autocorrelation(lag_max=64,normalize=normalize, use_db=use_db)["acf_mean"]
        if ac.size < 2:
            self._cache_chroma[key] = 0.0
            return 0.0
        
        region = ac[1:]
        val = float(np.clip(np.max(region), 0.0, 1.0)) if region.size else 0.0

        self._cache_chroma[key] = val
        return val

    def _pitch_class_peakedness(self, normalize=True, use_db=False):
        key = f"pitch_class_peakedness_{'norm' if normalize else 'raw'}_{'db' if use_db else 'lin'}"
        if key in self._cache_chroma:
            return self._cache_chroma[key]

        P = self._mean_chroma_profile(normalize=normalize, use_db=use_db)
        if P.size == 0:
            self._cache_chroma[key] = 0.0
            return 0.0

        p = P/(np.sum(P) + EPS)
        val = float(np.clip(np.max(p)/(np.mean(p) + EPS), 0.0, 12.0)/12.0)

        self._cache_chroma[key] = val
        return val
    
    def _key_estimation(self, normalize=True, use_db=False, method="cosine"):
        """
        Estimate the key of the audio signal
        Returns:
            key_idx: index of estimated key in template_labels (0-23)
            tonic: pitch class of tonic (0-11)
            mode: "maj" or "min"
            score: cosine similarity score of best matching template
            scores: array of cosine similarity scores for all templates
        """
        key = f"key_estimation_{'norm' if normalize else 'raw'}_{'db' if use_db else 'lin'}_{method}"
        if key in self._cache_chroma:
            return self._cache_chroma[key]
        
        P = self._mean_chroma(normalize=normalize, use_db=use_db)
        T = self._chroma_template()

        if method == "cosine":
            Pn = P/(np.linalg.norm(P) + EPS)
            Tn = T/(np.linalg.norm(T, axis=1, keepdims=True) + EPS)
            scores = Tn @ Pn
        elif method == "dot":
            scores = T @ P
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        key_idx = int(np.argmax(scores))
        tonic = key_idx%12
        mode = "maj" if key_idx < 12 else "min"
        score = float(scores[key_idx])

        result = {
            "key_idx": key_idx,
            "tonic": tonic,
            "mode": mode,
            "score": score,
            "scores": scores
        }

        self._cache_chroma[key] = result
        return result
    
    def _energy_chroma(self, normalize=True, use_db=False):
        key = f"energy_chroma_{'norm' if normalize else 'raw'}_{'db' if use_db else 'lin'}"
        if key in self._cache_chroma:
            return self._cache_chroma[key]

        fit = self._harmonic_template_fit(normalize=normalize, use_db=use_db, method="dot")["best_score"]
        spread = self._chroma_spread(normalize=normalize, use_db=use_db)
        stability = self._tonal_stability(normalize=normalize, use_db=use_db)

        val = safe_clip01(0.35*fit + 0.35*stability + 0.3*(1.0 - np.mean(spread)))

        self._cache_chroma[key] = val
        return val

    def _speechiness_chroma(self, normalize=True, use_db=False):
        key = f"speechiness_chroma_{'norm' if normalize else 'raw'}_{'db' if use_db else 'lin'}"
        if key in self._cache_chroma:
            return self._cache_chroma[key]

        ent = self._chroma_entropy(normalize=normalize, use_db=use_db)
        flux = self._chroma_flux_mean(normalize=normalize, use_db=use_db)
        var = self._chroma_flux_variance(normalize=normalize, use_db=use_db)
        repetitive = self._chroma_repetition(normalize=normalize, use_db=use_db)

        val = safe_clip01(0.4*ent + 0.25*np.tanh(flux) + 0.20*np.tanh(var) + 0.15*(1.0 - repetitive))

        self._cache_chroma[key] = val
        return val

    def _acousticness_chroma(self, normalize=True, use_db=False):
        key = f"acousticness_chroma_{'norm' if normalize else 'raw'}_{'db' if use_db else 'lin'}"
        if key in self._cache_chroma:
            return self._cache_chroma[key]

        fit = self._harmonic_template_fit(normalize=normalize, use_db=use_db, method="dot")["best_score"]
        stability = self._tonal_stability(normalize=normalize, use_db=use_db)
        entropy = self._chroma_entropy(normalize=normalize, use_db=use_db)
        flux = self._chroma_flux_mean(normalize=normalize, use_db=use_db)

        val = safe_clip01(0.35*fit + 0.30*stability + 0.20*(1.0 - entropy) + 0.15*(1.0 - np.tanh(flux)))

        self._cache_chroma[key] = val
        return val
    
    def _danceability_chroma(self, normalize=True, use_db=False):
        key = f"danceability_chroma_{'norm' if normalize else 'raw'}_{'db' if use_db else 'lin'}"
        if key in self._cache_chroma:
            return self._cache_chroma[key]

        rep = self._chroma_repetition(normalize=normalize, use_db=use_db)
        smooth = self._chroma_smoothness(normalize=normalize, use_db=use_db, metric="l2")["smoothness"]
        stability = self._tonal_stability(normalize=normalize, use_db=use_db)
        flux = self._chroma_flux_mean(normalize=normalize, use_db=use_db)

        val = safe_clip01(0.35*rep + 0.25*smooth + 0.20*stability + 0.20*(1.0 - np.tanh(flux)))

        self._cache_chroma[key] = val
        return val
    
    def _valence_chroma(self, normalize=True, use_db=False):
        key = f"valence_chroma_{'norm' if normalize else 'raw'}_{'db' if use_db else 'lin'}"
        if key in self._cache_chroma:
            return self._cache_chroma[key]
        
        prof = self._mean_chroma_profile(normalize=normalize, use_db=use_db)
        if prof.size == 0:
            self._cache_chroma[key] = 0.5
            return 0.5
        
        major_template = np.array([
            [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]
        ])
        major = self._mode_classification(normalize=normalize, use_db=use_db, method="cosine")["mode"] == "maj"
        key_res = self._key_estimation(normalize=normalize, use_db=use_db, method="cosine")
        tonic = key_res["tonic"]
        templates = self._chroma_template()["templates"]
        maj_score = float(key_res["scores"][tonic])
        min_score = float(key_res["scores"][tonic + 12])
        clarity = self._tonal_clarity(normalize=normalize, use_db=use_db, method="cosine")["tonal_clarity"]
        fit = self._harmonic_template_fit(normalize=normalize, use_db=use_db, method="cosine")["best_score"]
        bright = float((prof[(tonic + 4)%12] + prof[(tonic + 7)%12] + prof[(tonic + 11)%12])/(np.sum(prof) + EPS))
        delta = maj_score - min_score

        val = safe_clip01(0.40*(0.5 + 0.5*np.tanh(delta)) + 0.25*clarity + 0.20*fit + 0.15*bright)

        self._cache_chroma[key] = val
        return val
    
    def _tempo_chroma(self, normalize=True, use_db=False):
        key = f"tempo_chroma_{'norm' if normalize else 'raw'}_{'db' if use_db else 'lin'}"
        if key in self._cache_chroma:
            return self._cache_chroma[key]
        
        rep = self._chroma_repetition(normalize=normalize, use_db=use_db)
        flux = self._chroma_flux_mean(normalize=normalize, use_db=use_db)
        var = self._chroma_flux_variance(normalize=normalize, use_db=use_db)

        tempo_proxy = float(np.clip(40.0 + 200.0*(0.45*rep + 0.35*np.tanh(flux) + 0.20*np.tanh(var)), 40.0, 240.0))

        self._cache_chroma[key] = tempo_proxy
        return tempo_proxy
    
    def _instrumentalness_chroma(self, normalize=True, use_db=False):
        key = f"instrumentalness_chroma_{'norm' if normalize else 'raw'}_{'db' if use_db else 'lin'}"
        if key in self._cache_chroma:
            return self._cache_chroma[key]

        fit = self._harmonic_template_fit(normalize=normalize, use_db=use_db, method="dot")["best_score"]
        stability = self._tonal_stability(normalize=normalize, use_db=use_db)
        entropy = self._chroma_entropy(normalize=normalize, use_db=use_db)
        tonal_focus = self._pitch_class_peakedness(normalize=normalize, use_db=use_db)

        val = safe_clip01(0.35*fit + 0.25*stability + 0.20*tonal_focus + 0.20*(1.0 - entropy))

        self._cache_chroma[key] = val
        return val

    def _time_signature_chroma(self, normalize=True, use_db=False):
        key = f"time_signature_chroma_{'norm' if normalize else 'raw'}_{'db' if use_db else 'lin'}"
        if key in self._cache_chroma:
            return self._cache_chroma[key]

        rep = self._chroma_repetition(normalize=normalize, use_db=use_db)
        flux = self._chroma_flux_mean(normalize=normalize, use_db=use_db)
        smooth = self._chroma_smoothness(normalize=normalize, use_db=use_db, metric="l2")["smoothness"]

        meter_score = safe_clip01(0.5*rep + 0.3*smooth + 0.2*(1.0 - np.tanh(flux)))
        ts = 3 if meter_score > 0.55 else 4

        self._cache_chroma[key] = ts
        return ts

    def _mode_classification(self, normalize=True, use_db=False, method="cosine"):
        """
        Classify the mode (major vs minor) of the audio signal
        Returns:
            mode string: "maj" or "min"
            score_major: float score for best matching major template
            score_minor: float score for best matching minor template
            delta_score: score_major - score_minor, higher = more major-like, lower = more minor-like
        """
        key = f"mode_classification_{'norm' if normalize else 'raw'}_{'db' if use_db else 'lin'}_{method}"
        if key in self._cache_chroma:
            return self._cache_chroma[key]
        
        result = self._key_estimation(normalize=normalize, use_db=use_db, method=method)
        tonic = result["tonic"]
        score_major = result["scores"][tonic]
        score_minor = result["scores"][tonic + 12]
        delta_score = score_major - score_minor

        mode = "maj" if delta_score >= 0 else "min"

        result = {
            "mode": mode,
            "score_major": score_major,
            "score_minor": score_minor,
            "delta_score": delta_score
        }

        self._cache_chroma[key] = result
        return result
    
    def spotify_audio_features(self, normalize=True, use_db=False, method="cosine"):
        key_res = self._key_estimation(normalize=normalize, use_db=use_db, method=method)
        mode_res = self._mode_classification(normalize=normalize, use_db=use_db, method=method)

        return {
            "energy": self._energy_chroma(normalize=normalize, use_db=use_db),
            "speechiness": self._speechiness_chroma(normalize=normalize, use_db=use_db),
            "acousticness": self._acousticness_chroma(normalize=normalize, use_db=use_db),
            "danceability": self._danceability_chroma(normalize=normalize, use_db=use_db),
            "valence": self._valence_chroma(normalize=normalize, use_db=use_db),
            "tempo": self._tempo_chroma(normalize=normalize, use_db=use_db),
            "instrumentalness": self._instrumentalness_chroma(normalize=normalize, use_db=use_db),
            "key": key_res["key"],
            "mode": mode_res["mode"],
            "time_signature": self._time_signature_chroma(normalize=normalize, use_db=use_db),
        }

class TempogramFeatures:
    def __init__(self, sig, center=True):
        self.y = sig.y
        self.sr = sig.sr
        self.H = sig.H
        self.N = sig.N
        self.center = center

        self._cache_tempogram = {}

        # Frame Rate
        self.frame_rate = float(self.sr)/float(self.H)

        # Time vector for frames
        self.times = None # Will not be set when onset strength is computed

    def _get_filtered_spectrum(self, S, bpm, bpm_min=30, bpm_max=300):
        """
        Internal helper to filter the tempogram spectrum to a reasonable BPM range and apply logarithmic scaling
        """
        mask = (bpm >= bpm_min) & (bpm <= bpm_max) & np.isfinite(S)
        return S[mask], bpm[mask]

    # Onset Strength Envelope
    def _onset_strength(self, max_size=1, detrend=False, aggregate=np.mean, smooth=False, smooth_width=5):
        key = f"onset_strength_max_{max_size}_detrend_{detrend}_{aggregate.__name__}_{smooth}_{smooth_width}"
        if key in self._cache_tempogram:
            return self._cache_tempogram[key]
        
        onset_env = librosa.onset.onset_strength(
            y=self.y,
            sr=self.sr,
            hop_length=self.H,
            n_fft=self.N,
            aggregate=aggregate
        )

        if max_size > 1:
            onset_env = maximum_filter(onset_env, size=max_size, mode="constant")

        if detrend and onset_env.size > 1:
            x = np.arange(onset_env.size, dtype=float)
            p = np.polyfit(x, onset_env, deg=1)
            onset_env = onset_env - np.polyval(p, x)

        onset_env = np.maximum(onset_env, 0.0)

        if smooth and smooth_width > 1:
            kernel = np.ones(int(smooth_width), dtype=float)/float(smooth_width)
            onset_env = np.convolve(onset_env, kernel, mode='same')
            onset_env = np.maximum(onset_env, 0.0)

        times = librosa.frames_to_time(np.arange(len(onset_env)), sr=self.sr, hop_length=self.H)

        result = {
            'onset_env': onset_env,
            'times': times
        }

        self._cache_tempogram[key] = result
        return result
    
    def _onset_energy(self, normalize=False, max_size=1, detrend=False):
        """
        Total or average onset energy 
        """
        key = f"onset_energy_{normalize}_{max_size}_{detrend}"
        if key in self._cache_tempogram:
            return self._cache_tempogram[key]
        
        onset = self._onset_strength(max_size=max_size, detrend=detrend)['onset_env']
        energy = float(np.mean(onset) if normalize else np.sum(onset))

        self._cache_tempogram[key] = energy
        return energy
    
    def _transient_curve(self, smooth=False, smooth_width=5, normalize=True):
        """
        Transient curve from first differences of the onset envelope
        """
        key = f"transient_curve_{smooth}_{smooth_width}_{normalize}"
        if key in self._cache_tempogram:
            return self._cache_tempogram[key]
        
        onset = self._onset_strength()['onset_env']

        if onset.size < 2:
            result = {
                "curve": np.array([], dtype=float),
                "peak_count": 0,
                "mean_slope": 0.0
            }

            self._cache_tempogram[key] = result
            return result
        
        curve = np.maximum(0.0, np.diff(onset))

        if smooth and smooth_width > 1:
            kernel = np.ones(smooth_width, dtype=float)/float(smooth_width)
            curve = np.convolve(curve, kernel, mode="same")

        if normalize:
            curve = curve/(np.max(curve) + EPS)

        peaks = np.where((curve[1:-1] > curve[:-2]) & (curve[1:-1] >= curve[2:]))[0] + 1

        result = {
            "curve": curve,
            "peak_count": int(peaks.size),
            "mean_slope": float(np.mean(curve)) if curve.size > 0 else 0.0,
            "peaks": peaks 
        }

        self._cache_tempogram[key] = result
        return result 
    
    def _envelope_periodicity(self, lag_max=None, normalize=True, method="autocorr"):
        """
        Periodicity of the onset envelope via autocorrelation or Fourier spectrum
        """
        key = f"envelope_periodicity_{lag_max}_{normalize}_{method}"
        if key in self._cache_tempogram:
            return self._cache_tempogram[key]
        
        onset = self._onset_strength()["onset_env"]
        if onset.size < 3:
            result = {
                "periodicity": 0.0,
                "acf": np.array([], dtype=float),
                "bpm": np.array([], dtype=float),
                "scores": np.array([], dtype=float)
            }
            self._cache_tempogram[key] = result
            return result
        
        x = onset - np.mean(onset)
        if lag_max is None:
            lag_max = min(len(x) - 1, int(self.frame_rate*4.0))

        lag_max = int(max(1, min(lag_max, len(x) - 1)))

        if method == "autocorr":
            acf = np.correlate(x, x, mode="full")[len(x) - 1:len(x) - 1 + lag_max + 1]
            if normalize:
                acf = acf/(acf[0] + EPS)

            acf_pos = acf[1:] if acf.size > 1 else acf
            best_lag = int(np.argmax(acf_pos) + 1) if acf_pos.size > 0 else 0
            periodicity = float(np.max(acf_pos)) if acf_pos.size > 0 else 0.0

            bpm = np.zeros_like(acf, dtype=float)
            bpm[1:] = 60*self.frame_rate/np.arange(1, len(acf))
            result = {
                "periodicity": safe_clip01(periodicity),
                "acf": acf,
                "bpm": bpm,
                "best_lag": best_lag
            }
        elif method == "fourier":
            fft = np.fft.rfft(x*np.hanning(len(x)), n=len(x))
            scores = np.abs(fft)
            freqs = np.fft.rfftfreq(len(x), d=1.0/self.frame_rate)
            bpm = 60.0*freqs

            if scores.size > 1:
                best_idx = int(np.argmax(scores[1:]) + 1)
                periodicity = float(scores[best_idx]/(np.sum(scores) + EPS))
            else:
                best_idx = 0
                periodicity = 0.0

            result = {
                "periodicity": safe_clip01(periodicity),
                "scores": scores,
                "bpm": bpm,
                "best_idx": best_idx 
            }
        else:
            raise ValueError("method must be 'autocorr' or 'fourier'")
        
        self._cache_tempogram[key] = result
        return result
    
    # Autocorrelation Tempogram
    # def _tempo_autocorr(self, win_length=None, center=None, norm='l1'):
    #     if win_length is None:
    #         win_length = self.N
    #     if center is None:
    #         center = self.center

    #     key = f"tempogram_autocorr_win_{win_length}_center_{center}_norm_{norm}"
    #     if key in self._cache_tempogram:
    #         result = self._cache_tempogram[key]
    #         return result
        
    #     # Get onset strength
    #     onset_result = self._onset_strength()
    #     onset_env = onset_result['onset_env']

    #     n_frames = len(onset_env)

    #     # Compute number of windows
    #     if center:
    #         n_windows = n_frames
    #     else:
    #         n_windows = n_frames - win_length + 1

    #     if n_windows <= 0:
    #         # Not enough frames
    #         tempogram = np.zeros((win_length, 0), dtype=float)
    #         bpm = np.zeros(win_length, dtype=float)
    #         times = np.array([], dtype=float)

    #         result = {
    #             'tempogram': tempogram,
    #             'bpm': bpm,
    #             'times': times
    #         }
    #         self._cache_tempogram[key] = result
    #         return result
        
    #     # Initialize tempogram
    #     tempogram = np.zeros((win_length, n_windows), dtype=float)

    #     # Compute autocorrelation for each window
    #     for i in range(n_windows):
    #         if center:
    #             # Center window around frame i
    #             start = max(0, i - win_length//2)
    #             end = min(n_frames, i + win_length//2 + 1)
    #         else:
    #             # Sliding window
    #             start = i
    #             end = i + win_length

    #         # Extract window
    #         window = onset_env[start:end]

    #         # Compute autocorrelation
    #         if window.size < 2:
    #             continue

    #         window = window - np.mean(window)
    #         acf = np.correlate(window, window, mode='full')[len(window) - 1:]

    #         if norm == 'l1':
    #             acf = acf/(np.sum(np.abs(acf)) + EPS)
    #         elif norm == 'l2':
    #             acf = acf/(np.linalg.norm(acf) + EPS)
    #         elif norm is None:
    #             pass
    #         else:
    #             acf = acf/(acf[0] + EPS)

    #         tempogram[:min(win_length, len(acf)), i] = acf[:win_length]

    #     # Compute BPM axis
    #     # BPM = 60*frame_rate/lag
    #     lags = np.arange(win_length)
    #     bpm = np.zeros(win_length, dtype=float)
    #     bpm[1:] = 60.0*self.frame_rate/lags[1:]

    #     if center:
    #         window_times = onset_result['times'][:n_windows]
    #     else:
    #         window_centers = np.arange(n_windows) + win_length//2
    #         window_times = librosa.frames_to_time(window_centers, sr=self.sr, hop_length=self.H)

    #     result = {
    #         'tempogram': tempogram,
    #         'bpm': bpm,
    #         'times': window_times
    #     }

    #     self._cache_tempogram[key] = result
    #     return result
    
    def _tempogram_autocorr(self, win_length=None, center=None, norm_sum=True):
        if win_length is None:
            win_length = self.N
        if center is None:
            center = self.center

        key = f"tempogram_autocorr_win_{win_length}_center_{center}_norm_sum_{norm_sum}"
        if key in self._cache_tempogram:
            return self._cache_tempogram[key]
        
        onset = self._onset_strength()
        onset_env = onset['onset_env']
        times = onset['times']
        n_frames = len(onset_env)

        if center:
            n_windows = n_frames
        else:
            n_windows = n_frames - win_length + 1

        if n_windows <= 0:
            tempogram = np.zeros((win_length, 0), dtype=float)
            bpm = np.zeros(win_length, dtype=float)
            windows_times = np.array([], dtype=float)

            self._cache_tempogram[key] = {
                'tempogram': tempogram,
                'bpm': bpm,
                'times': windows_times
            }
            return result
        
        tempogram = np.zeros((win_length, n_windows), dtype=float)

        for n in range(n_windows):
            if center:
                start = max(0, n - win_length//2)
                end = min(n_frames, n + win_length//2 + 1)
            else:
                start = n
                end = n + win_length
            
            x = onset_env[start:end]
            if x.size < 2:
                continue

            x = x - np.mean(x)
            acf = np.correlate(x, x, mode='full')[len(x) - 1:]

            if norm_sum:
                acf = acf/(np.sum(acf) + EPS)
            else:
                acf = acf/(acf[0] + EPS)

            tempogram[:min(win_length, len(acf)), n] = acf[:win_length]

        lags = np.arange(win_length, dtype=float)
        bpm = np.zeros(win_length, dtype=float)
        bpm[1:] = 60.0*self.frame_rate/lags[1:]

        if center:
            windows_times = times[:n_windows]
        else:
            window_centers = np.arange(n_windows) + win_length//2
            windows_times = librosa.frames_to_time(window_centers, sr=self.sr, hop_length=self.H)

        result = {
            'tempogram': tempogram,
            'bpm': bpm,
            'times': windows_times
        }
        
        self._cache_tempogram[key] = result
        return result
        
    def _global_bpm(self, bpm_min=40.0, bpm_max=240.0, norm_sum=True):
        key = f"global_bpm_{bpm_min}_{bpm_max}_{norm_sum}"
        if key in self._cache_tempogram:
            return self._cache_tempogram[key]
        
        ac = self._tempogram_autocorr(norm_sum=norm_sum)
        tg = ac['tempogram']
        bpm = ac['bpm']

        if tg.size == 0:
            result = {
                "bpm": 0.0,
                "lag": 0,
                "strength": 0.0,
                "bpm_axis": bpm
            }
            self._cache_tempogram[key] = result
            return result
        
        mask = (bpm >= bpm_min) & (bpm <= bpm_max) & np.isfinite(bpm)
        if not np.any(mask):
            result = {
                "bpm": 0.0,
                "lag": 0,
                "strength": 0.0,
                "bpm_axis": bpm
            }
            self._cache_tempogram[key] = result
            return result
        
        global_ac = np.mean(tg, axis=1)

        search = global_ac[mask]

        idxs = np.where(mask)[0]
        best_rel = int(np.argmax(search))
        best_idx = int(idxs[best_rel])

        detected_bpm = float(bpm[best_idx])
        detected_strength = float(global_ac[best_idx])

        if detected_bpm < 80:
            for multiplier in [2, 3, 4]:
                candidate_bpm = detected_bpm*multiplier

                # Only consider if in valid range
                if candidate_bpm > bpm_max:
                    break

                # Find closest bin
                candidate_idx = np.argmin(np.abs(bpm - candidate_bpm))
                candidate_strength = global_ac[candidate_idx]

                # If octave multiple is at least 70% as strong, prefer it
                if candidate_strength > 0.7*detected_strength:
                    detected_bpm = float(bpm[candidate_idx])
                    best_idx = int(candidate_idx)
                    detected_strength = candidate_strength
        elif detected_bpm > 200:
            for divisor in [2, 3, 4]:
                candidate_bpm = detected_bpm/divisor

                if candidate_bpm < bpm_min:
                    break

                candidate_idx = np.argmin(np.abs(bpm - candidate_bpm))
                candidate_strength = global_ac[candidate_idx]

                # Prefer sub-octave if it's reasonably strong
                if candidate_strength > detected_strength:
                    detected_bpm = float(bpm[candidate_idx])
                    best_idx = int(candidate_idx)
                    detected_strength = candidate_strength
                    break

        result = {
            "bpm": float(bpm[best_idx]),
            "lag": int(best_idx),
            "strength": float(global_ac[best_idx]),
            "bpm_axis": bpm,
            "global_ac":global_ac
        }

        self._cache_tempogram[key] = result
        return result
    
    def _local_bpm_curve(self, bpm_min=40.0, bpm_max=240.0, norm_sum=True):
        key = f"local_bpm_curve_{bpm_min}_{bpm_max}_{norm_sum}"
        if key in self._cache_tempogram:
            return self._cache_tempogram[key]
        
        ac = self._tempogram_autocorr(norm_sum=norm_sum)
        tg = ac['tempogram']
        bpm_axis = ac['bpm']

        if tg.size == 0:
            result = {
                "bpm_curve": np.array([], dtype=float),
                "strength_curve": np.array([], dtype=float)
            }

            self._cache_tempogram[key] = result
            return result
        
        mask = (bpm_axis >= bpm_min) & (bpm_axis <= bpm_max) & np.isfinite(bpm_axis)
        if not np.any(mask):
            result = {
                "bpm_curve": np.array([], dtype=float),
                "strength_curve": np.array([], dtype=float)
            }

            self._cache_tempogram[key] = result
            return result
        
        idxs = np.where(mask)[0]
        local_slice = tg[mask, :]
        best_rel = np.argmax(local_slice, axis=0)
        best_idx = idxs[best_rel]
        bpm_curve = bpm_axis[best_idx]
        strength_curve = local_slice[best_rel, np.arange(local_slice.shape[1])]

        result = {
            "bpm_curve": bpm_curve.astype(float),
            "strength_curve": strength_curve.astype(float),
            "times": ac["times"]
        }

        self._cache_tempogram[key] = result
        return result
    
    def _pulse_clarity(self, bpm_min=40.0, bpm_max=240.0, norm_sum=True):
        key = f"pulse_clarity_{bpm_min}_{bpm_max}_{norm_sum}"
        if key in self._cache_tempogram:
            return self._cache_tempogram[key]
        
        ac = self._tempogram_autocorr(norm_sum=norm_sum)
        tg = ac["tempogram"]
        bpm_axis = ac["bpm"]

        if tg.size == 0:
            result = {
                "clarity": 0.0,
                "best_peak": 0.0,
                "runner-up": 0.0
            }

            self._cache_tempogram[key] = result
            return result
        
        mask = (bpm_axis >= bpm_min) & (bpm_axis <= bpm_max) & np.isfinite(bpm_axis)

        if not np.any(mask):
            result = {
                "clarity": 0.0,
                "best_peak": 0.0,
                "runner-up": 0.0
            }

            self._cache_tempogram[key] = result
            return result

        vals = np.mean(tg[mask, :], axis=1)

        if vals.size == 0:
            clarity = 0.0
            best = 0.0
            second = 0.0
        elif vals.size == 1:
            vals_normalized = vals / (np.max(vals) + EPS)
            best = float(vals_normalized[0])
            second = 0.0
            clarity = 1.0  # Or a specific logic for single-peak signals
        else:
            vals_normalized = vals / (np.max(vals) + EPS)
            s = np.sort(vals_normalized)
            best = float(s[-1])
            second = float(s[-2])
            clarity = (best - second) / (best + EPS)

        result = {
            "clarity": safe_clip01(clarity),
            "best_peak": best,
            "runner-up": second
        }

        self._cache_tempogram[key] = result
        return result
    
    def _beat_periodicity_strength(self, bpm_min=40.0, bpm_max=240.0, norm_sum=True):
        key = f"beat_periodicity_strength_{bpm_min}_{bpm_max}_{norm_sum}"
        if key in self._cache_tempogram:
            return self._cache_tempogram[key]
        
        g = self._global_bpm(bpm_min=bpm_min, bpm_max=bpm_max, norm_sum=norm_sum)
        strength = safe_clip01(g["strength"])

        result = {
            "strength": strength,
            "bpm": g["bpm"],
            "lag": g["lag"]
        }

        self._cache_tempogram[key] = result
        return result
    
    def _tempo_stability_index(self, bpm_min=40.0, bpm_max=240.0, norm_sum=True):
        key = f"tempo_stability_index_{bpm_min}_{bpm_max}_{norm_sum}"
        if key in self._cache_tempogram:
            return self._cache_tempogram[key]
        
        curve = self._local_bpm_curve(bpm_min=bpm_min, bpm_max=bpm_max, norm_sum=norm_sum)["bpm_curve"]
        if curve.size < 2:
            result = {
                "stability": 1.0,
                "mean_bpm": 0.0,
                "std_bpm": 0.0
            }

            self._cache_tempogram[key] = result
            return result
        
        mean_bpm = float(np.mean(curve))
        std_bpm = float(np.std(curve))
        stability = 1.0 - (std_bpm/(mean_bpm + EPS))

        result = {
            "stability": safe_clip01(stability),
            "mean_bpm": mean_bpm,
            "std_bpm": std_bpm
        }

        self._cache_tempogram[key] = result
        return result
    
    def _tempo_variation_curve(self, bpm_min=40.0, bpm_max=240.0, norm_sum=True):
        key = f"tempo_variation_curve_{bpm_min}_{bpm_max}_{norm_sum}"
        if key in self._cache_tempogram:
            return self._cache_tempogram[key]
        
        g = self._global_bpm(bpm_min=bpm_min, bpm_max=bpm_max, norm_sum=norm_sum)
        l = self._local_bpm_curve(bpm_min=bpm_min, bpm_max=bpm_max, norm_sum=norm_sum)

        if l["bpm_curve"].size == 0:
            result = {
                "variation": np.array([], dtype=float),
                "abs_variation": np.array([], dtype=float)
            }

            self._cache_tempogram[key] = result
            return result
        
        variation = l["bpm_curve"] - g["bpm"]
        result = {
            "variation": variation.astype(float),
            "abs_variation": np.abs(variation).astype(float),
            "times": l["times"]
        }

        self._cache_tempogram[key] = result
        return result
    
    def _beat_fluctuation_rate(self, bpm_min=40.0, bpm_max=240.0, norm_sum=True):
        key = f"beat_fluctuation_rate_{bpm_min}_{bpm_max}_{norm_sum}"
        if key in self._cache_tempogram:
            return self._cache_tempogram[key]
        
        curve = self._local_bpm_curve(bpm_min=bpm_min, bpm_max=bpm_max, norm_sum=norm_sum)["bpm_curve"]
        if curve.size < 2:
            self._cache_tempogram[key] = 0.0
            return 0.0
        
        rate = float(np.mean(np.abs(np.diff(curve))))

        self._cache_tempogram[key] = rate
        return rate
    
    def _multi_periodic_structure(self, bpm_min=40.0, bpm_max=240.0, norm_sum=True):
        key = f"multi_periodic_structure_{bpm_min}_{bpm_max}_{norm_sum}"
        if key in self._cache_tempogram:
            return self._cache_tempogram[key]
        
        ac = self._tempogram_autocorr(norm_sum=norm_sum)
        tg = ac["tempogram"]
        bpm_axis = ac["bpm"]

        if tg.size == 0:
            result = {
                "score": 0.0,
                "primary_bpm": 0.0,
                "half_bpm": 0.0,
                "double_bpm": 0.0
            }

            self._cache_tempogram[key] = result
            return result
        
        global_ac = np.mean(tg, axis=1)
        mask = (bpm_axis >= bpm_min) & (bpm_axis <= bpm_max) & np.isfinite(bpm_axis)
        idxs = np.where(mask)[0]
        if idxs.size == 0:
            result = {
                "score": 0.0,
                "primary_bpm": 0.0,
                "half_bpm": 0.0,
                "double_bpm": 0.0
            }

            self._cache_tempogram[key] = result
            return result
        
        best_idx = int(idxs[np.argmax(global_ac[mask])])
        primary_bpm = float(bpm_axis[best_idx])
        half_bpm = primary_bpm/2.0
        double_bpm = primary_bpm*2.0

        def nearest_val(target):
            j = int(np.argmin(np.abs(bpm_axis - target)))
            return float(global_ac[j]), float(bpm_axis[j])
        
        half_score, half_bpm_near = nearest_val(half_bpm)
        double_score, double_bpm_near = nearest_val(double_bpm)
        primary_score = float(global_ac[best_idx])

        score = (half_score + double_score)/(primary_score + EPS)

        result = {
            "score": float(np.clip(score, 0.0, 2.0)),
            "primary_bpm": primary_bpm,
            "half_bpm": half_bpm_near,
            "double_bpm": double_bpm_near,
            "primary_score": primary_score,
            "half_score": half_score,
            "double_score": double_score 
        }

        self._cache_tempogram[key] = result
        return result
    
    def _swing_ratio(self, bpm_min=40.0, bpm_max=240.0, norm_sum=True):
        key = f"swing_ratio_{bpm_min}_{bpm_max}_{norm_sum}"
        if key in self._cache_tempogram:
            return self._cache_tempogram[key]
        
        ac = self._tempogram_autocorr(norm_sum=norm_sum)
        tg = ac["tempogram"]
        bpm_axis = ac["bpm"]

        if tg.size == 0:
            result = {
                "ratio": 1.0,
                "symmetry": 1.0
            }

            self._cache_tempogram[key] = result
            return result
        
        global_ac = np.mean(tg, axis=1)
        mask = (bpm_axis >= bpm_min) & (bpm_axis <= bpm_max) & np.isfinite(bpm_axis)
        if not np.any(mask):
            result = {
                "ratio": 1.0,
                "symmetry": 1.0
            }

            self._cache_tempogram[key] = result
            return result
        
        idxs = np.where(mask)[0]
        best_idx = int(idxs[np.argmax(global_ac[mask])])

        half_idx = int(np.argmin(np.abs(bpm_axis - bpm_axis[best_idx]/2.0)))
        double_idx = int(np.argmin(np.abs(bpm_axis - bpm_axis[best_idx]*2.0)))

        half_score = float(global_ac[half_idx])
        double_score = float(global_ac[double_idx])

        ratio = (double_score + EPS)/(half_score + EPS)
        symmetry = 1.0 - abs(np.log(ratio))/np.log(2.0)
        symmetry = safe_clip01(symmetry)

        result = {
            "ratio": float(ratio),
            "symmetry": symmetry,
            "primary_bpm": float(bpm_axis[best_idx]),
            "half_bpm": float(bpm_axis[half_idx]),
            "double_bpm": float(bpm_axis[double_idx])
        }

        self._cache_tempogram[key] = result
        return result 
    
    # Fourier Tempogram
    # def _tempogram_fourier(self, win_length=None, center=None, window='hann'):
    #     if win_length is None:
    #         win_length = self.N
    #     if center is None:
    #         center = self.center

    #     key = f"tempogram_fourier_win_{win_length}_center_{center}_window_{window}"
    #     if key in self._cache_tempogram:
    #         result = self._cache_tempogram[key]
    #         return result
        
    #     # Get onset strength
    #     onset_result = self._onset_strength()
    #     onset_env = onset_result['onset_env']

    #     onset_env = onset_env - np.mean(onset_env)

    #     n_frames = len(onset_env)

    #     # Create window function
    #     if window == 'hann':
    #         win_func = np.hanning(win_length)
    #     elif window == 'hamming':
    #         win_func = np.hamming(win_length)
    #     elif window == 'blackman':
    #         win_func = np.blackman(win_length)
    #     else:
    #         win_func = np.ones(win_length)

    #     # Computer number of windows
    #     if center:
    #         n_windows = n_frames
    #     else:
    #         n_windows = n_frames - win_length + 1

    #     if n_windows <= 0:
    #         # Not enough frames
    #         n_bins = win_length//2 + 1
    #         tempogram = np.zeros((n_bins, 0), dtype=float)
    #         bpm = np.zeros(n_bins, dtype=float)
    #         times = np.array([], dtype=float)

    #         result = {
    #             'tempogram': tempogram,
    #             'bpm': bpm,
    #             'times': times
    #         }

    #         self._cache_tempogram[key] = result
    #         return result
        
    #     # Initialize tempogram
    #     n_bins = win_length//2 + 1 # Positive frequencies only
    #     tempogram = np.zeros((n_bins, n_windows), dtype=float)

    #     # Compute DFT for each window
    #     for i in range(n_windows):
    #         if center:
    #             # Center window around frame i
    #             start = max(0, i - win_length//2)
    #             end = min(n_frames, i + win_length//2 + 1)

    #             # Extract window
    #             window_data = onset_env[start:end]

    #             # Pad if necessary
    #             if window_data.size < win_length:
    #                 pad_left = (win_length - window_data.size)//2
    #                 pad_right = win_length - window_data.size - pad_left
    #                 window_data = np.pad(window_data, (pad_left, pad_right), mode='constant')
    #         else:
    #             # Sliding window
    #             start = i
    #             end = i + win_length
    #             window_data = onset_env[start:end]

    #         window_data = window_data - np.mean(window_data)

    #         # Apply window function
    #         windowed = window_data*win_func

    #         # Compute FFT
    #         fft = np.fft.rfft(windowed, n=win_length)

    #         # Store magnitude
    #         tempogram[:, i] = np.abs(fft)

    #     # Compute BPM axis
    #     # Frequency bins
    #     freqs = np.fft.rfftfreq(win_length, d=1.0/self.frame_rate)

    #     # Convert to BPM (beats per minute)
    #     bpm = 60.0*freqs

    #     # Compute window center times
    #     if center:
    #         window_times = onset_result["times"][:n_windows]
    #     else:
    #         window_centers = np.arange(n_windows) + win_length//2
    #         window_times = librosa.frames_to_time(
    #             window_centers,
    #             sr=self.sr,
    #             hop_length=self.H
    #         )

    #     result = {
    #         'tempogram': tempogram,
    #         'bpm': bpm,
    #         'times': window_times
    #     }

    #     self._cache_tempogram[key] = result
    #     return result

    def _tempogram_fourier(self, win_length=None, center=None, window='hann', bpm_min=30.0, bpm_max=300.0):
        if win_length is None:
            win_length = self.N
        if center is None:
            center = self.center
        
        key= f"tempogram_fourier_win_{win_length}_center_{center}_window_{window}_{bpm_min}_{bpm_max}"
        if key in self._cache_tempogram:
            return self._cache_tempogram[key]
        
        onset = self._onset_strength()
        onset_env = onset['onset_env']
        times = onset['times']
        n_frames = len(onset_env)

        if window == "hann":
            win_func = np.hanning(win_length)
        elif window == "hamming":
            win_func = np.hamming(win_length)
        elif window == "blackman":
            win_func = np.blackman(win_length)
        else:
            win_func = np.ones(win_length, dtype=float)

        if center:
            n_windows = n_frames
        else:
            n_windows = n_frames - win_length + 1

        if n_windows <= 0:
            n_bins = win_length//2 + 1
            tempogram = np.zeros((n_bins, 0), dtype=float)
            bpm = np.zeros(n_bins, dtype=float)
            window_times = np.array([], dtype=float)

            result = {
                'tempogram': tempogram,
                'bpm': bpm,
                'times': window_times
            }
            self._cache_tempogram[key] = result
            return result
        
        tempogram = np.zeros((win_length//2 + 1, n_windows), dtype=float)

        for n in range(n_windows):
            if center:
                start = max(0, n - win_length//2)
                end = min(n_frames, n + win_length//2 + 1)
                x = onset_env[start:end]
                if x.size < win_length:
                    pad_left = (win_length - x.size)//2
                    pad_right = win_length - x.size - pad_left
                    x = np.pad(x, (pad_left, pad_right), mode='constant')
            else:
                x = onset_env[n:n + win_length]

            if x.size < win_length:
                continue

            x = x - np.mean(x)
            X = np.fft.rfft(x*win_func, n=win_length)
            tempogram[:, n] = np.abs(X)
            
        freqs = np.fft.rfftfreq(win_length, d=1.0/self.frame_rate)
        bpm = 60.0*freqs

        if center:
            window_times = times[:n_windows]
        else:
            window_centers = np.arange(n_windows) + win_length//2
            window_times = librosa.frames_to_time(window_centers, sr=self.sr, hop_length=self.H)

        result = {
            'tempogram': tempogram,
            'bpm': bpm,
            'times': window_times
        }

        self._cache_tempogram[key] = result
        return result
    
    def _tempo_spectrum(self, win_length=None, center=None, window="hann", average="mean"):
        key = f"tempo_spectrum_{win_length}_{center}_{window}_{average}"
        if key in self._cache_tempogram:
            return self._cache_tempogram[key]
        
        ft = self._tempogram_fourier(win_length=win_length, center=center, window=window)
        T = ft["tempogram"]
        bpm = ft["bpm"]

        if T.size == 0:
            result = {
                "spectrum": np.array([], dtype=float),
                "bpm": bpm
            }

            self._cache_tempogram[key] = result
            return result
        
        if average == "mean":
            spectrum = np.mean(T, axis=1)
        elif average == "median":
            spectrum = np.median(T, axis=1)
        else:
            raise ValueError("average must be 'mean' or 'median'")
        
        result = {
            "spectrum": spectrum.astype(float),
            "bpm": bpm
        }

        self._cache_tempogram[key] = result
        return result
    
    def _spectral_energy_at_tempo(self, tempo_bpm=None, win_length=None, center=None, window="hann"):
        key = f"spectral_energy_at_tempo_{tempo_bpm}_{win_length}_{center}_{window}"
        if key in self._cache_tempogram:
            return self._cache_tempogram[key]
        
        spec = self._tempo_spectrum(win_length=win_length, center=center, window=window)
        S = spec["spectrum"]
        bpm = spec["bpm"]

        if S.size == 0:
            result = {
                "energy": 0.0,
                "bpm": bpm,
                "spectrum": S
            }

            self._cache_tempogram[key] = result
            return result

        if tempo_bpm is None:
            result = {
                "energy": S,
                "bpm": bpm,
                "spectrum": S
            }

            self._cache_tempogram[key] = result
            return result
        
        idx = int(np.argmin(np.abs(bpm - tempo_bpm)))
        energy = float(S[idx])

        result = {
            "energy": energy,
            "bpm": float(bpm[idx]),
            "spectrum": S 
        }

        self._cache_tempogram[key] = result
        return result
    
    def _dominant_tempo_energy(self, win_length=None, center=None, window="hann", top_k=5, bpm_min=40):
        key = f"dominant_tempo_energy_{win_length}_{center}_{window}_{top_k}"
        if key in self._cache_tempogram:
            return self._cache_tempogram[key]
        
        spec = self._tempo_spectrum(win_length=win_length, center=center, window=window)
        S = spec["spectrum"]
        bpm = spec["bpm"]

        if S.size == 0:
            result = {
                "bpm_peaks": np.array([], dtype=float),
                "energies": np.array([], dtype=float),
                "peak_ratios": np.array([], dtype=float)
            }

            self._cache_tempogram[key] = result
            return result
        
        # Exclude DC and very low sub-harmonics
        valid = np.where((bpm >= bpm_min))[0]
        if valid.size == 0:
            result = {
                "bpm_peaks": np.array([], dtype=float),
                "energies": np.array([], dtype=float),
                "peak_ratios": np.array([], dtype=float)
            }

            self._cache_tempogram[key] = result
            return result
        
        idx_sorted = valid[np.argsort(S[valid])[::-1]]
        idx_top = idx_sorted[:top_k]

        bpm_peaks = bpm[idx_top]
        energies = S[idx_top]
        peak_ratios = energies/(energies[0] + EPS)

        result = {
            "bpm_peaks": bpm_peaks.astype(float),
            "energies": energies.astype(float),
            "peak_ratios": peak_ratios.astype(float)
        }

        self._cache_tempogram[key] = result
        return result
    
    def _tempo_spectral_centroid(self, win_length=None, center=None, window="hann", 
                             bpm_min=30.0, bpm_max=300.0):
        """
        Tempo spectral centroid with BPM range filtering
        """
        key = f"tempo_spectral_centroid_{win_length}_{center}_{window}_{bpm_min}_{bpm_max}"
        if key in self._cache_tempogram:
            return self._cache_tempogram[key]
        
        spec = self._tempo_spectrum(win_length=win_length, center=center, window=window)
        S = spec["spectrum"]
        bpm = spec["bpm"]

        if S.size == 0:
            result = {
                "centroid": 0.0,
                "bpm": bpm 
            }
            self._cache_tempogram[key] = result
            return result
        
        # Filter to plausible BPM range
        valid_mask = (bpm >= bpm_min) & (bpm <= bpm_max)
        S_filtered = S[valid_mask]
        bpm_filtered = bpm[valid_mask]
        
        if S_filtered.size == 0 or np.sum(S_filtered) < EPS:
            result = {
                "centroid": 0.0,
                "bpm": bpm 
            }
            self._cache_tempogram[key] = result
            return result
        
        centroid = float(np.sum(bpm_filtered * S_filtered) / (np.sum(S_filtered) + EPS))
        
        result = {
            "centroid": centroid,
            "bpm": bpm 
        }

        self._cache_tempogram[key] = result
        return result

    def _tempo_bandwidth(self, win_length=None, center=None, window="hann",
                        bpm_min=30.0, bpm_max=300.0):
        """
        Tempo spectral bandwidth with BPM range filtering
        """
        key = f"tempo_bandwidth_{win_length}_{center}_{window}_{bpm_min}_{bpm_max}"
        if key in self._cache_tempogram:
            return self._cache_tempogram[key]
        
        spec = self._tempo_spectrum(win_length=win_length, center=center, window=window)
        S = spec["spectrum"]
        bpm = spec["bpm"]

        if S.size == 0:
            result = {
                "bandwidth": 0.0,
                "centroid": 0.0,
                "bpm": bpm 
            }
            self._cache_tempogram[key] = result
            return result
        
        # Filter to plausible BPM range
        valid_mask = (bpm >= bpm_min) & (bpm <= bpm_max)
        S_filtered = S[valid_mask]
        bpm_filtered = bpm[valid_mask]
        
        if S_filtered.size == 0 or np.sum(S_filtered) < EPS:
            result = {
                "bandwidth": 0.0,
                "centroid": 0.0,
                "bpm": bpm
            }
            self._cache_tempogram[key] = result
            return result
        
        mu = np.sum(bpm_filtered * S_filtered) / (np.sum(S_filtered) + EPS)
        bw = np.sqrt(np.sum((bpm_filtered - mu)**2 * S_filtered) / (np.sum(S_filtered) + EPS))

        result = {
            "bandwidth": float(bw),
            "centroid": float(mu),
            "bpm": bpm
        }

        self._cache_tempogram[key] = result
        return result

    def _tempo_skewness(self, win_length=None, center=None, window="hann",
                    bpm_min=30.0, bpm_max=300.0):
        """
        Tempo spectral skewness with BPM range filtering
        """
        key = f"tempo_skewness_{win_length}_{center}_{window}_{bpm_min}_{bpm_max}"
        if key in self._cache_tempogram:
            return self._cache_tempogram[key]
        
        spec = self._tempo_spectrum(win_length=win_length, center=center, window=window)
        S = spec["spectrum"]
        bpm = spec["bpm"]

        if S.size == 0:
            result = {
                "skewness": 0.0
            }
            self._cache_tempogram[key] = result
            return result
        
        # Filter to plausible BPM range
        valid_mask = (bpm >= bpm_min) & (bpm <= bpm_max)
        S_filtered = S[valid_mask]
        bpm_filtered = bpm[valid_mask]
        
        if S_filtered.size == 0 or np.sum(S_filtered) < EPS:
            result = {
                "skewness": 0.0,
                "centroid": 0.0,
                "bandwidth": 0.0,
                "bpm": bpm
            }
            self._cache_tempogram[key] = result
            return result
        
        mu = np.sum(bpm_filtered * S_filtered) / (np.sum(S_filtered) + EPS)
        var = np.sum((bpm_filtered - mu)**2 * S_filtered) / (np.sum(S_filtered) + EPS)
        sigma = np.sqrt(var)

        if sigma < EPS:
            skew = 0.0
        else:
            skew = np.sum((bpm_filtered - mu)**3 * S_filtered) / ((np.sum(S_filtered) + EPS) * (sigma**3 + EPS))

        result = {
            "skewness": float(skew),
            "centroid": float(mu),
            "bandwidth": float(sigma),
            "bpm": bpm
        }

        self._cache_tempogram[key] = result
        return result

    def _tempo_kurtosis(self, win_length=None, center=None, window="hann", excess=True,
                    bpm_min=30.0, bpm_max=300.0):
        """
        Tempo spectral kurtosis with BPM range filtering
        """
        key = f"tempo_kurtosis_{win_length}_{center}_{window}_{excess}_{bpm_min}_{bpm_max}"
        if key in self._cache_tempogram:
            return self._cache_tempogram[key]
        
        spec = self._tempo_spectrum(win_length=win_length, center=center, window=window)
        S = spec["spectrum"]
        bpm = spec["bpm"]

        if S.size == 0:
            result = {
                "kurtosis": 0.0
            }
            self._cache_tempogram[key] = result
            return result
        
        # Filter to plausible BPM range
        valid_mask = (bpm >= bpm_min) & (bpm <= bpm_max)
        S_filtered = S[valid_mask]
        bpm_filtered = bpm[valid_mask]
        
        if S_filtered.size == 0 or np.sum(S_filtered) < EPS:
            result = {
                "kurtosis": 0.0,
                "centroid": 0.0,
                "bandwidth": 0.0,
                "bpm": bpm
            }
            self._cache_tempogram[key] = result
            return result
        
        mu = np.sum(bpm_filtered * S_filtered) / (np.sum(S_filtered) + EPS)
        var = np.sum((bpm_filtered - mu)**2 * S_filtered) / (np.sum(S_filtered) + EPS)
        sigma2 = var + EPS

        kurt = np.sum((bpm_filtered - mu)**4 * S_filtered) / ((np.sum(S_filtered) + EPS) * (sigma2**2 + EPS))

        if excess:
            kurt -= 3.0

        result = {
            "kurtosis": float(kurt),
            "centroid": float(mu),
            "bandwidth": float(np.sqrt(var)),
            "bpm": bpm
        }

        self._cache_tempogram[key] = result
        return result
    
    # Beat Position
    def _beat_time_from_frames(self, beat_frames):
        beat_frames = np.asarray(beat_frames, dtype=int)
        beat_times = librosa.frames_to_time(beat_frames, sr=self.sr, hop_length=self.H)

        return beat_times
    
    def _beat_period_from_beats(self, beat_times):
        beat_times = np.asarray(beat_times, dtype=float)
        if beat_times.size < 2:
            return 0.0
        
        periods = np.diff(beat_times)
        median_period = float(np.median(periods))

        return median_period
    
    def _beat_position(self, beat_times=None, beat_frames=None, mode="phase"):
        """
        Beat phase or position for each onset-envelope frame
        mode:
            - "phase": normalized phase in [0, 1) of the beat cycle
            - "fractional": fractional position within the beat period (can be >1)
            - "nearest": time to nearest beat (can be negative)
        """
        key = f"beat_position_{mode}"
        if key in self._cache_tempogram:
            return self._cache_tempogram[key]
        
        onset = self._onset_strength()
        t = onset["times"]

        if beat_times is None and beat_frames is None:
            result = {
                "beat_position": np.array([], dtype=float),
                "beat_index": np.array([], dtype=int),
                "beat_period": 0.0
            }

            self._cache_tempogram[key] = result
            return result
        
        if beat_times is None:
            beat_times = self._beat_time_from_frames(beat_frames)

        beat_times = np.asarray(beat_times, dtype=float)
        if beat_times.size < 2:
            result = {
                "beat_position": np.array([], dtype=float),
                "beat_index": np.array([], dtype=int),
                "beat_period": 0.0
            }

            self._cache_tempogram[key] = result
            return result
        
        beat_period = self._beat_period_from_beats(beat_times)
        idx = np.searchsorted(beat_times, t, side="right") - 1
        idx = np.clip(idx, 0, beat_times.size - 2)

        phase = (t - beat_times[idx])/(beat_times[idx + 1] - beat_times[idx] + EPS)
        phase = np.mod(phase, 1.0)

        if mode in ("phase", "fractional"):
            pos = phase
        elif mode == "nearest":
            pos = phase
            pos = np.where(pos > 0.5, pos - 1.0, pos)
        else:
            raise ValueError("mode must be 'phase', 'fractional', or 'nearest'")
        
        result = {
            "beat_position": pos.astype(float),
            "beat_index": idx.astype(int),
            "beat_period": beat_period
        }

        self._cache_tempogram[key] = result
        return result
    
    def _beat_alignment_histogram(self, beat_times=None, beat_frames=None, n_bins=16, normalize=True):
        """
        Histogram of event positions within the beat cycle, aligned to nearest beat
        """
        key = f"beat_alignment_histogram_{n_bins}_{normalize}"
        if key in self._cache_tempogram:
            return self._cache_tempogram[key]
        
        bp = self._beat_position(beat_times=beat_times, beat_frames=beat_frames, mode="phase")
        phase = bp["beat_position"]

        if phase.size == 0:
            hist = np.zeros(n_bins, dtype=float)
            result = {
                "histogram": hist,
                "bins": np.linspace(0, 1, n_bins + 1),
                "peak_bin": -1
            }

            self._cache_tempogram[key] = result
            return result
        
        hist, bins = np.histogram(phase, bins=n_bins, range=(0.0, 1.0))
        hist = hist.astype(float)
        if normalize:
            hist = hist/(np.sum(hist) + EPS)

        peak_bin = int(np.argmax(hist)) if hist.size > 0 else -1

        result = {
            "histogram": hist,
            "bins": bins,
            "peak_bin": peak_bin
        }

        self._cache_tempogram[key] = result
        return result
    
    def _interbeat_interval_variance(self, beat_times=None, beat_frames=None, normalize=False):
        """
        Variance of inter-beat intervals, optionally normalized by mean interval
        """
        key = f"interbeat_interval_variance_{normalize}"
        if key in self._cache_tempogram:
            return self._cache_tempogram[key]
        
        if beat_times is None and beat_frames is None:
            self._cache_tempogram[key] = 0.0
            return 0.0
        
        if beat_times is None:
            beat_times = self._beat_time_from_frames(beat_frames)

        beat_times = np.asarray(beat_times, dtype=float)
        if beat_times.size < 3:
            self._cache_tempogram[key] = 0.0
            return 0.0
        
        ibi = np.diff(beat_times)
        var = float(np.var(ibi, ddof=1)) if ibi.size > 1 else 0.0

        if normalize:
            mean_ibi = float(np.mean(ibi))
            var /= (mean_ibi**2 + EPS)

        self._cache_tempogram[key] = var
        return var
    
    def _beat_sync_offset(self, beat_times=None, beat_frames=None, event_times=None, event_frames=None, absolute=True):
        """
        Offset of events from nearest beat, averaged across all events. If absolute=True, returns mean absolute offset, otherwise returns mean signed offset (positive means event occurs after beat)
        """
        key = f"beat_sync_offset_{absolute}"
        if key in self._cache_tempogram:
            return self._cache_tempogram[key]
        
        if beat_times is None and beat_frames is None:
            result = {
                "offsets": np.array([], dtype=float),
                "mean_offset": 0.0,
                "mean_abs_offset": 0.0
            }

            self._cache_tempogram[key] = result
            return result
        
        if beat_times is None:
            beat_times = self._beat_time_from_frames(beat_frames)

        if event_times is None:
            onset = self._onset_strength()
            event_times = onset["times"]
        elif event_times is None and event_frames is not None:
            event_times = librosa.frames_to_time(np.asarray(event_frames, dtype=int), sr=self.sr, hop_length=self.H)

        beat_times = np.asarray(beat_times, dtype=float)
        event_times = np.asarray(event_times, dtype=float)

        if beat_times.size < 2 or event_times.size == 0:
            result = {
                "offsets": np.array([], dtype=float),
                "mean_offset": 0.0,
                "mean_abs_offset": 0.0
            }

            self._cache_tempogram[key] = result
            return result
        
        idx = np.searchsorted(beat_times, event_times, side="right") - 1
        idx = np.clip(idx, 0, beat_times.size - 2)

        ibi = beat_times[idx + 1] - beat_times[idx]
        offset = event_times - beat_times[idx]
        offset_norm = offset/(ibi + EPS)

        if absolute:
            summary = float(np.mean(np.abs(offset)))
            summary_norm = float(np.mean(np.abs(offset_norm)))
        else:
            summary = float(np.mean(offset))
            summary_norm = float(np.mean(offset_norm))

        result = {
            "offsets": offset.astype(float),
            "offsets_norm": offset_norm.astype(float),
            "mean_offset": summary,
            "mean_abs_offset": float(np.mean(np.abs(offset))),
            "mean_offset_norm": summary_norm,
            "mean_abs_offset_norm": float(np.mean(np.abs(offset_norm)))
        }

        self._cache_tempogram[key] = result
        return result
    
    # Spotify-based Tempogram features
    def _beat_periodic_energy(self, bpm_min=40.0, bpm_max=240.0, norm_sum=True):
        key = f"beat_periodic_energy_{bpm_min}_{bpm_max}_{norm_sum}"
        if key in self._cache_tempogram:
            return self._cache_tempogram[key]

        spec = self._tempo_spectrum(win_length=self.N, center=self.center, window="hann")
        S = spec["spectrum"]
        bpm = spec["bpm"]

        if S.size == 0:
            out = np.array([], dtype=float)
        else:
            mask = (bpm >= bpm_min) & (bpm <= bpm_max) & np.isfinite(bpm)
            out = S[mask]

        self._cache_tempogram[key] = out
        return out

    def _loudness_tempogram_per_beat(self, beat_times=None, beat_frames=None):
        key = f"loudness_tempogram_per_beat_{beat_times is not None}_{beat_frames is not None}"
        if key in self._cache_tempogram:
            return self._cache_tempogram[key]
        
        onset = self._onset_strength()
        env = onset["onset_env"]
        times = onset["times"]

        if beat_times is None:
            if beat_frames is None:
                g = self._global_bpm()
                if g["bpm"] <= 0 or times.size < 2:
                    out = np.array([], dtype=float)
                    self._cache_tempogram[key] = out
                    return out
                beat_period = 60.0/g["bpm"]
                beat_times = np.arange(times[0], times[-1] + beat_period, beat_period)
            else:
                beat_times = self._beat_time_from_frames(beat_frames)

        beat_times = np.asarray(beat_times, dtype=float)
        if beat_times.size == 0:
            out = np.array([], dtype=float)
            self._cache_tempogram[key] = out
            return out

        vals = np.zeros(beat_times.size, dtype=float)
        half_win = 0.5*(np.median(np.diff(beat_times)) if beat_times.size > 1 else (self.H/float(self.sr)))

        for i, bt in enumerate(beat_times):
            lo = bt - half_win
            hi = bt + half_win
            mask = (times >= lo) & (times< hi)
            if np.any(mask):
                vals[i] = float(np.mean(env[mask]))
            else:
                idx = int(np.argmin(np.abs(times - bt)))
                vals[i] = float(env[idx])

        self._cache_tempogram[key] = vals
        return vals

    def _danceability_tempogram(self, bpm_min=40.0, bpm_max=240.0, norm_sum=True):
        key = f"danceability_tempogram_{bpm_min}_{bpm_max}_{norm_sum}"
        if key in self._cache_tempogram:
            return self._cache_tempogram[key]
        
        pulse = self._pulse_clarity(bpm_min=bpm_min, bpm_max=bpm_max, norm_sum=norm_sum)["clarity"]
        stab = self._tempo_stability_index(bpm_min=bpm_min, bpm_max=bpm_max, norm_sum=norm_sum)["stability"]
        beat_strength = self._beat_strength(bpm_min=bpm_min, bpm_max=bpm_max, norm_sum=norm_sum)["strength"]
        multi = self._multi_periodic_structure(bpm_min=bpm_min, bpm_max=bpm_max, norm_sum=norm_sum)["multi_periodicity"]
        var = self._beat_fluctuation_rate(bpm_min=bpm_min, bpm_max=bpm_max, norm_sum=norm_sum)

        val = float(np.clip(
            0.30*pulse +
            0.25*stab +
            0.20*beat_strength +
            0.15*(1.0 - np.tanh(var/10.0)) +
            0.10*(1.0 - np.clip(multi/2.0, 0.0, 1.0)),
            0.0, 1.0
        ))

        self._cache_tempogram[key] = val
        return val

    def _valence_tempogram(self, bpm_min=40.0, bpm_max=240.0, norm_sum=True):
        key = f"valence_tempogram_{bpm_min}_{bpm_max}_{norm_sum}"
        if key in self._cache_tempogram:
            return self._cache_tempogram[key]

        clarity = self._pulse_clarity(bpm_min=bpm_min, bpm_max=bpm_max, norm_sum=norm_sum)["clarity"]
        stab = self._tempo_stability_index(bpm_min=bpm_min, bpm_max=bpm_max, norm_sum=norm_sum)["stability"]
        cent = self._tempo_spectral_centroid(bpm_min=bpm_min, bpm_max=bpm_max)["centroid"]
        bandwidth = self._tempo_bandwidth(bpm_min=bpm_min, bpm_max=bpm_max)["bandwidth"]
        skew = self._tempo_skewness(bpm_min=bpm_min, bpm_max=bpm_max)["skewness"]

        centroid_score = float(np.clip((cent - bpm_min)/(bpm_max - bpm_min), 0.0, 1.0))
        spread_score = float(np.clip(1.0 - np.tanh(bandwidth/(0.5*(bpm_max - bpm_min) + EPS)), 0.0, 1.0))
        skew_score = float(np.clip(0.5 + 0.25*np.tanh(skew), 0.0, 1.0))

        val = float(np.clip(
            0.30*stab + 
            0.25*clarity + 
            0.20*centroid_score + 
            0.15*spread_score +
            0.10*skew_score,
            0.0, 1.0
        ))

        self._cache_tempogram[key] = val
        return val

    def _liveness_tempogram(self, bpm_min=40.0, bpm_max=240.0, norm_sum=True):
        key = f"liveness_tempogram_{bpm_min}_{bpm_max}_{norm_sum}"
        if key in self._cache_tempogram:
            return self._cache_tempogram[key]
        
        curve = self._tempo_variation_curve(bpm_min=bpm_min, bpm_max=bpm_max, norm_sum=norm_sum)["curve"]
        fluc = self._beat_fluctuation_rate(bpm_min=bpm_min, bpm_max=bpm_max, norm_sum=norm_sum)
        ibi_var = 0.0
        if "beat_times" in self._cache_tempogram:
            ibi_var = self._interbeat_interval_variance(beat_times=self._cache_tempogram["beat_times"], normalize=True)

        curve_score = float(np.clip(np.mean(curve)/50.0, 0.0, 1.0)) if curve.size > 0 else 0.0
        fluc_score = float(np.clip(np.tanh(fluc/10.0), 0.0, 1.0))
        ibi_score = float(np.clip(np.tanh(ibi_var), 0.0, 1.0))

        val = float(np.clip(
            0.40*curve_score +
            0.35*fluc_score +
            0.25*ibi_score,
            0.0, 1.0
        ))

        self._cache_tempogram[key] = val
        return val
    
    def _mode_tempogram(self, bpm_min=40.0, bpm_max=240.0, norm_sum=True):
        key = f"mode_tempogram_{bpm_min}_{bpm_max}_{norm_sum}"
        if key in self._cache_tempogram:
            return self._cache_tempogram[key]

        pulse = self._pulse_clarity(bpm_min=bpm_min, bpm_max=bpm_max, norm_sum=norm_sum)["clarity"]
        rep = self._beat_periodicity_strength(bpm_min=bpm_min, bpm_max=bpm_max, norm_sum=norm_sum)["strength"]
        multi = self._multi_periodic_structure(bpm_min=bpm_min, bpm_max=bpm_max, norm_score=norm_sum)["score"]
        stab = self._tempo_stability_index(bpm_min=bpm_min, bpm_max=bpm_max, norm_sum=norm_sum)["stability"]

        major_score = float(np.clip(0.35*stab + 0.30*pulse + 0.20*rep + 0.15*(1.0 - np.clip(multi/2.0, 0.0, 1.0)), 0.0, 1.0))
        minor_score = float(np.clip(0.65 - major_score, 0.0, 1.0))

        mode = "major" if major_score >= minor_score else "minor"

        out = {
            "mode": mode,
            "score_major": major_score,
            "score_minor": minor_score,
            "delta_score": float(major_score - minor_score)
        }

        self._cache_tempogram[key] = out
        return out
    
    def _time_signature_tempogram(self, bpm_min=40.0, bpm_max=240.0, norm_sum=True):
        key = f"time_signature_tempogram_{bpm_min}_{bpm_max}_{norm_sum}"
        if key in self._cache_tempogram:
            return self._cache_tempogram[key]

        g = self._global_bpm(bpm_min=bpm_min, bpm_max=bpm_max, norm_sum=norm_sum)
        m = self._multi_periodic_structure(bpm_min=bpm_min, bpm_max=bpm_max, norm_sum=norm_sum)
        p = self._pulse_clarity(bpm_min=bpm_min, bpm_max=bpm_max, norm_sum=norm_sum)
        s = self._tempo_stability_index(bpm_min=bpm_min, bpm_max=bpm_max, norm_sum=norm_sum)

        bpm = float(g["bpm"])
        score = float(np.clip(
            0.35 * m["score"] +
            0.25 * p["clarity"] +
            0.20 * s["stability"] +
            0.20 * (1.0 - np.clip(abs(bpm - 120.0) / 120.0, 0.0, 1.0)),
            0.0, 1.0
        ))

        if score >= 0.60:
            ts = 3 if m["half_bpm"] > 0 and abs(m["primary_bpm"] - 90.0) < abs(m["primary_bpm"] - 120.0) else 4
        else:
            ts = 4 if bpm >= 100.0 else 3

        out = {
            "time_signature": int(ts),
            "confidence": score,
            "primary_bpm": bpm,
            "structure_score": m["score"]
        }

        self._cache_tempogram[key] = out
        return out
    
    def spotify_audio_features(self, beat_times=None, beat_frames=None, bpm_min=40.0, bpm_max=240.0, norm_sum=True):
        loudness_per_beat = self._loudness_per_beat(beat_times=beat_times, beat_frames=beat_frames)
        return {
            "loudness_per_beat": loudness_per_beat,
            "danceability": self._danceability(bpm_min=bpm_min, bpm_max=bpm_max, norm_sum=norm_sum),
            "valence": self._valence(bpm_min=bpm_min, bpm_max=bpm_max, norm_sum=norm_sum),
            "liveness": self._liveness(bpm_min=bpm_min, bpm_max=bpm_max, norm_sum=norm_sum),
            "mode": self._mode(bpm_min=bpm_min, bpm_max=bpm_max, norm_sum=norm_sum)["mode"],
            "time_signature": self._time_signature(bpm_min=bpm_min, bpm_max=bpm_max, norm_sum=norm_sum)["time_signature"],
        }
    
class MFCCFeatures:
    def __init__(self, sig, n_mfcc=13, n_mels=40, n_fft=None, hop_length=None, fmin=0.0, fmax=None, dct_type=2, norm="ortho", lifter=0, htk=False, center=True, pad_mode="constant", log_mels=False, power=2.0, dtype=np.float32, compute=True):
        self.sig = sig
        self.y = np.asarray(sig.y, dtype=float)
        self.sr = sig.sr
        self.N = int(n_fft if n_fft is not None else sig.N)
        self.H = int(hop_length if hop_length is not None else sig.H)

        self.n_mfcc = int(n_mfcc)
        self.n_mels = int(n_mels)
        self.fmin = float(fmin)
        self.fmax = float(fmax) if fmax is not None else None
        self.dct_type = dct_type
        self.norm = norm
        self.lifter = int(lifter)
        self.htk = bool(htk)
        self.center = bool(center)
        self.pad_mode = pad_mode
        self.log_mels = bool(log_mels)
        self.power = float(power)
        self.dtype = dtype

        self._cache_mfcc = {}

        self.S = None
        self.S_db = None
        self.mfcc = None
        self.times = None

        if compute:
            self._compute_mfcc()

    def _compute_mfcc(self):
        key = f"mfcc_default"
        if key in self._cache_mfcc:
            return self._cache_mfcc[key]
        
        self.S = librosa.feature.melspectrogram(
            y=self.y,
            sr=self.sr,
            n_fft=self.N,
            hop_length=self.H,
            win_length=self.N,
            window="hann",
            center=self.center,
            pad_mode=self.pad_mode,
            n_mels=self.n_mels,
            power=self.power,
            fmin=self.fmin,
            fmax=self.fmax,
            htk=self.htk,
            dtype=self.dtype
        )

        if self.log_mels:
            S_in = np.log(self.S + EPS)
        else:
            self.S_db = librosa.power_to_db(self.S, ref=np.max)
            S_in = self.S_db

        self.mfcc = librosa.feature.mfcc(
            S=S_in,
            sr=self.sr,
            n_mfcc=self.n_mfcc,
            dct_type=self.dct_type,
            norm=self.norm,
            lifter=self.lifter
        )

        self.times = librosa.frames_to_time(
            np.arange(self.mfcc.shape[1]),
            sr=self.sr,
            hop_length=self.H
        )

        result = {
            "S": self.S,
            "S_db": self.S_db,
            "mfcc": self.mfcc,
            "times": self.times
        }

        self._cache_mfcc[key] = result
        return result
    
    # Staticistics Features
    def _mfcc_mean(self):
        key = "mfcc_mean"
        if key in self._cache_mfcc:
            return self._cache_mfcc[key]
        
        M = self.mfcc
        if M is None or M.size == 0:
            result = np.array([], dtype=float)
            self._cache_mfcc[key] = result
            return result
        
        mean = np.mean(M, axis=1).astype(float)
        self._cache_mfcc[key] = mean
        return mean
    
    def _mfcc_variance(self, ddof=1):
        key = f"mfcc_variance_ddof_{ddof}"
        if key in self._cache_mfcc:
            return self._cache_mfcc[key]
        
        M = self.mfcc
        if M is None or M.size == 0:
            result = np.array([], dtype=float)
            self._cache_mfcc[key] = result
            return result
        
        var = np.var(M, axis=1, ddof=ddof).astype(float)
        self._cache_mfcc[key] = var
        return var
    
    def _mfcc_skewness(self):
        key = "mfcc_skewness"
        if key in self._cache_mfcc:
            return self._cache_mfcc[key]
        
        M = self.mfcc
        if M is None or M.size == 0:
            result = np.array([], dtype=float)
            self._cache_mfcc[key] = result
            return result
        
        mu = self._mfcc_mean()
        # Reshape mean to (n_mfcc, 1) for broadcasting across time frames
        mu = mu[:, np.newaxis]
        x = M - mu
        m2 = np.mean(x**2, axis=1)
        m3 = np.mean(x**3, axis=1)
        skew = m3/(m2**1.5 + EPS)

        skew = skew.astype(float)
        self._cache_mfcc[key] = skew
        return skew

    def _mfcc_kurtosis(self, excess=True):
        key = f"mfcc_kurtosis_excess_{excess}"
        if key in self._cache_mfcc:
            return self._cache_mfcc[key]

        M = self.mfcc
        if M is None or M.size == 0:
            result = np.array([], dtype=float)
            self._cache_mfcc[key] = result
            return result
        
        mu = self._mfcc_mean()
        # Reshape mean to (n_mfcc, 1) for broadcasting across time frames
        mu = mu[:, np.newaxis]
        x = M - mu
        m2 = np.mean(x**2, axis=1)
        m4 = np.mean(x**4, axis=1)
        kurt = m4/(m2**2 + EPS)

        if excess:
            kurt -= 3.0

        kurt = kurt.astype(float)
        self._cache_mfcc[key] = kurt
        return kurt
    
    # Temporal Features
    def _mfcc_delta(self, width=9, order=1, mode="interp"):
        key = f"mfcc_delta_width_{width}_order_{order}_mode_{mode}"
        if key in self._cache_mfcc:
            return self._cache_mfcc[key]
        
        if self.mfcc is None or self.mfcc.size == 0:
            result = np.array([[]], dtype=float)
            self._cache_mfcc[key] = result
            return result
        
        delta = librosa.feature.delta(
            self.mfcc,
            width=width,
            order=order,
            axis=1,
            mode=mode
        )

        delta = delta.astype(float)
        self._cache_mfcc[key] = delta
        return delta
    
    def _mfcc_delta2(self, width=9, mode="interp"):
        return self._mfcc_delta(width=width, order=2, mode=mode)
    
    def _mfcc_temporal_stability(self, ddof=0):
        key = f"mfcc_temporal_stability_ddof_{ddof}"
        if key in self._cache_mfcc:
            return self._cache_mfcc[key]
        
        if self.mfcc is None or self.mfcc.size == 0:
            result = np.array([], dtype=float)
            self._cache_mfcc[key] = result
            return result
        
        mu = self._mfcc_mean()
        sigma = np.std(self.mfcc, axis=1, ddof=ddof)
        cv = sigma/(np.abs(mu) + EPS)
        stability = 1.0/(1.0 + cv)

        stability = stability.astype(float)
        self._cache_mfcc[key] = stability
        return stability
    
    def _mfcc_autocorrelation(self, max_lag=None, normalize=True):
        key = f"mfcc_autocorrelation_maxlag_{max_lag}_normalize_{normalize}"
        if key in self._cache_mfcc:
            return self._cache_mfcc[key]
        
        if self.mfcc is None or self.mfcc.size == 0:
            result = {
                "acf": np.array([[]], dtype=float),
                "lags": np.array([], dtype=int)
            }

            self._cache_mfcc[key] = result
            return result
        
        M, T = self.mfcc.shape
        if max_lag is None:
            max_lag = min(T - 1, int(self.frame_rate*4.0))
        max_lag = int(max(1, min(max_lag, T - 1)))

        acf = np.zeros((M, max_lag + 1), dtype=float)

        for k in range(M):
            x = self.mfcc[k] - np.mean(self.mfcc[k])
            r = np.correlate(x, x, mode='full')
            if normalize:
                r /= (r[0] + EPS)

            center = len(r)//2
            acf[k] = r[center:center + max_lag + 1]

        lags = np.arange(max_lag + 1, dtype=int)
        result = {
            "acf": acf,
            "lags": lags
        }
        self._cache_mfcc[key] = result
        return result

    # Spectral Shape Proxies
    def _mfcc_spectral_slope_proxy(self, coeff=0, aggregate="mean"):
        key = f"mfcc_spectral_slope_proxy_coeff_{coeff}_aggregate_{aggregate}"
        if key in self._cache_mfcc:
            return self._cache_mfcc[key]

        if self.mfcc is None or self.mfcc.size == 0:
            result = 0.0
            self._cache_mfcc[key] = result
            return result

        x = self.mfcc[coeff]
        if aggregate == "mean":
            proxy = float(np.mean(x))
        elif aggregate == "median":
            proxy = float(np.median(x))
        elif aggregate == "rms":
            proxy = float(np.sqrt(np.mean(x**2)))
        else:
            raise ValueError("Invalid aggregate method: must be 'mean', 'median', or 'rms'")
        
        self._cache_mfcc[key] = proxy
        return proxy
    
    def _mfcc_brightness_proxy(self, coeff=0, invert=False, aggregate="mean"):
        key = f"mfcc_brightness_proxy_coeff_{coeff}_invert_{invert}_aggregate_{aggregate}"
        if key in self._cache_mfcc:
            return self._cache_mfcc[key]
        
        vals = self._mfcc_spectral_slope_proxy(coeff=coeff, aggregate=aggregate)
        brightness = -vals if invert else vals

        self._cache_mfcc[key] = float(brightness)
        return float(brightness)
    
    def _mfcc_sharpness_proxy(self, coeff=0, aggregate="mean", absolute=True):
        key = f"mfcc_sharpness_proxy_coeff_{coeff}_aggregate_{aggregate}_absolute_{absolute}"
        if key in self._cache_mfcc:
            return self._cache_mfcc[key]
        
        if self.mfcc is None or self.mfcc.size == 0:
            result = 0.0
            self._cache_mfcc[key] = result
            return result
        
        x = self.mfcc[coeff]
        if aggregate == "mean":
            val = float(np.mean(x))
        elif aggregate == "median":
            val = float(np.median(x))
        elif aggregate == "rms":
            val = float(np.sqrt(np.mean(x**2)))
        else:
            raise ValueError("Invalid aggregate method: must be 'mean', 'median', or 'rms'")
        
        if absolute:
            val = np.abs(val)

        self._cache_mfcc[key] = float(val)
        return float(val)
    
    def _mfcc_high_order_energy(self, start_coeff=6, normalize=False, order='l2'):
        key = f"mfcc_high_order_energy_start_{start_coeff}_normalize_{normalize}_order_{order}"
        if key in self._cache_mfcc:
            return self._cache_mfcc[key]
        
        if self.mfcc is None or self.mfcc.size == 0:
            result = 0.0
            self._cache_mfcc[key] = result
            return result

        X = self.mfcc[start_coeff:, :]
        if X.size == 0:
            result = 0.0
            self._cache_mfcc[key] = result
            return result
        
        if order == 'l2':
            energy = float(np.sum(X**2))
        elif order == 'l1':
            energy = float(np.sum(np.abs(X)))
        elif order == 'rms':
            energy = float(np.sqrt(np.mean(X**2)))
        else:
            raise ValueError("Invalid order: must be 'l2', 'l1', or 'rms'")
        
        if normalize:
            denom = float(np.sum(self.mfcc**2) + EPS) if order == 'l2' else float(np.sum(np.abs(self.mfcc)) + EPS)
            energy /= denom

        self._cache_mfcc[key] = energy
        return energy
    
    def _mfcc_noise_inharmonicity_proxy(self, coeff=0, normalize=True):
        key = f"mfcc_noise_inharmonicity_proxy_coeff_{coeff}_normalize_{normalize}"
        if key in self._cache_mfcc:
            return self._cache_mfcc[key]
        
        result = self._mfcc_high_order_energy(start_coeff=coeff, normalize=normalize, order='l2')

        self._cache_mfcc[key] = result
        return result

    # Envelope Features
    def _mfcc_attack_smoothness(self, attack_frames=None, normalize=True):
        key = f"mfcc_attack_smoothness_normalize_{attack_frames}_{normalize}"
        if key in self._cache_mfcc:
            return self._cache_mfcc[key]
        
        if self.mfcc is None or self.mfcc.size == 0:
            result = 0.0
            self._cache_mfcc[key] = result
            return result
        
        X = self.mfcc
        T = X.shape[1]

        if attack_frames is None:
            attack_frames = max(2, min(T//5, 10))
        attack_frames = int(max(2, min(attack_frames, T)))

        segment = X[:, :attack_frames]
        diff = np.diff(segment, axis=1)
        step_energy = np.linalg.norm(diff, axis=0)

        if step_energy.size == 0:
            result = 1.0
            self._cache_mfcc[key] = result
            return result
        
        if normalize:
            denom = np.max(np.linalg.norm(segment, axis=0)) + EPS
            smoothness = 1.0 - (np.mean(step_energy)/denom)
        else:
            smoothness = 1.0/(1.0 + np.mean(step_energy))

        smoothness = float(safe_clip01(smoothness))
        self._cache_mfcc[key] = smoothness
        return smoothness
    
    def _mfcc_sustain_stability(self, attack_frames=None, sustain_frames=None, ddof=0, normalize=True):
        key = f"mfcc_sustain_stability_attack_{attack_frames}_sustain_{sustain_frames}_ddof_{ddof}_normalize_{normalize}"
        if key in self._cache_mfcc:
            return self._cache_mfcc[key]
        
        if self.mfcc is None or self.mfcc.size == 0:
            result = 0.0
            self._cache_mfcc[key] = result
            return result
        
        X = self.mfcc
        T = X.shape[1]

        if attack_frames is None:
            attack_frames = max(2, min(T//5, 10))
        attack_frames = int(max(0, min(attack_frames, T)))

        if sustain_frames is None:
            sustain_frames = T - attack_frames
        sustain_frames = int(max(0, min(sustain_frames, T - attack_frames)))

        start = attack_frames
        end = min(T, attack_frames + sustain_frames)

        if end <= start:
            result = 1.0
            self._cache_mfcc[key] = result
            return result
        
        sustain = X[:, start:end]
        mu = np.mean(sustain, axis=1)
        sigma = np.std(sustain, axis=1, ddof=ddof)

        if normalize:
            stability = 1.0 - np.mean(sigma/(np.abs(mu) + EPS))
        else:
            stability = 1.0/(1.0 + np.mean(sigma))

        result = float(safe_clip01(stability))
        self._cache_mfcc[key] = result
        return result
    
    def _mfcc_smoothness_index(self, weight_delta=1.0, weight_ddelta=0.5, normalize=True):
        key = f"mfcc_smoothness_index_weight_delta_{weight_delta}_weight_ddelta_{weight_ddelta}_normalize_{normalize}"
        if key in self._cache_mfcc:
            return self._cache_mfcc[key]
        
        if self.mfcc is None or self.mfcc.size == 0:
            result = 0.0
            self._cache_mfcc[key] = result
            return result
        
        X = self.mfcc
        diffs1 = np.diff(X, axis=1)
        diffs2 = np.diff(X, n=2, axis=1)

        e1 = np.mean(np.linalg.norm(diffs1, axis=0)) if diffs1.size > 0 else 0.0
        e2 = np.mean(np.linalg.norm(diffs2, axis=0)) if diffs2.size > 0 else 0.0

        roughness = weight_delta*e1 + weight_ddelta*e2

        if normalize:
            scale = np.mean(np.linalg.norm(X, axis=0)) + EPS
            smoothness = 1.0/(1.0 + roughness/scale)
        else:
            smoothness = 1.0/(1.0 + roughness)

        smoothness = float(safe_clip01(smoothness))
        self._cache_mfcc[key] = smoothness
        return smoothness

    # Noise/Speech Proxies
    def _mfcc_high_order_magnitude(self, start_coeff=6, mode='l2', normalize=True):
        key = f"mfcc_high_order_magnitude_start_{start_coeff}_mode_{mode}_normalize_{normalize}"
        if key in self._cache_mfcc:
            return self._cache_mfcc[key]
        
        if self.mfcc is None or self.mfcc.size == 0:
            result = 0.0
            self._cache_mfcc[key] = result
            return result
        
        X = self.mfcc[start_coeff:, :]
        if X.size == 0:
            result = 0.0
            self._cache_mfcc[key] = result
            return result
        
        if mode == 'l2':
            val = np.sqrt(np.mean(X**2))
        elif mode == 'l1':
            val = np.mean(np.abs(X))
        elif mode == 'energy':
            val = np.mean(X**2)
        else:
            raise ValueError("Invalid mode: must be 'l2', 'l1', or 'energy'")
        
        if normalize:
            denom = np.sqrt(np.mean(self.mfcc**2)) + EPS if mode == 'l2' else np.mean(np.abs(self.mfcc)) + EPS
            val /= denom
        
        self._cache_mfcc[key] = float(val)
        return float(val)
    
    def _mfcc_formant_shape_detection(self, reference=None, normalize=True, coeffs=(0, 1, 2, 3)):
        key = f"mfcc_formant_shape_detection_{reference is not None}_normalize_{normalize}_coeffs_{coeffs}"
        if key in self._cache_mfcc:
            return self._cache_mfcc[key]
        
        if self.mfcc is None or self.mfcc.size == 0:
            result = 0.0
            self._cache_mfcc[key] = result
            return result
        
        idx = np.array(coeffs, dtype=int)
        idx = idx[idx < self.mfcc.shape[0]]
        if idx.size == 0:
            result = {
                "score": 0.0,
                "pattern": np.array([], dtype=float)
            }
            self._cache_mfcc[key] = result
            return result
        
        pattern = np.mean(self.mfcc[idx, :], axis=1).astype(float)

        if normalize:
            pattern = (pattern - np.mean(pattern))/(np.std(pattern) + EPS)

        if reference is None:
            score = float(np.linalg.norm(pattern))
        else:
            reference = np.asarray(reference, dtype=float).reshape(-1)
            m = min(reference.size, pattern.size)
            if m == 0:
                score = 0.0
            else:
                score = float(np.linalg.norm(pattern[:m] - reference[:m]))

        result = {
            "score": score,
            "pattern": pattern
        }
        self._cache_mfcc[key] = result
        return result
    
    def _mfcc_transient_roughness(self, width=9, normalize=True, mode="interp"):
        key = f"mfcc_transient_roughness_{width}_{normalize}_{mode}"
        if key in self._cache_mfcc:
            return self._cache_mfcc[key]
        
        if self.mfcc is None or self.mfcc.size == 0:
            result = 0.0
            self._cache_mfcc[key] = 0.0
            return result
        
        d1 = librosa.feature.delta(self.mfcc, width=width, order=1, axis=1, mode=mode)
        d2 = librosa.feature.delta(self.mfcc, width=width, order=2, axis=1, mode=mode)

        rough1 = np.mean(np.linalg.norm(d1, axis=0))
        rough2 = np.mean(np.linalg.norm(d2, axis=0))

        if normalize:
            scale = np.mean(np.linalg.norm(self.mfcc, axis=0)) + EPS
            roughness = (rough1 + 0.5*rough2)/scale
        else:
            roughness = rough1 + 0.5*rough2

        self._cache_mfcc[key] = float(roughness)
        return roughness
    
    # Spotify-based MFCC Features
    def _mfcc_frame_energy(self, normalize=True):
        key = f"mfcc_frame_energy_normalize_{normalize}"
        if key in self._cache_mfcc:
            return self._cache_mfcc[key]

        if self.mfcc is None or self.mfcc.size == 0:
            result = np.array([], dtype=float)
            self._cache_mfcc[key] = result
            return result
        
        E = np.sum(self.mfcc**2, axis=0).astype(float)
        if normalize:
            E /= (np.max(E) + EPS)

        self._cache_mfcc[key] = E
        return E

    def _mfcc_energy(self, normalize=True):
        key = f"mfcc_energy_normalize_{normalize}"
        if key in self._cache_mfcc:
            return self._cache_mfcc[key]

        E = self._mfcc_frame_energy(normalize=normalize)
        if E.size == 0:
            result = 0.0
        else:
            result = float(np.mean(E))

        self._cache_mfcc[key] = result
        return result

    def _mfcc_rms_energy(self, normalize=True):
        key = f"mfcc_rms_energy_normalize_{normalize}"
        if key in self._cache_mfcc:
            return self._cache_mfcc[key]

        if self.mfcc is None or self.mfcc.size == 0:
            result = 0.0
            self._cache_mfcc[key] = result
            return result

        if normalize:
            x = self.mfcc/(np.max(np.abs(self.mfcc)) + EPS)
        else:
            x = self.mfcc

        result = float(np.sqrt(np.mean(x**2)))

        self._cache_mfcc[key] = result
        return result

    def _mfcc_flux(self, normalize=True):
        key = f"mfcc_flux_normalize_{normalize}"
        if key in self._cache_mfcc:
            return self._cache_mfcc[key]

        if self.mfcc is None or self.mfcc.size == 0 or self.mfcc.shape[1] < 2:
            result = 0.0
            self._cache_mfcc[key] = 0.0
            return 0.0

        d = np.diff(self.mfcc, axis=1)
        flux = np.mean(np.linalg.norm(d, axis=0))

        if normalize:
            scale = np.mean(np.linalg.norm(self.mfcc, axis=0)) + EPS
            flux /= scale

        result = float(flux)

        self._cache_mfcc[key] = result
        return result

    def _mfcc_high_order_variance(self, start_coeff=6, normalize=True):
        key = f"mfcc_high_order_variance_start_{start_coeff}_normalize_{normalize}"
        if key in self._cache_mfcc:
            return self._cache_mfcc[key]
        
        if self.mfcc is None or self.mfcc.size == 0:
            result = 0.0
            self._cache_mfcc[key] = result
            return result
        
        X = self.mfcc[start_coeff:, :]
        if X.size == 0:
            result = 0.0
            self._cache_mfcc[key] = result
            return result

        v = float(np.mean(np.var(X, axis=1)))
        if normalize:
            v /= (np.mean(np.var(self.mfcc, axis=1)) + EPS)

        self._cache_mfcc[key] = v
        return v

    def _mfcc_entropy(self, normalize=True):
        key = f"mfcc_entropy_normalize_{normalize}"
        if key in self._cache_mfcc:
            return self._cache_mfcc[key]

        if self.mfcc is None or self.mfcc.size == 0:
            result = 0.0
            self._cache_mfcc[key] = result
            return result
        
        X = np.abs(self.mfcc)
        col = np.mean(X, axis=1)
        p = col/(np.sum(col) + EPS)
        ent = -np.sum(p*np.log(p + EPS))

        if normalize:
            ent /= np.log(len(p) + EPS)

        self._cache_mfcc[key] = float(ent)
        return float(ent)

    def _mfcc_smoothness(self, weight_delta=1.0, weight_ddelta=0.5, normalize=True):
        key = f"mfcc_smoothness_proxy_{weight_delta}_{weight_ddelta}_{normalize}"
        if key in self._cache_mfcc:
            return self._cache_mfcc[key]

        if self.mfcc is None or self.mfcc.size == 0:
            result = 0.0
            self._cache_mfcc[key] = result
            return result

        d1 = self._mfcc_delta()
        d2 = self._mfcc_delta2()

        e1 = float(np.mean(np.linalg.norm(d1, axis=0))) if d1.size > 0 else 0.0
        e2 = float(np.mean(np.linalg.norm(d2, axis=0))) if d2.size > 0 else 0.0
        rough = weight_delta*e1 + weight_ddelta*e2

        if normalize:
            scale = float(np.mean(np.linalg.norm(self.mfcc, axis=0)) + EPS)
            smooth = 1.0/(1.0 + rough/scale)
        else:
            smooth = 1.0/(1.0 + rough)

        smooth = float(safe_clip01(smooth))
        
        self._cache_mfcc[key] = smooth
        return smooth
    
    def _loudness_mfcc(self):
        key = "loudness_mfcc"
        if key in self._cache_mfcc:
            return self._cache_mfcc[key]

        loud = self._mfcc_energy(normalize=True)

        self._cache_mfcc[key] = loud
        return loud

    def _energy_mfcc(self):
        key = "energy_mfcc"
        if key in self._cache_mfcc:
            return self._cache_mfcc[key]
        
        e0 = self._mfcc_spectral_slope_proxy(coeff=0, aggregate="rms")
        e1 = self._mfcc_rms_energy(normalize=True)
        e2 = self._mfcc_energy(normalize=True)

        val = float(np.clip(0.4*abs(e0) + 0.3*e1 + 0.3*e2, 0.0, 1.0))

        self._cache_mfcc[key] = val
        return val
    
    def _speechiness_mfcc(self):
        key = "speechiness_mfcc"
        if key in self._cache_mfcc:
            return self._cache_mfcc[key]
        
        flux = self._mfcc_flux(normalize=True)
        high = self._mfcc_high_order_energy(start_coeff=6, normalize=True, order="l2")
        var = self._mfcc_high_order_variance(start_coeff=6, normalize=True)
        ent = self._mfcc_entropy(normalize=True)

        val = float(np.clip(
            0.40*flux + 
            0.30*np.clip(high, 0.0, 1.0) +
            0.20*np.clip(var, 0.0, 1.0) +
            0.10*ent,
            0.0, 1.0
        ))

        self._cache_mfcc[key] = val
        return val
    
    def _acousticness_mfcc(self):
        key = "acousticness_mfcc"
        if key in self._cache_mfcc:
            return self._cache_mfcc[key]

        smooth = self._mfcc_smoothness(normalize=True)
        ent = self._mfcc_entropy(normalize=True)
        high = self._mfcc_high_order_energy(start_coeff=6, normalize=True, order="l2")
        flux = self._mfcc_flux(normalize=True)

        val = float(np.clip(
            0.35*smooth +
            0.25*(1.0 - ent) +
            0.25*(1.0 - np.clip(high, 0.0, 1.0)) + 
            0.15*(1.0 - np.clip(flux, 0.0, 1.0)),
            0.0, 1.0
        ))

        self._cache_mfcc[key] = val
        return val

    def _valence_mfcc(self):
        key = "valence_mfcc"
        if key in self._cache_mfcc:
            return self._cache_mfcc[key]
        
        bright = self._mfcc_brightness_proxy(coeff=0, invert=False, aggregate="mean")
        smooth = self._mfcc_smoothness(normalize=True)
        rough = self._mfcc_transient_roughness(width=9, normalize=True, mode="interp")
        ent = self._mfcc_entropy(normalize=True)

        b = float(np.tanh(np.abs(bright)))
        r = float(np.clip(rough/(1.0 + rough), 0.0, 1.0))

        val = float(np.clip(
            0.35*b +
            0.30*smooth +
            0.20*(1.0 - r) +
            0.15*(1.0 - ent),
            0.0, 1.0
        ))

        self._cache_mfcc[key] = val
        return val

    def _liveness_mfcc(self):
        key = "liveness_mfcc"
        if key in self._cache_mfcc:
            return self._cache_mfcc[key]

        rough = self._mfcc_transient_roughness(width=9, normalize=True, mode="interp")
        flux = self._mfcc_flux(normalize=True)
        high = self._mfcc_high_order_energy(start_coeff=6, normalize=True, order="l2")
        ent = self._mfcc_entropy(normalize=True)

        val = float(np.clip(
            0.35*np.clip(rough/(1.0 + rough), 0.0, 1.0) +
            0.30*np.clip(flux, 0.0, 1.0) +
            0.20*np.clip(high, 0.0, 1.0) +
            0.15*ent,
            0.0, 1.0
        ))

        self._cache_mfcc[key] = val
        return val

    def _instrumentalness_mfcc(self):
        key = "instrumentalness_mfcc"
        if key in self._cache_mfcc:
            return self._cache_mfcc[key]

        speech = self._speechiness_mfcc()
        smooth = self._mfcc_smoothness(normalize=True)
        high = self._mfcc_high_order_energy(start_coeff=6, normalize=True, order="l2")

        val = float(np.clip(
            0.55*(1.0 - speech) +
            0.25*smooth +
            0.20*(1.0 - np.clip(high, 0.0, 1.0)),
            0.0, 1.0
        ))

        self._cache_mfcc[key] = val
        return val

    def spotify_audio_features(self):
        return {
            "loudness": self._loudness_mfcc(),
            "energy": self._energy_mfcc(),
            "speechiness": self._speechiness_mfcc(),
            "acousticness": self._acousticness_mfcc(),
            "valence": self._valence_mfcc(),
            "liveness": self._liveness_mfcc(),
            "instrumentalness": self._instrumentalness_mfcc(),
        }

class SpotifyFeatureProxies:
    """
    Proxies for Spotify-style features using:
        - time_features: time-domain stats (RMS, envelope, etc.)
        - freq_features: spectral stats (centroid, flatness, etc.)
        - mfcc_features: MFCCFeatures instance (statistical, temporal, shape, noise proxies)
        - chroma_features: ChromaFeatures isntance (key, mode, tonality)
        - tempo_features: TempogramFeatures instance (global/local tempo, pulse clarity, etc.)
    """

    def __init__(self, 
                 time_features=None,
                 freq_features=None,
                 mfcc_features=None,
                 chroma_features=None,
                 tempo_features=None):
        self.time_features = time_features
        self.freq_features = freq_features
        self.mfcc_features = mfcc_features
        self.chroma_features = chroma_features
        self.tempo_features = tempo_features

    # Loudness Features
    def loudness_time(self) -> float:
        if self.time_features is None:
            return 0.0
        
        return float(self.time_features._loudness_time_active_db())
    
    def loudness_freq(self) -> float:
        if self.freq_features is None:
            return 0.0
        
        return float(self.freq_features._loudness_freq_db())
    

    def loudness_mfcc(self) -> float:
        if self.mfcc_features is None:
            return 0.0
        
        return float(self.mfcc_features._loudness_mfcc_mean())
    
    def loudness_tempogram(self) -> float:
        if self.tempo_features is None or self.time_features is None:
            return 0.0
        
        return float(self.tempo_features._loudness_tempogram_per_beat(self.time_features))
    
    def loudness_fused(self) -> dict:
        """
        Fuse domain-based loudness estimates into one score.
        Returns both per-domain values and a fused estimate.

        Strategy:
          - work in a pseudo-dB space for time and frequency,
          - rescale MFCC/tempo proxies to roughly comparable ranges,
          - combine via weighted average.
        """
        L_time = self.loudness_time()       # dB
        L_freq = self.loudness_freq()       # dB
        L_mfcc = self.loudness_mfcc()       # arbitrary units
        L_temp = self.loudness_tempogram()  # RMS

        # Normalize partial proxies roughly into dB-like scale
        L_mfcc_db = 10.0*np.log10(abs(L_mfcc) + 1.0)
        L_temp_db = 20.0*np.log10(L_temp + EPS) if L_temp > 0.0 else -80.0

        # Weights: time + frequency stronger, MFCC + tempogram weaker
        vals = np.array([L_time, L_freq, L_mfcc_db, L_temp_db], dtype=float)
        weights = np.array([0.4, 0.4, 0.1, 0.1], dtype=float)

        # handle missing domains (zeros with weight zero)
        mask = ~np.isinf(vals) and ~np.isnan(vals)
        if not np.any(mask):
            fused = -80.0
        else:
            w = weights[mask]
            v = vals[mask]
            fused = float(np.sum(w*v)/(np.sum(w) + EPS))

        return {
            "loudness_time_db": float(L_time),
            "loudness_freq_db": float(L_freq),
            "loudness_mfcc_proxy": float(L_mfcc),
            "loudness_tempogram_proxy": float(L_temp),
            "loudness_fused_db": fused
        }
    
    # Energy Features
    def energy_time(self, normalize=True):
        """
        Full-support time-domain energy:
        mean short-time energy or RMS-derived energy.
        """
        if self.time_features is None:
            return 0.0
        
        key = f"energy_time_{normalize}"
        if key in self.time_features._cache_time:
            return self.time_features._cache_time[key]
        
        if hasattr(self.time_features, "_short_time_energy"):
            ste = self.time_features._short_time_energy()
            val = float(np.mean(ste)) if normalize else float(np.sum(ste))
        elif hasattr(self.time_features, "_rms_envelope"):
            rms = self.time_features._rms_envelope()
            val = float(np.mean(rms**2)) if normalize else float(np.sum(rms**2))
        else:
            val = float(np.mean(self.time_features.y**2))

        self.time_features._cache_time[key] = val
        return val
    
    def energy_frequency(self, weighted=False):
        """
        Full-support frequency-domain energy:
        mean power over the spectrogram, optionally perceptually weighted.
        """
        if self.freq_features is None:
            return 0.0
        
        key = f"energy_frequency_{weighted}"
        if hasattr(self.freq_features, "_cache_freq") and key in self.freq_features._cache_freq:
            return self.freq_features._cache_freq[key]
        
        if hasattr(self.freq_features, "_spectrogram_power"):
            S = self.freq_features._spectrogram_power()
        elif hasattr(self.freq_features, "S_pow") and self.freq_features.S_pow is not None:
            S = self.freq_features.S_pow
        elif hasattr(self.freq_features, "S_mag") and self.freq_features.S_mag is not None:
            S = self.freq_features.S_mag**2
        else:
            y = getattr(self.freq_features, "y", None)
            sr = getattr(self.freq_features, "sr", None)
            N = getattr(self.freq_features, "N", 2048)
            H = getattr(self.freq_features, 512)

            if y is None or sr is None:
                return 0.0

            S = np.abs(librosa.stft(y, n_fft=N, hop_length=H))**2

        if S.size == 0:
            val = 0.0
        elif weighted and hasattr(self.freq_features, "_fft_freqs"):
            f = np.asarray(self.freq_features._fft_freqs, dtype=float)
            w = 1.0 + 0.5*(f/(f.max() + EPS))
            val = float(np.mean(np.sum(S*w[:, None], axis=0)))
        else:
            val = float(np.mean(np.sum(S, axis=0)))

        if not hasattr(self.freq_features, "_cache_freq"):
            self.freq_features._cache_freq = {}
        
        self.freq_features._cache_freq[key] = val
        return val
    
    def energy_mfcc(self, start_coeff=0, normalize=True):
        """
        Partial MFCC energy proxy:
        low-order MFCC energy or c0-based energy if available.
        """
        if self.mfcc_features is None or getattr(self.mfcc_features, "mfcc", None) is None:
            return 0.0

        key = f"energy_mfcc_{start_coeff}_{normalize}"
        if key in self.mfcc._cache_mfcc:
            return self.mfcc_features._cache_mfcc[key]
        
        M = self.mfcc_features.mfcc
        if M.size == 0 or start_coeff >= M.shape[0]:
            val = 0.0
        else:
            X = M[start_coeff:, :]
            val = float(np.mean(np.sum(X**2, axis=0))) if normalize else float(np.sum(X**2))
        

        self.mfcc_features._cache_mfcc[key] = val
        return val
    
    def energy_chroma(self, normalize=True):
        """
        Weak partial chroma energy proxy:
        chroma vector norm or average pitch-class concentration.
        """
        if self.chroma_features is None:
            return 0.0
        
        key = f"energy_chroma_{normalize}"
        if hasattr(self.chroma_features, "_cache_chroma") and key in self.chroma_features._cache_chroma:
            return self.chroma_features._cache_chroma[key]

        if hasattr(self.chroma_features, "_chroma_vector"):
            C = self.chroma_features._chroma_vector()
        elif hasattr(self.chroma_features, "chroma"):
            C = self.chroma_features.chroma
        else:
            return 0.0
        
        if C is None or C.size == 0: 
            val = 0.0
        else:
            val = float(np.mean(np.sum(C**2, axis=0))) if normalize else float(np.sum(C**2))

        if not hasattr(self.chroma, "_cache_chroma"):
            self.chroma_features._cache_chroma = {}
        
        self.chroma_features._cache_chroma[key] = val
        return val

    def energy_fused(self, weights=None):
        """
        Fuse all domain energy proxies into a single scalar.
        Time and Frequency are primary; MFCC and chroma are secondary.
        """
        Et = self.energy_time(normalize=True)
        Ef = self.energy_frequency(weighted=True)
        Em = self.energy_mfcc(start_coeff=0, normalize=True)
        Ec = self.energy_chroma(normalize=True)

        vals = np.array([Et, Ef, Em, Ec], dtype=float)

        if weights is None:
            weights = np.array([0.4, 0.4, 0.15, 0.05], dtype=float)
        else:
            weights = np.asarray(weights, dtype=float)

        mask = np.isfinite(vals)
        if not np.any(mask):
            fused = 0.0
        else:
            fused = float(np.sum(weights[mask]*vals[mask])/(np.sum(weights[mask]) + EPS))

        return {
            "energy_time": float(Et),
            "energy_frequency": float(Ef),
            "energy_mfcc": float(Em),
            "energy_chroma": float(Ec),
            "energy_fused": fused,
        }