"""
Time domain audio features.
"""

import numpy as np
import librosa
from scipy.signal import find_peaks
from .utils import EPS, safe_clip01
from audio_features.audio_signal import AudioSignal


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

        # Keep the original signal length intact. Frame-based methods below
        # apply their own padding, so duration-based statistics remain correct
        # for recordings shorter than one analysis window.
        self._fft_freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.N)

    def _global_loudness_dB(self) -> float:
        if getattr(self, "invalid", False):
            return -80.0

        key = "global_loudness_dB"
        if key in self._cache_time:
            return self._cache_time[key]
        
        rms = np.sqrt(np.mean(self.y ** 2))
        if rms < 1e-10:
            self._cache_time[key] = -80.0
            return -80.0

        loud_db = 20.0 * np.log10(rms + EPS)

        self._cache_time[key] = loud_db
        return loud_db

    # Amplitude/Loudness Features
    def _rms_envelope(self) -> np.ndarray:
        key = "rms_env"
        if key in self._cache_time:
            return self._cache_time[key]
        
        y_padded = np.pad(self.y, int(self.N//2), mode='reflect')

        frames = librosa.util.frame(y_padded, frame_length=self.N, hop_length=self.H)

        rms_env = np.sqrt(np.mean(frames**2, axis=0)) + EPS

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
        
        # Mirror pad to achieve structural alignment with the STFT framework
        y_padded = np.pad(self.y, int(self.N // 2), mode='reflect')
        frames = librosa.util.frame(y_padded, frame_length=self.N, hop_length=self.H)
        
        # Calculate peak amplitude across window frames cleanly
        peak_amp = np.max(np.abs(frames), axis=0)

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

        frame_rate = self.sr / float(self.H)
        if center:
            times = librosa.frames_to_time(
                np.arange(len(onset_env)),
                sr=self.sr,
                hop_length=self.H,
                n_fft=self.N,
            )
        else:
            times = librosa.frames_to_time(
                np.arange(len(onset_env)),
                sr=self.sr,
                hop_length=self.H,
            )

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
        rms_db = 20.0*np.log10(rms_env + EPS)

        onsets = self._onset_frames()

        if len(onsets) == 0:
            self._cache_time[key] = 0.0
            return 0.0
        
        attack_slopes = []
        for onset_frame in onsets:
            start = max(0, onset_frame - 5)
            end = min(len(rms_env), onset_frame + 20)
            segment = rms_db[start:end]

            if len(segment) < 3:
                continue

            min_val = np.min(segment)
            max_val = np.max(segment)
            val_range = max_val - min_val

            if val_range < 1e-4:
                continue

            threshold_10 = min_val + 0.1*val_range
            threshold_90 = min_val + 0.9*val_range

            idx_10 = np.where(segment >= threshold_10)[0]
            idx_90 = np.where(segment >= threshold_90)[0]

            if len(idx_10) > 0 and len(idx_90) > 0:
                t_10 = idx_10[0]
                t_90 = idx_90[0]

                # An attack rises forward over time
                if t_90 > t_10:
                    db_change = segment[t_90] - segment[t_10]
                    time_change = (t_90 - t_10)*self.H/self.sr
                    slope = db_change/(time_change + EPS)
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
    
    def _energy_iqr_ratio(self) -> float:
        """Robust relative dispersion of active RMS energy (IQR / median)."""
        key = "energy_iqr_ratio"
        if key in self._cache_time:
            return self._cache_time[key]

        rms_env = self._rms_envelope()
        mask = self._active_rms_mask(db_threshold=-60.0)
        n = min(rms_env.size, mask.size)
        active = rms_env[:n][mask[:n]]

        if active.size < 2:
            self._cache_time[key] = 0.0
            return 0.0

        q25, q75 = np.percentile(active, [25, 75])
        iqr = q75 - q25
        ratio = float(iqr / (np.median(active) + EPS))

        self._cache_time[key] = ratio
        return ratio

    def _energy_variance(self) -> float:
        """Backward-compatible alias; returns IQR/median, not statistical variance."""
        return self._energy_iqr_ratio()
    
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
        
        y_padded = np.pad(self.y, int(self.N // 2), mode="reflect")
        zcr_frames = librosa.feature.zero_crossing_rate(y_padded, frame_length=self.N, hop_length=self.H, center=False)
        zcr = np.ravel(zcr_frames).astype(float)

        self._cache_time[key] = zcr
        return zcr
    
    def _zcr_iqr_ratio(self) -> float:
        """Robust relative dispersion of active-frame ZCR (IQR / median)."""
        key = "zcr_iqr_ratio"
        if key in self._cache_time:
            return self._cache_time[key]

        zcr = self._zero_crossing_rate()
        mask = self._active_rms_mask(db_threshold=-60.0)
        n = min(zcr.size, mask.size)
        active_zcr = zcr[:n][mask[:n]]

        if active_zcr.size < 2:
            self._cache_time[key] = 0.0
            return 0.0

        q25, q75 = np.percentile(active_zcr, [25, 75])
        iqr = q75 - q25
        ratio = float(iqr / (np.median(active_zcr) + EPS))

        self._cache_time[key] = ratio
        return ratio

    def _zcr_variance(self) -> float:
        """Backward-compatible alias; returns IQR/median, not statistical variance."""
        return self._zcr_iqr_ratio()
    
    def _harmonic_ratio(self, db_threshold: float = -60.0) -> float:
        """
        Estimate the fraction of active frames with strong periodicity in a
        human-voice-like pitch range. This is a *harmonicity* measure, not a
        vocal detector; pitched instruments can also score highly.
        """
        key = f"harmonic_ratio_{db_threshold}"
        if key in self._cache_time:
            return self._cache_time[key]

        zcr = self._zero_crossing_rate()
        mask = self._active_rms_mask(db_threshold=db_threshold)
        n = min(len(zcr), len(mask))
        if n < 2 or np.sum(mask[:n]) < 2:
            self._cache_time[key] = 0.0
            return 0.0

        mask = mask[:n]

        # Approximate human vocal fundamental range. This is deliberately kept
        # as periodicity evidence only; frequency/MFCC domains should establish
        # actual vocal presence during final fusion.
        f_min = 50.0
        f_max = 400.0
        min_lag = max(int(self.sr / f_max), 2)
        max_lag = int(self.sr / f_min)

        y_padded = np.pad(self.y, int(self.N // 2), mode="reflect")
        frames = librosa.util.frame(
            y_padded, frame_length=self.N, hop_length=self.H
        )

        num_avail = min(frames.shape[1], n)
        harmonic = np.zeros(n, dtype=bool)

        for i in range(num_avail):
            if not mask[i]:
                continue

            frame = frames[:, i].astype(float)
            frame -= np.mean(frame)
            if frame.size < 3:
                continue

            ac = np.correlate(frame, frame, mode="full")
            ac = ac[ac.size // 2:]
            if ac.size <= min_lag or ac[0] <= EPS:
                continue

            cur_max_lag = min(max_lag, ac.size - 1)
            if cur_max_lag <= min_lag:
                continue

            peak = float(np.max(ac[min_lag:cur_max_lag + 1])) / (ac[0] + EPS)
            harmonic[i] = peak >= 0.5 or (peak >= 0.35 and zcr[i] <= 0.35)

        ratio = float(np.sum(harmonic)) / float(np.sum(mask) + EPS)
        self._cache_time[key] = ratio
        return ratio

    def _voiced_ratio(self, db_threshold: float = -60.0) -> float:
        """Backward-compatible alias for harmonicity; not a vocal detector."""
        return self._harmonic_ratio(db_threshold=db_threshold)

    def _unvoiced_ratio(self) -> float:
        return 1.0 - self._harmonic_ratio()
    
    def _transient_rate(self) -> float:
        key = "transient_rate"
        if key in self._cache_time:
            return self._cache_time[key]
        
        ste = self._short_time_energy()

        # Normalize STE to [0, 1] range
        ste_max = np.max(ste) + EPS
        ste_norm = ste / ste_max

        # Compute frame rate for short-time energy
        # Uses self.H if defined, otherwise defaults to self.sr
        frame_rate = float(self.sr) / float(getattr(self, 'H', 1))

        # 1. Compute robust stats on normalized energy
        median = np.median(ste_norm)
        mad = np.median(np.abs(ste_norm - median))
        
        # Adaptive prominence threshold
        min_prominence = max(0.05, 1.0 * mad)
        
        # Min distance between transients (~30ms)
        min_distance = max(1, int(0.030 * frame_rate))

        # 2. Peak picking via local prominence
        peaks, _ = find_peaks(
            ste_norm, 
            prominence=min_prominence,
            distance=min_distance
        )

        duration = len(self.y) / float(self.sr) if self.sr > 0 else 0.0
        rate = float(len(peaks)) / duration if duration > EPS else 0.0

        self._cache_time[key] = rate
        return rate

    def _transient_counts(self) -> int:
        rate = self._transient_rate()
        duration = len(self.y) / float(self.sr)

        return int(round(rate * duration))
    
    # Rhythm/Beats Features
    def _onset_times(self) -> np.ndarray:
        """
        Return onset times (seconds) derived from the canonical _onset_frames()
        so all onset-derived features use the same event set.
        """
        key = "onset_times"
        if key in self._cache_time:
            return self._cache_time[key]

        frames = self._onset_frames()  # single source of truth

        if frames.size == 0:
            self._cache_time[key] = np.array([], dtype=float)
            return self._cache_time[key]

        onset_cfg = self._onset_envelope()
        centered = onset_cfg.get("times", np.array([], dtype=float)).size > 0 and onset_cfg["times"][0] > 0.0
        if centered:
            times = librosa.frames_to_time(
                frames, sr=self.sr, hop_length=self.H, n_fft=self.N
            )
        else:
            times = librosa.frames_to_time(
                frames, sr=self.sr, hop_length=self.H
            )

        # Never let center-padding create onset times beyond the actual signal.
        duration = len(self.y) / float(self.sr)
        times = times[times < max(duration, self.H / float(self.sr))]

        self._cache_time[key] = times.astype(float)
        return self._cache_time[key]
    
    def _onset_rate(self) -> float:
        """
        Average number of onsets per second
        """
        key = "onset_rate"
        if key in self._cache_time:
            return self._cache_time[key]
        
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
        ac = ac.astype(float)
        if normalize and ac.size > 0 and ac[0] > 0:
            ac = ac/ac[0]

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
    
    def _tempo_from_onset_ac(
        self, bpm_min: float = 40.0, bpm_max: float = 240.0
    ) -> float:
        """
        Estimate global tempo (BPM) from the onset envelope.

        Uses librosa's established tempo estimator rather than imposing a
        custom 80--160 BPM preference. The search range is still constrained
        to the requested musical BPM interval.
        """
        key = f"tempo_from_onset_ac_{bpm_min}_{bpm_max}"
        if key in self._cache_time:
            return self._cache_time[key]

        onset_env = self._onset_envelope()["onset_env"]
        if onset_env.size < 3 or np.all(onset_env <= EPS):
            self._cache_time[key] = 0.0
            return 0.0

        mask = self._active_rms_mask(db_threshold=-60.0)
        if mask.size == 0 or not np.any(mask):
            self._cache_time[key] = 0.0
            return 0.0

        frame_rate = self.sr / float(self.H)
        try:
            tempo_array = librosa.feature.tempo(
                onset_envelope=onset_env,
                sr=self.sr,
                hop_length=self.H,
                start_bpm=120.0,
                max_tempo=float(bpm_max),
                aggregate=np.mean,
            )
            tempo = float(np.asarray(tempo_array).reshape(-1)[0])
        except Exception:
            # Conservative fallback: strongest autocorrelation peak within the
            # requested BPM range. This keeps the class usable across librosa
            # versions where tempo APIs differ.
            ac = self._onset_autocorrelation()
            lag_min = max(1, int(np.floor(60.0 * frame_rate / bpm_max)))
            lag_max = min(ac.size - 1, int(np.ceil(60.0 * frame_rate / bpm_min)))
            if lag_max <= lag_min:
                self._cache_time[key] = 0.0
                return 0.0

            region = ac[lag_min:lag_max + 1]
            peak_lag = lag_min + int(np.argmax(region))
            refined_lag = (
                self._parabolic_interpolation(ac, peak_lag)
                if 1 <= peak_lag < ac.size - 1
                else float(peak_lag)
            )
            tempo = 60.0 * frame_rate / max(refined_lag, 1.0)

        tempo = float(np.clip(tempo, bpm_min, bpm_max))
        self._cache_time[key] = tempo
        return tempo
    
    def _pulse_clarity_ac(self) -> float:
        """
        Pulse clarity based on the dominance of the strongest tempo-range
        autocorrelation peak over the runner-up, gated by absolute strength.
        """
        key = "pulse_clarity_ac"
        if key in self._cache_time:
            return self._cache_time[key]

        ac = self._onset_autocorrelation()
        if ac.size < 3:
            self._cache_time[key] = 0.0
            return 0.0

        frame_rate = self.sr / float(self.H)
        lag_min = max(1, int(np.floor(60.0 * frame_rate / 240.0)))
        lag_max = min(ac.size - 1, int(np.ceil(60.0 * frame_rate / 40.0)))
        if lag_max <= lag_min:
            self._cache_time[key] = 0.0
            return 0.0

        search = ac[lag_min:lag_max + 1]
        peaks, _ = find_peaks(search, prominence=max(0.01, 0.05 * float(np.std(search))))

        if peaks.size == 0:
            self._cache_time[key] = 0.0
            return 0.0

        strengths = np.asarray(search[peaks], dtype=float)
        strengths = np.sort(strengths)[::-1]
        top = float(strengths[0])
        runner_up = float(strengths[1]) if strengths.size > 1 else 0.0

        dominance = (top - runner_up) / (abs(top) + EPS)
        absolute_strength = safe_clip01((top + 1.0) / 2.0)
        clarity = float(np.clip(dominance * np.sqrt(absolute_strength), 0.0, 1.0))

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

        if win_len < 3:
            return np.array([], dtype=float)

        if hop_sec <= 0:
            return np.array([], dtype=float)

        hop_len = max(1, int(np.ceil(hop_sec * fs_env)))

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
            # Insufficient evidence is not evidence of perfect stability.
            result = {
                "tempo_var": 0.0,
                "stability_exp": 0.5,
                "stability_cv": 0.5,
                "confidence": 0.0,
            }
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

        confidence = float(np.clip(tempos.size / 4.0, 0.0, 1.0))
        result = {
            "tempo_var":     var_tempo,
            "stability_exp": stability_exp,
            "stability_cv":  stability_cv,
            "confidence":    confidence,
        }
        self._cache_time["rhythmic_stability"] = result
        return result
    
    def _onset_periodicity_entropy(self, num_bins: int = 20) -> float:
        """
        Entropy-based regularity of detected onset spacing.

        This is intentionally named *onset* periodicity because IOIs are not
        necessarily musical beat intervals.
        """
        key = f"onset_periodicity_entropy_{num_bins}"
        if key in self._cache_time:
            return self._cache_time[key]

        ioi = self._ioi_values()
        if ioi.size < 3:
            return 0.0

        # Use a broad musical IOI range; do not hard-code a 2-second upper limit.
        ioi_clipped = ioi[(ioi > 0.05) & (ioi < 4.0)]
        if ioi_clipped.size < 3:
            return 0.0

        hist, _ = np.histogram(ioi_clipped, bins=num_bins, density=False)
        total = np.sum(hist)
        if total == 0:
            return 0.0

        p = hist.astype(float) / float(total)
        p = p[p > 0.0]
        entropy = -np.sum(p * np.log2(p))
        h_max = np.log2(num_bins)
        periodicity = 1.0 - (entropy / h_max if h_max > 0 else 1.0)

        periodicity = float(np.clip(periodicity, 0.0, 1.0))
        self._cache_time[key] = periodicity
        return periodicity

    def _beat_periodicity_entropy(self, num_bins: int = 20) -> float:
        """Backward-compatible alias for onset-spacing periodicity."""
        return self._onset_periodicity_entropy(num_bins=num_bins)

    def _ac_value(self, ac: np.ndarray, lag: float) -> float:
        """Linearly interpolated autocorrelation value at a fractional lag."""
        if ac.size == 0 or lag < 0 or lag >= ac.size:
            return 0.0
        lo = int(np.floor(lag))
        hi = min(lo + 1, ac.size - 1)
        frac = float(lag - lo)
        return float((1.0 - frac) * ac[lo] + frac * ac[hi])

    def _beat_regularity_from_tempo(self, tempo_bpm: float | None = None) -> float:
        """
        Estimate beat-period regularity from onset-envelope autocorrelation
        after anchoring the lag search to the estimated tempo.
        """
        key = f"beat_regularity_{tempo_bpm}"
        if key in self._cache_time:
            return self._cache_time[key]

        if tempo_bpm is None or tempo_bpm <= 0.0:
            tempo_bpm = self._spotify_tempo()

        if tempo_bpm <= 0.0:
            self._cache_time[key] = 0.0
            return 0.0

        ac = self._onset_autocorrelation()
        if ac.size < 3:
            self._cache_time[key] = 0.0
            return 0.0

        fs_env = self.sr / float(self.H)
        beat_lag = 60.0 * fs_env / float(tempo_bpm)

        vals = []
        for multiple in (1.0, 2.0, 4.0):
            lag = beat_lag * multiple
            window = max(1, int(round(0.04 * fs_env)))
            center = int(round(lag))
            lo = max(1, center - window)
            hi = min(ac.size - 1, center + window)
            if lo <= hi:
                vals.append(float(np.max(ac[lo:hi + 1])))

        regularity = float(np.mean(vals)) if vals else 0.0
        regularity = safe_clip01((regularity + 1.0) / 2.0)
        self._cache_time[key] = regularity
        return regularity

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
    
    def _frames(self) -> dict:
        """
        Returns descriptive statistics of the framed signal rather than
        the raw array, so downstream exporters receive scalars.
        """
        key = "frames"
        if key in self._cache_time:
            return self._cache_time[key]

        L = len(self.y)
        if L < self.N:
            result = {
                "num_frames": 0,
                "mean_frame_energy": 0.0,
                "std_frame_energy": 0.0,
                "max_frame_energy": 0.0,
            }
            self._cache_time[key] = result
            return result

        y_padded = np.pad(self.y, int(self.N // 2), mode='reflect')
        framed = librosa.util.frame(y_padded, frame_length=self.N, hop_length=self.H)
        # framed shape: (frame_length, num_frames)
        frame_energies = np.sqrt(np.mean(framed ** 2, axis=0))  # RMS per frame

        result = {
            "num_frames":        int(framed.shape[1]),
            "mean_frame_energy": float(np.mean(frame_energies)),
            "std_frame_energy":  float(np.std(frame_energies)),
            "max_frame_energy":  float(np.max(frame_energies)),
        }

        self._cache_time[key] = result
        return result
    
    def _self_similarity_matrix(self) -> np.ndarray:
        """ 
        Time-domain self-similarity matrix using cosine similarity between frames.
        S[k, 1] in [-1, 1], with 1 = identical (up to scaling), 0 = orthogonal
        """
        if "self_sim" in self._cache_time:
            return self._cache_time["self_sim"]
        
        L = len(self.y)
        if L < self.N:
            self._cache_time["self_sim"] = np.zeros((0, 0), dtype=float)
            return self._cache_time["self_sim"]
        y_padded = np.pad(self.y, int(self.N // 2), mode='reflect')
        framed = librosa.util.frame(y_padded, frame_length=self.N, hop_length=self.H)
        frames = framed.T.astype(float)  # shape (num_frames, N)
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
        
        # 1. COMPUTE ON ENVELOPE (Slashes array size from millions to thousands)
        # Using librosa's root-mean-square feature acts as a massive downsampler
        rms_env = librosa.feature.rms(y=self.y, frame_length=self.N, hop_length=self.H)[0]
        
        n = len(rms_env)
        if n < 2:
            self._cache_time["lz_complexity"] = 0.0
            return 0.0
        
        # 2. Binary symbolic sequence quantization around the median
        thr = np.median(rms_env)
        s = (rms_env > thr).astype(int)
        
        # Pack into a standard Python string for fast matching
        seq = "".join(s.astype(str))
        
        # 3. Optimized LZ76 parsing loop
        i = 0
        c = 1
        k = 1
        while i + k <= n:
            sub = seq[i:i + k]
            # Check if the substring exists anywhere in the sequence examined so far
            if seq[:i + k - 1].find(sub) != -1:
                k += 1
            else:
                c += 1
                i += k
                k = 1

        # 4. Normalize by the envelope length n
        c = float(c)
        if n > 1:
            c_norm = c / (n / np.log(n))
        else:
            c_norm = 0.0

        c_norm = float(max(0.0, min(1.0, c_norm)))
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
        
        # 1. COMPUTE ON ENVELOPE (Reduces array size from millions to thousands)
        rms_env = librosa.feature.rms(y=self.y, frame_length=self.N, hop_length=self.H)[0]
        
        x = rms_env.astype(float)
        n = len(x)  # Frame length (~10,000 instead of 5,000,000+)
        
        if n < 2 or k_max < 2:
            self._cache_time[key] = 1.0
            return 1.0
        
        k_max = min(k_max, n - 1)
        Lk = []
        ln_k = []
        
        # 2. Algorithmic loops now process safely sized arrays
        for k in range(1, k_max + 1):
            Lm = []
            for m in range(k):
                idxs = np.arange(m, n, k)
                if idxs.size < 2:
                    continue
                
                x_m = x[idxs]
                diff = np.abs(np.diff(x_m)).sum()
                n_m = idxs.size
                
                # Higuchi normalization length formula
                L_mk = (diff * (n - 1) / ((n_m - 1) * k)) / k
                Lm.append(L_mk)
            
            if len(Lm) == 0:
                continue
            
            Lk.append(np.mean(Lm))
            ln_k.append(np.log(1.0 * k))
        
        Lk = np.array(Lk, dtype=float)
        ln_k = np.array(ln_k, dtype=float)
        
        if Lk.size < 2 or np.any(Lk <= 0):
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
    
    # Time-domain feature evidence
    #
    # These methods intentionally produce domain-specific evidence for the
    # central fusion layer. They do not reproduce Spotify's proprietary
    # Audio Features model or scoring functions.

    def _time_loudness_db(self, active_only: bool = True) -> float:
        """Return an RMS-based dBFS loudness proxy for time-domain evidence."""
        key = f"time_loudness_db_{active_only}"
        if key in self._cache_time:
            return self._cache_time[key]

        if getattr(self, "invalid", False):
            self._cache_time[key] = -80.0
            return -80.0

        if self.y.size == 0:
            self._cache_time[key] = -80.0
            return -80.0

        if active_only:
            rms = self._rms_envelope()
            mask = self._active_rms_mask(db_threshold=-60.0)
            n = min(rms.size, mask.size)
            active = rms[:n][mask[:n]]
            if active.size == 0:
                value = -80.0
            else:
                rms_active = float(np.sqrt(np.mean(active ** 2)))
                value = 20.0 * np.log10(max(rms_active, EPS))
        else:
            rms_value = float(np.sqrt(np.mean(self.y.astype(float) ** 2)))
            value = 20.0 * np.log10(max(rms_value, EPS)) if rms_value > 0.0 else -80.0

        value = float(np.clip(value, -80.0, 0.0))
        self._cache_time[key] = value
        return value

    def _time_energy(self, active_only: bool = True) -> float:
        """
        Return a track-relative time-domain activity/energy score in [0, 1].

        This is intentionally relative to the recording's own RMS distribution;
        it is not an absolute or perceptual Spotify energy estimate.
        """
        key = f"time_energy_{active_only}"
        if key in self._cache_time:
            return self._cache_time[key]

        if getattr(self, "invalid", False):
            self._cache_time[key] = 0.0
            return 0.0

        rms = self._rms_envelope()
        if rms.size == 0:
            self._cache_time[key] = 0.0
            return 0.0

        if active_only:
            mask = self._active_rms_mask(db_threshold=-60.0)
            n = min(rms.size, mask.size)
            active = rms[:n][mask[:n]]
            if active.size == 0:
                self._cache_time[key] = 0.0
                return 0.0
            level = float(np.median(active))
            reference = active
        else:
            level = float(np.median(rms))
            reference = rms

        q10, q90 = np.percentile(reference, [10.0, 90.0])
        denom = max(float(q90 - q10), EPS)
        level_score = safe_clip01((level - q10) / denom)

        onset_rate_score = safe_clip01(self._onset_rate() / 8.0)

        crest = self._crest_factor()
        mask = self._active_rms_mask(db_threshold=-60.0)
        n = min(crest.size, mask.size)
        active_crest = crest[:n][mask[:n]]
        crest_value = float(np.median(active_crest)) if active_crest.size else 1.0
        crest_score = safe_clip01((crest_value - 1.0) / 6.0)

        value = safe_clip01(
            0.65 * level_score +
            0.20 * onset_rate_score +
            0.15 * crest_score
        )
        self._cache_time[key] = value
        return value

    def _time_speechiness_evidence(self) -> float:
        """
        Return weak time-domain speech-like evidence in [0, 1].

        ZCR variability and non-harmonicity are supporting signals only; this
        method must not be treated as a vocal/speech classifier.
        """
        key = "time_speechiness_evidence"
        if key in self._cache_time:
            return self._cache_time[key]

        if getattr(self, "invalid", False):
            self._cache_time[key] = 0.0
            return 0.0

        mask = self._active_rms_mask(db_threshold=-60.0)
        if mask.size == 0 or not np.any(mask):
            self._cache_time[key] = 0.0
            return 0.0

        zcr_disp = self._zcr_iqr_ratio()
        harmonic_ratio = self._harmonic_ratio()

        zcr_score = safe_clip01(zcr_disp / 1.5)
        nonharmonic_score = safe_clip01(1.0 - harmonic_ratio)

        value = safe_clip01(
            0.65 * zcr_score +
            0.35 * nonharmonic_score
        )
        self._cache_time[key] = value
        return value

    def _time_acousticness_evidence(self) -> float:
        """
        Return weak envelope/timbre smoothness evidence in [0, 1].

        This is not acousticness classification. Frequency and MFCC domains
        should carry most of the acousticness evidence.
        """
        key = "time_acousticness_evidence"
        if key in self._cache_time:
            return self._cache_time[key]

        if getattr(self, "invalid", False):
            self._cache_time[key] = 0.0
            return 0.0

        mask = self._active_rms_mask(db_threshold=-60.0)
        if mask.size == 0 or not np.any(mask):
            self._cache_time[key] = 0.0
            return 0.0

        rms = self._rms_envelope()
        if rms.size < 2:
            self._cache_time[key] = 0.0
            return 0.0

        diffs = np.abs(np.diff(rms))
        smoothness = 1.0 - float(np.mean(diffs)) / (float(np.mean(rms)) + EPS)
        smooth_score = safe_clip01(smoothness)

        attack_time = self._attack_time()
        decay_slope = self._decay_slope()
        silence = self._silence_ratio(db_threshold=-60.0)

        attack_score = safe_clip01(attack_time / (attack_time + 0.05))
        decay_score = safe_clip01(1.0 / (1.0 + abs(decay_slope) / 60.0))
        silence_score = safe_clip01(silence)

        value = safe_clip01(
            0.45 * smooth_score +
            0.25 * attack_score +
            0.20 * decay_score +
            0.10 * silence_score
        )
        self._cache_time[key] = value
        return value

    def _time_danceability_evidence(self, tempo_bpm: float | None = None) -> float:
        """
        Return time-domain rhythmic/danceability evidence in [0, 1].

        Tempogram analysis should be the primary source for the final
        danceability estimate; this method is a supporting time-domain signal.
        """
        key = f"time_danceability_evidence_{tempo_bpm}"
        if key in self._cache_time:
            return self._cache_time[key]

        if getattr(self, "invalid", False):
            self._cache_time[key] = 0.0
            return 0.0

        mask = self._active_rms_mask(db_threshold=-60.0)
        if mask.size == 0 or not np.any(mask):
            self._cache_time[key] = 0.0
            return 0.0

        if tempo_bpm is None or tempo_bpm <= 0.0:
            tempo_bpm = self._time_tempo_bpm()

        pulse = self._pulse_clarity_ac()
        beat_regularity = self._beat_regularity_from_tempo(tempo_bpm)
        stability = self._rhythmic_stability()
        onset_rate = self._onset_rate()

        stability_score = safe_clip01(
            0.5 * stability.get("stability_exp", 0.5) +
            0.5 * stability.get("stability_cv", 0.5)
        )
        onset_score = safe_clip01(onset_rate / 8.0)

        value = safe_clip01(
            0.35 * pulse +
            0.30 * beat_regularity +
            0.20 * stability_score +
            0.10 * onset_score +
            0.05 * safe_clip01(stability.get("confidence", 0.0))
        )
        self._cache_time[key] = value
        return value

    def _time_tempo_bpm(
        self,
        bpm_min: float = 40.0,
        bpm_max: float = 240.0,
    ) -> float:
        """Return the time-domain BPM estimate without claiming Spotify parity."""
        key = f"time_tempo_bpm_{bpm_min}_{bpm_max}"
        if key in self._cache_time:
            return self._cache_time[key]

        if getattr(self, "invalid", False):
            self._cache_time[key] = 0.0
            return 0.0

        tempo = self._tempo_from_onset_ac(bpm_min=bpm_min, bpm_max=bpm_max)
        if tempo <= 0.0:
            self._cache_time[key] = 0.0
            return 0.0

        value = float(np.clip(tempo, bpm_min, bpm_max))
        self._cache_time[key] = value
        return value

    def _time_tempo_confidence(self, tempo_bpm: float | None = None) -> float:
        """Estimate confidence in the time-domain tempo evidence."""
        key = f"time_tempo_confidence_{tempo_bpm}"
        if key in self._cache_time:
            return self._cache_time[key]

        if getattr(self, "invalid", False):
            self._cache_time[key] = 0.0
            return 0.0

        if tempo_bpm is None or tempo_bpm <= 0.0:
            tempo_bpm = self._time_tempo_bpm()
        if tempo_bpm <= 0.0:
            self._cache_time[key] = 0.0
            return 0.0

        onset_env = self._onset_envelope()["onset_env"]
        if onset_env.size < 8 or np.max(onset_env) <= EPS:
            self._cache_time[key] = 0.0
            return 0.0

        pulse = self._pulse_clarity_ac()
        stability = self._rhythmic_stability()
        sample_conf = safe_clip01(stability.get("confidence", 0.0))

        value = safe_clip01(0.70 * pulse + 0.30 * sample_conf)
        self._cache_time[key] = value
        return value

    def _time_performance_variability(self) -> float:
        """
        Return performance/dynamic variability evidence in [0, 1].

        This should not be interpreted as direct live-audience detection.
        """
        key = "time_performance_variability"
        if key in self._cache_time:
            return self._cache_time[key]

        if getattr(self, "invalid", False):
            self._cache_time[key] = 0.0
            return 0.0

        mask = self._active_rms_mask(db_threshold=-60.0)
        if mask.size == 0 or not np.any(mask):
            self._cache_time[key] = 0.0
            return 0.0

        transient_rate = self._transient_rate()
        attack_time = self._attack_time()
        attack_slope = self._attack_slope()
        decay_slope = self._decay_slope()
        stability = self._rhythmic_stability()
        zcr_disp = self._zcr_iqr_ratio()

        transient_score = safe_clip01(transient_rate / (transient_rate + 4.0))
        attack_irregularity = safe_clip01(1.0 - attack_time / (attack_time + 0.12))
        dynamic_attack = safe_clip01(abs(attack_slope) / (abs(attack_slope) + 30.0))
        dynamic_decay = safe_clip01(abs(decay_slope) / (abs(decay_slope) + 30.0))
        tempo_var_score = safe_clip01(
            stability.get("tempo_var", 0.0) /
            (stability.get("tempo_var", 0.0) + 10.0)
        )
        zcr_score = safe_clip01(zcr_disp / (1.0 + zcr_disp))

        value = safe_clip01(
            0.25 * transient_score +
            0.20 * attack_irregularity +
            0.20 * dynamic_attack +
            0.15 * dynamic_decay +
            0.10 * tempo_var_score +
            0.10 * zcr_score
        )
        self._cache_time[key] = value
        return value

    def _time_instrumentalness_evidence(self) -> float:
        """
        Return weak time-domain non-vocal evidence in [0, 1].

        This is only a prior. Harmonicity is not treated as proof of
        instrumental content because vocals can also be strongly harmonic.
        """
        key = "time_instrumentalness_evidence"
        if key in self._cache_time:
            return self._cache_time[key]

        if getattr(self, "invalid", False):
            self._cache_time[key] = 0.0
            return 0.0

        mask = self._active_rms_mask(db_threshold=-60.0)
        if mask.size == 0 or not np.any(mask):
            self._cache_time[key] = 0.0
            return 0.0

        speech_proxy = self._time_speechiness_evidence()
        harmonic_ratio = self._harmonic_ratio()
        transient_rate = self._transient_rate()
        zcr_disp = self._zcr_iqr_ratio()

        non_speech = safe_clip01(1.0 - speech_proxy)
        stable_timbre = safe_clip01(1.0 - zcr_disp / (1.0 + zcr_disp))
        harmonicity = safe_clip01(harmonic_ratio)
        sparse_transients = safe_clip01(
            1.0 - transient_rate / (transient_rate + 6.0)
        )

        value = safe_clip01(
            0.45 * non_speech +
            0.20 * stable_timbre +
            0.20 * harmonicity +
            0.15 * sparse_transients
        )
        self._cache_time[key] = value
        return value

    def _time_signature_evidence(
        self,
        tempo_bpm: float | None = None,
    ) -> dict:
        """
        Estimate meter evidence from beat-synchronous onset structure.

        Returns a candidate meter plus confidence. A fallback of 4 is retained
        for API compatibility, but its confidence is zero when the evidence is
        insufficient, so the fusion layer can ignore it.
        """
        key = f"time_signature_evidence_{tempo_bpm}"
        if key in self._cache_time:
            return self._cache_time[key]

        neutral = {
            "time_signature": 4,
            "confidence": 0.0,
            "scores": {},
        }

        if getattr(self, "invalid", False):
            self._cache_time[key] = neutral
            return neutral

        if tempo_bpm is None or tempo_bpm <= 0.0:
            tempo_bpm = self._time_tempo_bpm()
        if tempo_bpm <= 0.0:
            self._cache_time[key] = neutral
            return neutral

        onset_env = self._onset_envelope()["onset_env"]
        if onset_env.size < 8:
            self._cache_time[key] = neutral
            return neutral

        fs_env = self.sr / float(self.H)
        beat_lag = 60.0 * fs_env / float(tempo_bpm)
        if beat_lag < 1.0:
            self._cache_time[key] = neutral
            return neutral

        beat_positions = np.arange(0.0, len(onset_env), beat_lag)
        beat_strength = np.interp(
            beat_positions,
            np.arange(len(onset_env)),
            onset_env,
        )
        if beat_strength.size < 12:
            self._cache_time[key] = neutral
            return neutral

        beat_strength = beat_strength.astype(float)
        q10, q90 = np.percentile(beat_strength, [10.0, 90.0])
        scale = float(q90 - q10)
        if scale > EPS:
            beat_strength = (beat_strength - q10) / scale
        else:
            beat_strength = np.zeros_like(beat_strength)

        scores = {}
        for meter in range(3, 8):
            n_measures = beat_strength.size // meter
            if n_measures < 4:
                scores[meter] = 0.0
                continue

            usable = beat_strength[:n_measures * meter].reshape(n_measures, meter)

            centered = usable - usable.mean(axis=1, keepdims=True)
            numer = float(np.sum(centered[:-1] * centered[1:]))
            denom = float(
                np.sqrt(
                    np.sum(centered[:-1] ** 2) *
                    np.sum(centered[1:] ** 2)
                ) + EPS
            )
            pattern_repeat = float(np.clip(numer / denom, -1.0, 1.0))
            pattern_repeat = safe_clip01((pattern_repeat + 1.0) / 2.0)

            phase_scores = []
            for phase in range(meter):
                rotated = np.roll(usable, -phase, axis=1)
                first = float(np.mean(rotated[:, 0]))
                rest = float(np.mean(rotated[:, 1:])) if meter > 1 else 0.0
                accent_ratio = (first - rest) / (first + rest + EPS)
                phase_scores.append(safe_clip01(accent_ratio))

            accent_score = max(phase_scores) if phase_scores else 0.5
            scores[meter] = 0.75 * pattern_repeat + 0.25 * accent_score

        finite_scores = {
            meter: score
            for meter, score in scores.items()
            if np.isfinite(score)
        }
        if not finite_scores:
            self._cache_time[key] = neutral
            return neutral

        ranked = sorted(
            finite_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        best_meter, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        score4 = float(finite_scores.get(4, 0.0))

        # Preserve the existing conservative preference for 4/4 when the
        # candidates are nearly tied, but make that uncertainty explicit.
        if best_meter != 4 and best_score - score4 < 0.05:
            meter = 4
            separation = max(0.0, score4 - second_score)
        else:
            meter = int(best_meter)
            separation = max(0.0, best_score - second_score)

        structure_strength = safe_clip01(float(best_score))
        separation_conf = safe_clip01(
            separation / (abs(float(best_score)) + EPS)
        )
        sample_conf = safe_clip01(
            beat_strength.size / max(16.0, 16.0 * 8.0)
        )
        confidence = safe_clip01(
            0.50 * structure_strength +
            0.35 * separation_conf +
            0.15 * sample_conf
        )

        result = {
            "time_signature": meter,
            "confidence": float(confidence),
            "scores": {int(k): float(v) for k, v in finite_scores.items()},
        }
        self._cache_time[key] = result
        return result

    def time_domain_evidence(self) -> dict:
        """
        Return time-domain evidence for central cross-domain fusion.

        Keys are explicitly domain-scoped. They are not claims of Spotify's
        proprietary Audio Features values.
        """
        tempo = self._time_tempo_bpm()
        tempo_confidence = self._time_tempo_confidence(tempo_bpm=tempo)
        time_signature = self._time_signature_evidence(tempo_bpm=tempo)

        return {
            "loudness_db_time": self._time_loudness_db(active_only=True),
            "energy_time": self._time_energy(active_only=True),
            "speechiness_time": self._time_speechiness_evidence(),
            "acousticness_time": self._time_acousticness_evidence(),
            "danceability_time": self._time_danceability_evidence(tempo_bpm=tempo),
            "tempo_time_bpm": tempo,
            "tempo_confidence_time": tempo_confidence,
            "performance_variability_time": self._time_performance_variability(),
            "instrumentalness_time": self._time_instrumentalness_evidence(),
            "time_signature_time": int(time_signature["time_signature"]),
            "time_signature_confidence_time": float(time_signature["confidence"]),
        }

    def time_domain_spotify_features(self) -> dict:
        """
        Backward-compatible public alias.

        The returned values are still domain-specific evidence and must be
        fused centrally before producing a final Spotify-style feature vector.
        """
        return self.time_domain_evidence()

    def spotify_audio_features(self, weights=None) -> dict:
        """
        Backward-compatible alias for callers using the previous API.

        ``weights`` is intentionally ignored. Cross-domain weighting belongs in
        the central fusion layer.
        """
        del weights
        return self.time_domain_evidence()

    # Backward-compatible method aliases. These retain the old callable API
    # while making the implementation semantics explicit above.
    def _spotify_loudness(self, active_only: bool = True) -> float:
        return self._time_loudness_db(active_only=active_only)

    def _spotify_energy(self, active_only: bool = True) -> float:
        return self._time_energy(active_only=active_only)

    def _spotify_speechiness(self) -> float:
        return self._time_speechiness_evidence()

    def _spotify_acousticness(self) -> float:
        return self._time_acousticness_evidence()

    def _spotify_danceability(self) -> float:
        tempo = self._time_tempo_bpm()
        return self._time_danceability_evidence(tempo_bpm=tempo)

    def _spotify_tempo(self) -> float:
        return self._time_tempo_bpm()

    def _spotify_liveness(self) -> float:
        return self._time_performance_variability()

    def _spotify_instrumentalness(self) -> float:
        return self._time_instrumentalness_evidence()

    def _spotify_time_signature(self, tempo_bpm: float | None = None) -> int:
        return int(self._time_signature_evidence(tempo_bpm=tempo_bpm)["time_signature"])

