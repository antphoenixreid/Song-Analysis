"""
Time domain audio features.
"""

import numpy as np
import librosa
from scipy.signal import find_peaks
from scipy.ndimage import maximum_filter
from .utils import EPS, safe_clip01, safe_median
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
        
        y_padded = np.pad(self.y, int(self.N // 2), mode="reflect")
        zcr_frames = librosa.feature.zero_crossing_rate(y_padded, frame_length=self.N, hop_length=self.H, center=False)
        zcr = np.ravel(zcr_frames).astype(float)

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

        n = min(len(zcr), len(mask))
        if n < 2 or mask.sum() < 2:
            self._cache_time[key] = 0.0
            return 0.0
        
        zcr = zcr[:n]
        mask = mask[:n]

        num_frames = n
        ac_peaks = np.zeros(num_frames, dtype=float)

        f_min = 50.0
        f_max = 400.0

        # Convert to lag range
        min_lag = int(self.sr/f_max) # smallest lag (highest pitch)
        max_lag = int(self.sr/f_min) # largest lag (lowest pitch)
        min_lag = max(min_lag, 2)

        # Use framed signal for efficiency (pad to match other routines)
        y_padded = np.pad(self.y, int(self.N//2), mode='reflect')
        frames = librosa.util.frame(y_padded, frame_length=self.N, hop_length=self.H)

        # frames shape (N, num_frames_available)
        num_avail = min(frames.shape[1], num_frames)

        for i in range(num_avail):
            if not mask[i]:
                continue

            frame = frames[:, i]

            if frame.size < 3:
                continue

            frame_zm = frame - np.mean(frame)
            ac_full = np.correlate(frame_zm, frame_zm, mode='full')
            ac = ac_full[len(ac_full)//2:]

            if ac[0] <= EPS:
                ac_peaks[i] = 0.0
                continue

            cur_max_lag = min(max_lag, len(ac) - 1)

            if cur_max_lag <= min_lag:
                ac_peaks[i] = 0.0
                continue

            ac_peaks[i] = float(np.max(ac[min_lag:cur_max_lag + 1]))

        strong_periodic = mask & (ac_peaks >= 0.5)
        moderate_periodic = mask & (ac_peaks >= 0.35)

        low_zcr = zcr <= 0.35

        voiced = strong_periodic | (moderate_periodic & low_zcr)

        voiced_frames = int(np.sum(voiced[:n]))
        active_frames = int(np.sum(mask[:n]))

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

        duration = len(self.y) / float(self.sr)
        rate = float(len(peaks)) / duration

        self._cache_time[key] = rate
        return rate

    def _transient_counts(self) -> int:
        rate = self._transient_rate()
        duration = len(self.y) / float(self.sr)

        return int(round(rate * duration))
    
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
        
        # Use librosa.util.frame for efficient framing (returns shape (frame_length, num_frames))
        y_padded = np.pad(self.y, int(self.N//2), mode='reflect')
        framed = librosa.util.frame(y_padded, frame_length=self.N, hop_length=self.H)
        frames = framed.T.astype(float) # shape (num_frames, N)

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
    
    def _spotify_instrumentalness(self) -> float:
        key = "spotify_instrumentalness"
        if key in self._cache_time:
            return self._cache_time[key]
        
        if getattr(self, "invalid", False):
            self._cache_time[key] = 0.0
            return 0.0
        
        rms = self._rms_envelope()
        if rms.size < 2:
            self._cache_time[key] = 0.0
            return 0.0
        
        silence = self._silence_ratio(db_threshold=-60.0)
        voiced = self._voiced_ratio(db_threshold=-60.0)
        unvoiced = self._unvoiced_ratio()

        attack_time = self._attack_time()
        attack_slope = self._attack_slope()
        decay_slope = self._decay_slope()
        transient_rate = self._transient_rate()
        zcr_var = self._zcr_variance()

        diffs = np.abs(np.diff(rms))
        smoothness = 1.0 - (np.mean(diffs)/(np.mean(rms) + EPS))
        smoothness = safe_clip01(smoothness)

        silence_score = safe_clip01(silence)
        nonvocal_score = safe_clip01(unvoiced)

        attack_time_score = safe_clip01(attack_time/(attack_time + 0.05))
        attack_slope_score = safe_clip01(1.0/(1.0 + abs(attack_slope)/50.0))
        decay_slope_score = safe_clip01(1.0/(1.0 + abs(decay_slope)/50.0))

        transient_score = safe_clip01(1.0 - transient_rate/(transient_rate + 3.0))
        zcr_score = safe_clip01(1.0 - zcr_var/(1.0 + zcr_var))

        w_nv = 0.22
        w_sm = 0.20
        w_ss = 0.16
        w_ats = 0.14
        w_ass = 0.10
        w_dss = 0.08
        w_ts = 0.05
        w_zcr = 0.05

        val = (w_nv*nonvocal_score + w_sm*smoothness + w_ss*silence_score + w_ats*attack_time_score + + w_ass*attack_slope_score + w_dss*decay_slope_score + w_ts*transient_score + w_zcr*zcr_score)

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
        instrumentalness = self._spotify_instrumentalness()
        time_signature = self._spotify_time_signature()

        loudness_score = safe_clip01((loudness + 60.0) / 60.0)
        energy_score = safe_clip01(energy / (energy + 0.01))
        tempo_score = safe_clip01(tempo / 200.0)
        time_signature_score = 1.0 if time_signature == 4 else 0.5 if time_signature == 3 else 0.0

        vals = np.array([
            loudness_score,
            energy_score,
            speechiness,
            acousticness,
            danceability,
            tempo_score,
            liveness,
            instrumentalness,
            time_signature_score,
        ], dtype=float)

        if weights is None:
            weights = np.array([
                0.14,
                0.14,
                0.11,
                0.11,
                0.16,
                0.13,
                0.08,
                0.09,
                0.04,
            ], dtype=float)

        weights = np.asarray(weights, dtype=float)
        if weights.size != vals.size:
            raise ValueError(f"weights must have length {vals.size}, got {weights.size}")

        fused = float(np.sum(vals * weights) / (np.sum(weights) + EPS))

        return {
            "loudness_db": loudness,
            "energy": energy,
            "speechiness": speechiness,
            "acousticness": acousticness,
            "danceability": danceability,
            "tempo_bpm": tempo,
            "liveness": liveness,
            "instrumentalness": instrumentalness,
            "time_signature": time_signature,
            "spotify_fused": safe_clip01(fused),
        }