"""
Tempogram and rhythm features.
"""

import numpy as np
import librosa
from scipy.signal import find_peaks
from scipy.ndimage import maximum_filter
from .utils import EPS, safe_clip01


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
        filtered_S = S[mask]
        filtered_bpm = bpm[mask]

        log_S = np.log1p(filtered_S)

        return log_S, filtered_bpm

    def _get_default_beats(self):
        """
        Retrieve or compute beat tracking if not cached
        """
        if hasattr(self, "beat_times") and self.beat_times is not None:
            return self.beat_times, self.beat_frames

        # Run standard librosa beat tracker as fallback
        tempo, beat_frames = librosa.beat.beat_track(
            y=self.y, 
            sr=self.sr, 
            hop_length=self.H
        )
        beat_times = librosa.frames_to_time(beat_frames, sr=self.sr, hop_length=self.H)
        self.beat_times = beat_times
        self.beat_frames = beat_frames
        self._cache_tempogram["beat_times"] = beat_times
        self._cache_tempogram["beat_frames"] = beat_frames
        return beat_times, beat_frames

    def _apply_tempo_prior(self, bpms, strengths, mu=None, sigma=None):
        """
        Deprecated compatibility helper.

        Tempo selection is intentionally data-driven in this implementation;
        no fixed preferred BPM is applied.  If callers explicitly provide a
        prior center and width, the prior is applied as a soft weighting only.
        """
        bpms = np.asarray(bpms, dtype=float)
        strengths = np.asarray(strengths, dtype=float)

        if bpms.size == 0 or strengths.size == 0 or bpms.shape != strengths.shape:
            return np.array([], dtype=float)
        if not np.any(np.isfinite(strengths)):
            return np.zeros_like(strengths, dtype=float)

        out = np.nan_to_num(strengths, nan=0.0, posinf=0.0, neginf=0.0).copy()
        if mu is None or sigma is None or sigma <= 0:
            return out

        prior = np.exp(-0.5 * ((bpms - float(mu)) / float(sigma)) ** 2)
        return out * prior

    def _correct_octave_dips(
        self,
        detected_idx,
        bpms,
        global_ac,
        bpm_min=40.0,
        bpm_max=240.0,
        fast_threshold=140.0,
        slow_threshold=75.0,
        down_ratio=0.80,
        up_ratio=0.55
    ):
        """
        Guards against octave halving (0.5x) and octave doubling (2x) 
        using asymmetric strength thresholds and tempo-dependent guards.
        """
        bpms = np.asarray(bpms, dtype=float)
        global_ac = np.asarray(global_ac, dtype=float)

        # Basic index and shape guards
        if bpms.size == 0 or global_ac.size == 0:
            return None
        if not (0 <= detected_idx < bpms.size and detected_idx < global_ac.size):
            return None

        detected_bpm = bpms[detected_idx]
        detected_strength = global_ac[detected_idx]

        best_idx = detected_idx
        best_bpm = detected_bpm
        best_strength = detected_strength

        # 1. Downward Octave Guard (Check 0.5x and 0.25x)
        # ONLY pull down if the original detected BPM is fast (> fast_threshold)
        # AND the sub-harmonic strength is over 'down_ratio' of the primary peak strength
        if detected_bpm > fast_threshold:
            for divisor in [2.0, 4.0]:
                candidate_bpm = detected_bpm/divisor
                if candidate_bpm < bpm_min:
                    continue

                candidate_idx = int(np.argmin(np.abs(bpms - candidate_bpm)))
                if not (0 <= candidate_idx < bpms.size and candidate_idx < global_ac.size):
                    continue

                candidate_strength = global_ac[candidate_idx]

                if candidate_strength > down_ratio*detected_strength:
                    best_bpm = candidate_bpm
                    best_idx = candidate_idx
                    best_strength = candidate_strength
                    break

        # 2. Upward Octave Guard (Check 2x and 3x)
        # If detected tempo dropped into slow range (< slow_threshold), check for double-time harmonic.
        elif detected_bpm < slow_threshold:
            for multiplier in [2.0, 3.0]:
                candidate_bpm = detected_bpm*multiplier
                if candidate_bpm > bpm_max:
                    continue

                candidate_idx = int(np.argmin(np.abs(bpms - candidate_bpm)))
                if not (0 <= candidate_idx < bpms.size and candidate_idx < global_ac.size):
                    continue

                candidate_strength = global_ac[candidate_idx]

                # If double-time harmonic has at least 'up-ratio' of the slow peak strength, promote to 2x/3x
                if candidate_strength > up_ratio*detected_strength:
                    best_bpm = candidate_bpm
                    best_idx = candidate_idx
                    best_strength = candidate_strength
                    break

        return int(best_idx), float(best_bpm), float(best_strength)

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
            result = {
                'tempogram': np.zeros((win_length, 0), dtype=float),
                'bpm': np.zeros(win_length, dtype=float),
                'times': np.array([], dtype=float),
            }
            self._cache_tempogram[key] = result
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
        """
        Estimate a single global BPM from the autocorrelation tempogram.

        Returns:
            dict with keys:
                - "bpm": float (chosen BPM)
                - "lag": int (lag index on bpm_axis)
                - "strength": float (global_ac value at that lag)
                - "bpm_axis": 1D array of BPMs (full axis)
                - "global_ac": 1D array (global periodicity curve over lags)
        """
        key = f"global_bpm_{bpm_min}_{bpm_max}_{norm_sum}"
        if key in self._cache_tempogram:
            return self._cache_tempogram[key]

        ac = self._tempogram_autocorr(norm_sum=norm_sum)
        tg = ac["tempogram"]
        bpm = ac["bpm"]

        # Empty tempogram guard
        if tg.size == 0 or bpm.size == 0:
            result = {
                "bpm": 0.0,
                "lag": 0,
                "strength": 0.0,
                "bpm_axis": bpm,
                "global_ac": np.array([], dtype=float)
            }
            self._cache_tempogram[key] = result
            return result

        # Mask to valid BPM range and finite values
        mask = (bpm >= bpm_min) & (bpm <= bpm_max) & np.isfinite(bpm)
        if not np.any(mask):
            result = {
                "bpm": 0.0,
                "lag": 0,
                "strength": 0.0,
                "bpm_axis": bpm,
                "global_ac": np.array([], dtype=float)
            }
            self._cache_tempogram[key] = result
            return result

        # Global periodicity curve (average over time)
        global_ac = np.mean(tg, axis=1)

        # Slice to valid search space
        search_strengths = global_ac[mask]
        search_bpms = bpm[mask]
        idxs = np.where(mask)[0]

        # Guard: if no finite strengths, return zero result
        if not np.any(np.isfinite(search_strengths)) or search_strengths.size == 0:
            result = {
                "bpm": 0.0,
                "lag": 0,
                "strength": 0.0,
                "bpm_axis": bpm,
                "global_ac": global_ac
            }
            self._cache_tempogram[key] = result
            return result

        # 1. Find the primary peak index (no prior weighting, no octave correction)
        best_rel   = int(np.argmax(search_strengths))
        primary_idx = int(idxs[best_rel])

        # 2. Parabolic interpolation for sub-integer lag precision
        #    Refines the lag to a fractional value before converting to BPM,
        #    eliminating the quantisation error from integer FFT lags.
        refined_bpm = float(bpm[primary_idx])    # fallback to grid value

        if 0 < primary_idx < len(global_ac) - 1:
            y0 = global_ac[primary_idx - 1]
            y1 = global_ac[primary_idx]
            y2 = global_ac[primary_idx + 1]
            denom = 2.0 * y1 - y0 - y2
            if abs(denom) > EPS:
                delta = 0.5 * (y0 - y2) / denom        # sub-sample offset in [-0.5, 0.5]
                refined_lag = primary_idx + delta
                if refined_lag > 0:
                    candidate = 60.0 * self.frame_rate / refined_lag
                    if bpm_min <= candidate <= bpm_max:
                        refined_bpm = float(candidate)

        primary_bpm      = refined_bpm
        primary_strength = float(global_ac[primary_idx])

        # 3. Find secondary peak — different from primary by at least 20% BPM
        #    Gives the caller visibility into alternative tempo candidates
        #    without forcing an octave correction internally.
        secondary_bpm      = 0.0
        secondary_strength = 0.0

        sorted_rel = np.argsort(search_strengths)[::-1]
        for rel in sorted_rel[1:]:
            candidate_bpm = float(search_bpms[rel])
            if abs(candidate_bpm - primary_bpm) / (primary_bpm + EPS) > 0.20:
                secondary_bpm      = candidate_bpm
                secondary_strength = float(search_strengths[rel])
                break

        # 4. Confidence: how far primary is above the secondary candidate
        if secondary_strength > EPS:
            confidence = float(np.clip(
                (primary_strength - secondary_strength) / (primary_strength + EPS),
                0.0, 1.0
            ))
        else:
            confidence = 1.0

        result = {
            "bpm":                primary_bpm,       # interpolated, no prior, no octave forcing
            "lag":                int(primary_idx),   # original integer lag index
            "strength":           primary_strength,
            "confidence":         confidence,
            "secondary_bpm":      secondary_bpm,
            "secondary_strength": secondary_strength,
            "bpm_axis":           bpm,
            "global_ac":          global_ac,
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
        local_slice = tg[mask, :]          # shape (n_bpms_in_range, n_windows)
        n_bpms, n_windows = local_slice.shape

        if n_windows == 0 or n_bpms == 0:
            bpm_curve      = np.array([], dtype=float)
            strength_curve = np.array([], dtype=float)
        else:
            # Viterbi-style: penalise octave jumps between adjacent windows
            # Transition cost between BPM bins i and j proportional to |log2(bpm_i/bpm_j)|
            bpms_in_range = bpm_axis[idxs]

            # Greedy forward pass with continuity penalty (faster than full Viterbi)
            JUMP_PENALTY = 0.3   # fraction of strength subtracted per octave of jump
            chosen = np.zeros(n_windows, dtype=int)
            chosen[0] = int(np.argmax(local_slice[:, 0]))

            for t in range(1, n_windows):
                prev_bpm = bpms_in_range[chosen[t - 1]]
                scores = local_slice[:, t].copy()
                for b, bpm_b in enumerate(bpms_in_range):
                    octave_dist = abs(np.log2((bpm_b + EPS) / (prev_bpm + EPS)))
                    scores[b] -= JUMP_PENALTY * octave_dist
                chosen[t] = int(np.argmax(scores))

            best_idx       = idxs[chosen]
            bpm_curve      = bpm_axis[best_idx]
            strength_curve = local_slice[chosen, np.arange(n_windows)]

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
                "runner_up": 0.0
            }

            self._cache_tempogram[key] = result
            return result
        
        mask = (bpm_axis >= bpm_min) & (bpm_axis <= bpm_max) & np.isfinite(bpm_axis)

        if not np.any(mask):
            result = {
                "clarity": 0.0,
                "best_peak": 0.0,
                "runner_up": 0.0
            }

            self._cache_tempogram[key] = result
            return result

        vals = np.mean(tg[mask, :], axis=1)

        if vals.size < 3:
            best   = float(np.max(vals)) if vals.size else 0.0
            second = 0.0
            clarity = best   # no prominence possible with < 3 points
        else:
            peaks, props = find_peaks(vals, prominence=0.0)
            if peaks.size > 0:
                prominences = props["prominences"]
                best_peak_idx = peaks[np.argmax(prominences)]
                best   = float(vals[best_peak_idx])
                margin = float(np.max(prominences))
                # Clarity = absolute strength × normalised prominence
                clarity = float(best * np.clip(margin / (best + EPS), 0.0, 1.0))
                second = float(vals[peaks[np.argsort(prominences)[-2]]]) if peaks.size > 1 else 0.0
            else:
                # No prominent peaks — flat or noisy signal
                best   = float(np.max(vals))
                second = float(np.sort(vals)[-2]) if vals.size > 1 else 0.0
                clarity = 0.0

        result = {
            "clarity":   safe_clip01(clarity),
            "best_peak": best,
            "runner_up": second
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
                "stability": 0.0,
                "mean_bpm": float(np.mean(curve)) if curve.size else 0.0,
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
                "curve": np.array([], dtype=float),
                "variation": np.array([], dtype=float),
                "abs_variation": np.array([], dtype=float)
            }

            self._cache_tempogram[key] = result
            return result
        
        variation = l["bpm_curve"] - g["bpm"]
        abs_var = np.abs(variation)
        result = {
            "curve":         abs_var.astype(float),
            "variation":     variation.astype(float),
            "abs_variation": abs_var.astype(float),
            "times":         l["times"],
            "mean":          float(np.mean(abs_var))   if abs_var.size else 0.0,
            "std":           float(np.std(abs_var))    if abs_var.size else 0.0,
            "max":           float(np.max(abs_var))    if abs_var.size else 0.0,
            "p95":           float(np.percentile(abs_var, 95)) if abs_var.size else 0.0,
        }

        self._cache_tempogram[key] = result
        return result
    
    def _beat_fluctuation_density(self, bpm_min=40.0, bpm_max=240.0, norm_sum=True):
        """
        Renamed from _beat_fluctuation_rate.
        Returns mean absolute BPM change per second (a true rate).
        """
        key = f"beat_fluctuation_density_{bpm_min}_{bpm_max}_{norm_sum}"
        if key in self._cache_tempogram:
            return self._cache_tempogram[key]

        curve_res = self._local_bpm_curve(bpm_min=bpm_min, bpm_max=bpm_max, norm_sum=norm_sum)
        curve = curve_res["bpm_curve"]
        if curve.size < 2:
            self._cache_tempogram[key] = 0.0
            return 0.0

        # Mean absolute change per frame, divided by hop duration → change per second
        hop_sec = self.H / float(self.sr)
        rate = float(np.mean(np.abs(np.diff(curve))) / hop_sec)

        self._cache_tempogram[key] = rate
        return rate

    def _beat_fluctuation_rate(self, bpm_min=40.0, bpm_max=240.0, norm_sum=True):
        """Deprecated: use _beat_fluctuation_density (true rate per second)."""
        return self._beat_fluctuation_density(bpm_min=bpm_min, bpm_max=bpm_max, norm_sum=norm_sum)
    
    def _tempo_harmonicity(self, bpm_min=40.0, bpm_max=240.0, norm_sum=True):
        """
        Renamed from _multi_periodic_structure.
        Measures energy at beat-harmonic periods (half and double the primary BPM)
        relative to the primary. High score = rich sub/super-beat structure.
        """
        key = f"tempo_harmonicity_{bpm_min}_{bpm_max}_{norm_sum}"
        if key in self._cache_tempogram:
            return self._cache_tempogram[key]
        
        # Use the octave-corrected global BPM as the authoritative primary_bpm
        global_res = self._global_bpm(bpm_min=bpm_min, bpm_max=bpm_max, norm_sum=norm_sum)
        primary_bpm = global_res["bpm"]
        primary_score = global_res["strength"]

        if primary_bpm == 0.0 or global_res["global_ac"].size == 0:
            result = {
                "score": 0.0,
                "primary_bpm": 0.0,
                "half_bpm": 0.0,
                "double_bpm": 0.0,
                "primary_score": 0.0,
                "half_score": 0.0,
                "double_score": 0.0
            }
            self._cache_tempogram[key] = result
            return result

        bpm_axis = global_res["bpm_axis"]
        global_ac = global_res["global_ac"]

        half_bpm = primary_bpm/2.0
        double_bpm = primary_bpm*2.0

        def nearest_val(target):
            if bpm_axis.size == 0:
                return 0.0, 0.0
            j = int(np.argmin(np.abs(bpm_axis - target)))
            return float(global_ac[j]), float(bpm_axis[j])

        half_score, half_bpm_near = nearest_val(half_bpm)
        double_score, double_bpm_near = nearest_val(double_bpm)

        # Compute multi-periodicity score relative to the corrected primary_bpm
        score = (half_score + double_score)/(primary_score + EPS)

        result = {
            "harmonicity_score": float(np.clip(score, 0.0, 2.0)),  # was "score"
            "score":             float(np.clip(score, 0.0, 2.0)),  # keep for compatibility
            "primary_bpm":       primary_bpm,
            "half_bpm":          half_bpm_near,
            "double_bpm":        double_bpm_near,
            "primary_strength":  primary_score,   # was "primary_score"
            "half_strength":     half_score,      # was "half_score"
            "double_strength":   double_score,    # was "double_score"
            # Keep old names too for backwards compat:
            "primary_score":     primary_score,
            "half_score":        half_score,
            "double_score":      double_score,
        }

        self._cache_tempogram[key] = result
        return result

    def _multi_periodic_structure(self, bpm_min=40.0, bpm_max=240.0, norm_sum=True):
        """Deprecated alias for _tempo_harmonicity."""
        return self._tempo_harmonicity(bpm_min=bpm_min, bpm_max=bpm_max, norm_sum=norm_sum)
    
    def _tempo_harmonic_ratio(self, bpm_min=40.0, bpm_max=240.0, norm_sum=True):
        """
        Renamed from _swing_ratio.
        This measures the ratio of energy at double-BPM vs half-BPM lags in
        the autocorrelation — not musical swing (which requires sub-beat timing).
        Rename communicates what it actually measures.
        """
        key = f"tempo_harmonic_ratio_{bpm_min}_{bpm_max}_{norm_sum}"
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

        ratio = (double_score + EPS) / (half_score + EPS)
        ratio = max(ratio, EPS)
        symmetry = 1.0 - abs(np.log(ratio)) / np.log(2.0)
        symmetry = safe_clip01(symmetry)

        result = {
            "ratio":       float(ratio),
            "symmetry":    symmetry,
            "primary_bpm": float(bpm_axis[best_idx]),
            "half_bpm":    float(bpm_axis[half_idx]),
            "double_bpm":  float(bpm_axis[double_idx]),
        }

        self._cache_tempogram[key] = result
        return result 

    def _swing_ratio(self, bpm_min=40.0, bpm_max=240.0, norm_sum=True):
        """Deprecated alias for _tempo_harmonic_ratio."""
        return self._tempo_harmonic_ratio(bpm_min=bpm_min, bpm_max=bpm_max, norm_sum=norm_sum)
    
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

            if len(x) > win_length:
                x = x[:win_length]
            elif len(x) < win_length:
                x = np.pad(x, (0, win_length - len(x)), mode='constant')

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
            - "fractional": fractional position relative to the median beat period (can be >1)
            - "nearest": signed phase offset to the nearest beat (in beat-phase units)
        """
        key = f"beat_position_{mode}"
        if key in self._cache_tempogram:
            return self._cache_tempogram[key]
        
        # 1. Onset times
        onset = self._onset_strength()
        t = onset["times"]

        # 2. Resolve beat timing sources
        if beat_times is None and beat_frames is None:
            beat_times, beat_frames = self._get_default_beats()

        if beat_times is None:
            beat_times = self._beat_time_from_frames(beat_frames=beat_frames)

        beat_times = np.asarray(beat_times, dtype=float)
        if beat_times.size < 2:
            result = {
                "beat_position": np.array([], dtype=float),
                "beat_index": np.array([], dtype=int),
                "beat_period": 0.0,
            }
            self._cache_tempogram[key] = result
            return result

        # Median beat period
        beat_period = float(self._beat_period_from_beats(beat_times))

        # 3. For each onset time, find the beat interval
        idx = np.searchsorted(beat_times, t, side="right") - 1
        idx = np.clip(idx, 0, beat_times.size - 2)

        # Local interval [beat_times[idx], beat_times[idx + 1]]
        dt = beat_times[idx + 1] - beat_times[idx]
        dt[dt <= 0] = EPS

        # Phase within current interval
        phase = (t - beat_times[idx])/dt
        phase = np.mod(phase, 1.0)

        # 4. Map to requested mode
        if mode == "phase":
            pos = phase
        elif mode == "fractional":
            # Fractional position relative to median beat period
            pos = (t - beat_times[idx])/(beat_period + EPS)
        elif mode == "nearest":
            # Signed phase offset to current beat, wrapped to [-0.5, 0.5)
            pos = np.where(phase > 0.5, phase - 1.0, phase)
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

        # Filter out NaNs (out-of-bounds frames)
        valid_phase = phase[np.isfinite(phase)]

        if valid_phase.size == 0:
            hist = np.zeros(n_bins, dtype=float)
            result = {
                "histogram": hist,
                "bins": np.linspace(0, 1, n_bins + 1),
                "peak_bin": -1
            }
            self._cache_tempogram[key] = result
            return result
        
        hist, bins = np.histogram(valid_phase, bins=n_bins, range=(0.0, 1.0))
        hist = hist.astype(float)
        if normalize:
            hist = hist / (np.sum(hist) + EPS)

        peak_bin = int(np.argmax(hist)) if hist.size > 0 else -1

        if hist.size > 0 and np.sum(hist) > EPS:
            p = hist / (np.sum(hist) + EPS)
            entropy = float(-np.sum(p * np.log2(p + EPS)) / np.log2(n_bins))  # normalised [0,1]
            concentration = float(np.max(p))                                    # peak bin fraction
        else:
            entropy       = 1.0   # maximum uncertainty
            concentration = 0.0

        result = {
            "histogram":     hist,
            "bins":          bins,
            "peak_bin":      peak_bin,
            "entropy":       entropy,       # 1.0 = flat/random, 0.0 = all events on one beat
            "concentration": concentration, # fraction of events at the dominant beat phase
        }

        self._cache_tempogram[key] = result
        return result
    
    def _interbeat_interval_variance(self, beat_times=None, beat_frames=None, normalize=False):
        """
        Variance of inter-beat intervals, optionally normalized by mean interval
        """
        bt_hash = hash(beat_times.tobytes()) if beat_times is not None else "none"
        bf_hash = hash(tuple(beat_frames)) if beat_frames is not None else "none"
        key = f"interbeat_interval_variance_{normalize}_{bt_hash}_{bf_hash}"
        if key in self._cache_tempogram:
            return self._cache_tempogram[key]
        
        # 1. Fallback to instance/computed beat tracking if None passed
        if beat_times is None and beat_frames is None:
            beat_times, beat_frames = self._get_default_beats()
        
        if beat_times is None:
            beat_times = self._beat_time_from_frames(beat_frames)

        beat_times = np.asarray(beat_times, dtype=float)
        if beat_times.size < 3:
            self._cache_tempogram[key] = 0.0
            return 0.0
        
        ibi = np.diff(beat_times)
        if ibi.size < 2:
            self._cache_tempogram[key] = 0.0
            return 0.0

        var     = float(np.var(ibi, ddof=1))
        mean_ibi = float(np.mean(ibi))
        cv      = float(np.std(ibi, ddof=1) / (mean_ibi + EPS))   # coefficient of variation

        # normalize=True now returns CV (dimensionless, interpretable across tempos)
        result_val = cv if normalize else var
        self._cache_tempogram[key] = result_val
        return result_val
    
    def _beat_sync_offset(self, beat_times=None, beat_frames=None, event_times=None, event_frames=None, absolute=True):
        """
        Offset of events from nearest beat, averaged across all events. If absolute=True, returns mean absolute offset, otherwise returns mean signed offset (positive means event occurs after beat)
        """
        key = f"beat_sync_offset_{absolute}"
        if key in self._cache_tempogram:
            return self._cache_tempogram[key]
        
        # 1. Fallback to instance/computed beat tracking if None passed
        if beat_times is None and beat_frames is None:
            beat_times, beat_frames = self._get_default_beats()
        
        if beat_times is None:
            beat_times = self._beat_time_from_frames(beat_frames)

        if event_times is None and event_frames is not None:
            event_times = librosa.frames_to_time(np.asarray(event_frames, dtype=int), sr=self.sr, hop_length=self.H)
        elif event_times is None:
            onset = self._onset_strength()
            event_times = onset["times"]

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
        
        idx_before = np.searchsorted(beat_times, event_times, side="right") - 1
        idx_before = np.clip(idx_before, 0, beat_times.size - 1)
        idx_after  = np.clip(idx_before + 1, 0, beat_times.size - 1)

        dist_before = np.abs(event_times - beat_times[idx_before])
        dist_after  = np.abs(event_times - beat_times[idx_after])

        # Pick whichever beat is closer
        nearest_idx = np.where(dist_before <= dist_after, idx_before, idx_after)

        # Signed offset: positive = event after beat, negative = event before beat
        offset = event_times - beat_times[nearest_idx]

        # Normalise by local IBI (use preceding interval for the nearest beat)
        ibi_idx = np.clip(nearest_idx, 0, beat_times.size - 2)
        ibi = beat_times[ibi_idx + 1] - beat_times[ibi_idx]
        offset_norm = offset / (ibi + EPS)

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
    
    # Rhythm-domain feature evidence
    def _beat_periodic_energy(self, bpm_min=40.0, bpm_max=240.0, norm_sum=True):
        """Return the Fourier-tempo energy distribution in the requested BPM range."""
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
        """Average onset strength around each beat; rhythm/transient evidence, not LUFS."""
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
                beat_period = 60.0 / g["bpm"]
                beat_times = np.arange(times[0], times[-1] + beat_period, beat_period)
            else:
                beat_times = self._beat_time_from_frames(beat_frames)

        beat_times = np.asarray(beat_times, dtype=float)
        if beat_times.size == 0:
            out = np.array([], dtype=float)
            self._cache_tempogram[key] = out
            return out

        vals = np.zeros(beat_times.size, dtype=float)
        half_win = 0.5 * (
            np.median(np.diff(beat_times)) if beat_times.size > 1 else self.H / float(self.sr)
        )

        for i, bt in enumerate(beat_times):
            lo = bt - half_win
            hi = bt + half_win
            mask = (times >= lo) & (times < hi)
            if np.any(mask):
                vals[i] = float(np.mean(env[mask]))
            else:
                idx = int(np.argmin(np.abs(times - bt)))
                vals[i] = float(env[idx])

        self._cache_tempogram[key] = vals
        return vals

    def _rhythmic_drive(self, bpm_min=40.0, bpm_max=240.0, norm_sum=True):
        """
        Rhythmic regularity and pulse-strength evidence.

        This is a useful rhythm-domain input for a downstream danceability
        model, but it is not Spotify's proprietary danceability model.
        """
        key = f"rhythmic_drive_{bpm_min}_{bpm_max}_{norm_sum}"
        if key in self._cache_tempogram:
            return self._cache_tempogram[key]

        pulse = self._pulse_clarity(
            bpm_min=bpm_min, bpm_max=bpm_max, norm_sum=norm_sum
        )["clarity"]
        stab = self._tempo_stability_index(
            bpm_min=bpm_min, bpm_max=bpm_max, norm_sum=norm_sum
        )["stability"]
        beat_strength = self._beat_periodicity_strength(
            bpm_min=bpm_min, bpm_max=bpm_max, norm_sum=norm_sum
        )["strength"]
        harmonicity = self._tempo_harmonicity(
            bpm_min=bpm_min, bpm_max=bpm_max, norm_sum=norm_sum
        )["harmonicity_score"]
        var = self._beat_fluctuation_density(
            bpm_min=bpm_min, bpm_max=bpm_max, norm_sum=norm_sum
        )

        val = float(np.clip(
            0.35 * pulse +
            0.25 * stab +
            0.20 * beat_strength +
            0.10 * (1.0 - np.tanh(var / 10.0)) +
            0.10 * (1.0 - np.clip(harmonicity / 2.0, 0.0, 1.0)),
            0.0,
            1.0,
        ))

        self._cache_tempogram[key] = val
        return val

    def _danceability_tempogram(self, bpm_min=40.0, bpm_max=240.0, norm_sum=True):
        """Deprecated compatibility alias for :meth:`_rhythmic_drive`."""
        return self._rhythmic_drive(
            bpm_min=bpm_min, bpm_max=bpm_max, norm_sum=norm_sum
        )

    def _rhythmic_coherence(self, bpm_min=40.0, bpm_max=240.0, norm_sum=True):
        """
        Rhythm coherence evidence.

        This measures temporal/rhythmic organization. It is deliberately not
        exposed as valence because tempo alone does not identify emotional
        positivity/negativity.
        """
        key = f"rhythmic_coherence_{bpm_min}_{bpm_max}_{norm_sum}"
        if key in self._cache_tempogram:
            return self._cache_tempogram[key]

        clarity = self._pulse_clarity(
            bpm_min=bpm_min, bpm_max=bpm_max, norm_sum=norm_sum
        )["clarity"]
        stab = self._tempo_stability_index(
            bpm_min=bpm_min, bpm_max=bpm_max, norm_sum=norm_sum
        )["stability"]
        cent = self._tempo_spectral_centroid(
            bpm_min=bpm_min, bpm_max=bpm_max
        )["centroid"]
        bandwidth = self._tempo_bandwidth(
            bpm_min=bpm_min, bpm_max=bpm_max
        )["bandwidth"]
        skew = self._tempo_skewness(
            bpm_min=bpm_min, bpm_max=bpm_max
        )["skewness"]

        centroid_score = float(np.clip(
            (cent - bpm_min) / (bpm_max - bpm_min + EPS), 0.0, 1.0
        ))
        spread_score = float(np.clip(
            1.0 - np.tanh(bandwidth / (0.5 * (bpm_max - bpm_min) + EPS)),
            0.0,
            1.0,
        ))
        skew_score = float(np.clip(0.5 + 0.25 * np.tanh(skew), 0.0, 1.0))

        val = float(np.clip(
            0.30 * stab +
            0.25 * clarity +
            0.20 * centroid_score +
            0.15 * spread_score +
            0.10 * skew_score,
            0.0,
            1.0,
        ))

        self._cache_tempogram[key] = val
        return val

    def _valence_tempogram(self, bpm_min=40.0, bpm_max=240.0, norm_sum=True):
        """Deprecated compatibility alias for :meth:`_rhythmic_coherence`."""
        return self._rhythmic_coherence(
            bpm_min=bpm_min, bpm_max=bpm_max, norm_sum=norm_sum
        )

    def _tempo_performance_variability(self, bpm_min=40.0, bpm_max=240.0, norm_sum=True):
        """
        Tempo/performance variability evidence.

        This measures tempo fluctuation and beat-timing variability. It is not
        a detector for audience noise or Spotify's liveness feature.
        """
        key = f"tempo_performance_variability_{bpm_min}_{bpm_max}_{norm_sum}"
        if key in self._cache_tempogram:
            return self._cache_tempogram[key]

        curve_res = self._tempo_variation_curve(
            bpm_min=bpm_min, bpm_max=bpm_max, norm_sum=norm_sum
        )
        curve = curve_res["curve"]
        fluc = self._beat_fluctuation_density(
            bpm_min=bpm_min, bpm_max=bpm_max, norm_sum=norm_sum
        )

        beat_times, beat_frames = self._get_default_beats()
        ibi_cv = self._interbeat_interval_variance(
            beat_times=beat_times, beat_frames=beat_frames, normalize=True
        )

        curve_score = float(np.clip(np.mean(curve) / 20.0, 0.0, 1.0)) if curve.size else 0.0
        fluc_score = float(np.clip(np.tanh(fluc / 10.0), 0.0, 1.0))
        ibi_score = float(np.clip(np.tanh(ibi_cv), 0.0, 1.0))

        val = float(np.clip(
            0.40 * curve_score +
            0.35 * fluc_score +
            0.25 * ibi_score,
            0.0,
            1.0,
        ))

        self._cache_tempogram[key] = val
        return val

    def _liveness_tempogram(self, bpm_min=40.0, bpm_max=240.0, norm_sum=True):
        """Deprecated compatibility alias for :meth:`_tempo_performance_variability`."""
        return self._tempo_performance_variability(
            bpm_min=bpm_min, bpm_max=bpm_max, norm_sum=norm_sum
        )

    def _mode_tempogram(self, bpm_min=40.0, bpm_max=240.0, norm_sum=True):
        """
        Return no mode evidence.

        Rhythm/tempo features do not provide a reliable major/minor estimate;
        Chroma owns key and mode estimation in the feature pipeline.
        """
        key = f"mode_tempogram_{bpm_min}_{bpm_max}_{norm_sum}"
        if key in self._cache_tempogram:
            return self._cache_tempogram[key]

        out = {
            "mode": None,
            "mode_value": None,
            "score_major": None,
            "score_minor": None,
            "delta_score": None,
            "confidence": 0.0,
        }
        self._cache_tempogram[key] = out
        return out

    def _time_signature_tempogram(self, bpm_min=40.0, bpm_max=240.0, norm_sum=True):
        """
        Estimate simple meter from beat-period autocorrelation.

        The result is evidence for 3/4 versus 4/4 only. Other meters and
        compound meters require a dedicated meter model, so confidence should
        be treated as evidence strength rather than calibrated probability.
        """
        key = f"time_signature_tempogram_{bpm_min}_{bpm_max}_{norm_sum}"
        if key in self._cache_tempogram:
            return self._cache_tempogram[key]

        g = self._global_bpm(
            bpm_min=bpm_min, bpm_max=bpm_max, norm_sum=norm_sum
        )
        bpm = float(g["bpm"])
        global_ac = np.asarray(g["global_ac"], dtype=float)

        if bpm <= 0.0 or global_ac.size < 2:
            out = {
                "time_signature": None,
                "confidence": 0.0,
                "primary_bpm": bpm,
                "score_3": 0.0,
                "score_4": 0.0,
                "structure_score": 0.0,
            }
            self._cache_tempogram[key] = out
            return out

        beat_lag = self.frame_rate * 60.0 / bpm

        def ac_at_lag(lag_float):
            lo = int(np.floor(lag_float))
            hi = lo + 1
            if lo < 0 or hi >= global_ac.size:
                return 0.0
            frac = lag_float - lo
            return float((1.0 - frac) * global_ac[lo] + frac * global_ac[hi])

        score_4_half = ac_at_lag(beat_lag * 2.0)
        score_4_full = ac_at_lag(beat_lag * 4.0)
        score_3_full = ac_at_lag(beat_lag * 3.0)

        score_4 = 0.6 * score_4_half + 0.4 * score_4_full
        score_3 = score_3_full
        total = score_3 + score_4 + EPS

        if score_4 >= score_3:
            ts = 4
            winner = score_4
            runner_up = score_3
        else:
            ts = 3
            winner = score_3
            runner_up = score_4

        margin_confidence = float(np.clip(
            (winner - runner_up) / total, 0.0, 1.0
        ))
        structure_strength = float(np.clip(max(score_3, score_4), 0.0, 1.0))
        confidence = float(np.clip(
            0.5 * margin_confidence + 0.5 * structure_strength,
            0.0,
            1.0,
        ))

        out = {
            "time_signature": int(ts),
            "confidence": confidence,
            "primary_bpm": bpm,
            "score_3": float(score_3),
            "score_4": float(score_4),
            "structure_score": structure_strength,
        }

        self._cache_tempogram[key] = out
        return out

    def tempogram_domain_evidence(
        self,
        beat_times=None,
        beat_frames=None,
        bpm_min=40.0,
        bpm_max=240.0,
        norm_sum=True,
    ):
        """
        Return rhythm-domain evidence for the central fusion/model layer.

        Values named here are model inputs/proxies, not claims to reproduce
        Spotify's proprietary Audio Features implementation.
        """
        global_bpm = self._global_bpm(
            bpm_min=bpm_min, bpm_max=bpm_max, norm_sum=norm_sum
        )
        rhythmic_drive = self._rhythmic_drive(
            bpm_min=bpm_min, bpm_max=bpm_max, norm_sum=norm_sum
        )
        rhythmic_coherence = self._rhythmic_coherence(
            bpm_min=bpm_min, bpm_max=bpm_max, norm_sum=norm_sum
        )
        performance_variability = self._tempo_performance_variability(
            bpm_min=bpm_min, bpm_max=bpm_max, norm_sum=norm_sum
        )
        time_signature = self._time_signature_tempogram(
            bpm_min=bpm_min, bpm_max=bpm_max, norm_sum=norm_sum
        )
        periodicity = self._beat_periodicity_strength(
            bpm_min=bpm_min, bpm_max=bpm_max, norm_sum=norm_sum
        )
        stability = self._tempo_stability_index(
            bpm_min=bpm_min, bpm_max=bpm_max, norm_sum=norm_sum
        )

        loudness_per_beat = self._loudness_tempogram_per_beat(
            beat_times=beat_times, beat_frames=beat_frames
        )

        return {
            "tempo_bpm": float(global_bpm["bpm"]),
            "tempo_confidence": float(global_bpm.get("confidence", 0.0)),
            "tempo_strength": float(global_bpm.get("strength", 0.0)),
            "tempo_secondary_bpm": float(global_bpm.get("secondary_bpm", 0.0)),
            "tempo_secondary_strength": float(global_bpm.get("secondary_strength", 0.0)),
            "rhythmic_drive": float(rhythmic_drive),
            "rhythmic_coherence": float(rhythmic_coherence),
            "performance_variability": float(performance_variability),
            "beat_periodicity_strength": float(periodicity["strength"]),
            "tempo_stability": float(stability["stability"]),
            "time_signature": time_signature["time_signature"],
            "time_signature_confidence": float(time_signature["confidence"]),
            "time_signature_score_3": float(time_signature["score_3"]),
            "time_signature_score_4": float(time_signature["score_4"]),
            "loudness_per_beat": loudness_per_beat,
            "mode": None,
            "mode_value": None,
            "mode_confidence": 0.0,
            "valence_evidence": None,
            "liveness_evidence": None,
        }

    def spotify_audio_features(
        self,
        beat_times=None,
        beat_frames=None,
        bpm_min=40.0,
        bpm_max=240.0,
        norm_sum=True,
    ):
        """Backward-compatible alias for :meth:`tempogram_domain_evidence`."""
        return self.tempogram_domain_evidence(
            beat_times=beat_times,
            beat_frames=beat_frames,
            bpm_min=bpm_min,
            bpm_max=bpm_max,
            norm_sum=norm_sum,
        )

