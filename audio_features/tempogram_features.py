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
                    continue

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
                "curve": np.array([], dtype=float),
                "variation": np.array([], dtype=float),
                "abs_variation": np.array([], dtype=float)
            }

            self._cache_tempogram[key] = result
            return result
        
        variation = l["bpm_curve"] - g["bpm"]
        result = {
            "curve": np.abs(variation).astype(float),
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

        ratio = (double_score + EPS) / (half_score + EPS)
        ratio = max(ratio, EPS)
        symmetry = 1.0 - abs(np.log(ratio)) / np.log(2.0)
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

        if mode == "phase":
            pos = phase
        elif mode == "fractional":
            pos = (t - beat_times[idx])/(beat_period + EPS)
        elif mode == "nearest":
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
        bt_hash = hash(beat_times.tobytes()) if beat_times is not None else "none"
        bf_hash = hash(tuple(beat_frames)) if beat_frames is not None else "none"
        key = f"interbeat_interval_variance_{normalize}_{bt_hash}_{bf_hash}"
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
        beat_strength = self._beat_periodicity_strength(bpm_min=bpm_min, bpm_max=bpm_max, norm_sum=norm_sum)["strength"]
        multi = self._multi_periodic_structure(bpm_min=bpm_min, bpm_max=bpm_max, norm_sum=norm_sum)["score"]
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
        multi = self._multi_periodic_structure(bpm_min=bpm_min, bpm_max=bpm_max, norm_sum=norm_sum)["score"]
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