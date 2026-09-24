"""
MFCC (Mel-Frequency Cepstral Coefficients) features.
"""

import numpy as np
import librosa
from .utils import EPS, safe_clip01

VAR_FLOOR = 1e-4   # minimum variance before skewness is computed

class MFCCFeatures:
    def __init__(self, sig, n_mfcc=13, n_mels=40, n_fft=None, hop_length=None, fmin=0.0, fmax=None, dct_type=2, norm="ortho", lifter=0, htk=False, center=True, pad_mode="constant", log_mels=False, power=2.0, dtype=np.float32, compute=True):
        self.sig = sig
        self.y = np.asarray(sig.y, dtype=float)
        self.sr = sig.sr
        self.N = int(n_fft if n_fft is not None else sig.N)
        self.H = int(hop_length if hop_length is not None else sig.H)
        self.frame_rate = self.sr / self.H

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
        key = "mfcc_default"
        if key in self._cache_mfcc:
            return self._cache_mfcc[key]

        # 1. Compute Mel Spectrogram (Linear Power)
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

        # 2. Convert to dB scale for local inspection/caching
        # Pass ref=np.max if relative peak normalization is desired, or ref=1.0
        self.S_db = librosa.power_to_db(self.S, ref=np.max)

        # 3. Compute MFCC cleanly from LINEAR power S (Librosa applies power_to_db internally)
        self.mfcc = librosa.feature.mfcc(
            S=librosa.power_to_db(self.S, ref=1.0), # OR pass linear S directly if dct handling requires power
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
    
    def _mfcc_variance(self, ddof=0):
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
        skew = m3 / (np.maximum(m2, VAR_FLOOR)**1.5 + EPS)

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
        kurt = m4 / (np.maximum(m2, VAR_FLOOR)**2 + EPS)

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
        
        sigma = np.std(self.mfcc, axis=1, ddof=ddof)
        rms   = np.sqrt(np.mean(self.mfcc**2, axis=1))    # RMS per coefficient
        cv    = sigma / (rms + EPS)                         # dimensionless, scale-invariant
        stability = 1.0 / (1.0 + cv)

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
                r /= (r[len(r)//2] + EPS)

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
    def _mfcc_spectral_slope_proxy(self, start_coeff=1, aggregate="mean"):
        """
        Slope of the mean MFCC coefficient profile across coefficients 1+.
        Negative slope = energy concentrated in low coefficients (smooth timbre).
        Positive slope = energy rising toward high coefficients (complex/noisy timbre).
        coeff=0 is excluded: it tracks log-energy, not timbral shape.
        """
        key = f"mfcc_spectral_slope_proxy_start_{start_coeff}_agg_{aggregate}"
        if key in self._cache_mfcc:
            return self._cache_mfcc[key]

        if self.mfcc is None or self.mfcc.size == 0:
            self._cache_mfcc[key] = 0.0
            return 0.0

        # Profile: mean absolute value per coefficient (energy proxy per coeff)
        profile = np.mean(np.abs(self.mfcc[start_coeff:, :]), axis=1)  # shape (n_coeffs-1,)
        n = profile.size
        if n < 2:
            self._cache_mfcc[key] = 0.0
            return 0.0

        # Linear regression slope via least-squares
        x = np.arange(n, dtype=float)
        x -= x.mean()
        slope = float(np.dot(x, profile) / (np.dot(x, x) + EPS))

        self._cache_mfcc[key] = slope
        return slope
    
    def _mfcc_brightness_proxy(self, start_coeff=1, invert=False):
        """
        MFCC shape centroid: energy-weighted center of mass across coefficients 1+.
        High centroid = energy concentrated in high-order coefficients (bright/complex).
        Low centroid  = energy concentrated in low-order coefficients (dark/smooth).
        """
        key = f"mfcc_shape_centroid_start_{start_coeff}_invert_{invert}"
        if key in self._cache_mfcc:
            return self._cache_mfcc[key]

        if self.mfcc is None or self.mfcc.size == 0:
            self._cache_mfcc[key] = 0.0
            return 0.0

        profile = np.mean(np.abs(self.mfcc[start_coeff:, :]), axis=1)
        n = profile.size
        if n == 0 or np.sum(profile) < EPS:
            self._cache_mfcc[key] = 0.0
            return 0.0

        coeff_idx = np.arange(start_coeff, start_coeff + n, dtype=float)
        centroid = float(np.dot(coeff_idx, profile) / (np.sum(profile) + EPS))
        if invert:
            centroid = -centroid

        self._cache_mfcc[key] = centroid
        return centroid
    
    def _mfcc_sharpness_proxy(self, start_coeff=1, aggregate="mean", absolute=True):
        """
        MFCC shape spread: weighted standard deviation of energy across coefficients 1+.
        High spread = energy distributed across many coefficients (complex timbre).
        Low spread  = energy concentrated in few coefficients (focused/tonal timbre).
        """
        key = f"mfcc_shape_spread_start_{start_coeff}"
        if key in self._cache_mfcc:
            return self._cache_mfcc[key]

        if self.mfcc is None or self.mfcc.size == 0:
            self._cache_mfcc[key] = 0.0
            return 0.0

        profile = np.mean(np.abs(self.mfcc[start_coeff:, :]), axis=1)
        n = profile.size
        if n < 2 or np.sum(profile) < EPS:
            self._cache_mfcc[key] = 0.0
            return 0.0

        coeff_idx = np.arange(start_coeff, start_coeff + n, dtype=float)
        centroid = float(np.dot(coeff_idx, profile) / (np.sum(profile) + EPS))
        spread = float(np.sqrt(np.dot(profile, (coeff_idx - centroid)**2) / (np.sum(profile) + EPS)))

        self._cache_mfcc[key] = spread
        return spread
    
    def _mfcc_high_order_energy_ratio(self, start_coeff=6, normalize=False, order='l2'):
        """
        Renamed from _mfcc_high_order_energy.
        When normalize=True returns the ratio of high-order to total energy.
        """
        key = f"mfcc_high_order_energy_ratio_start_{start_coeff}_normalize_{normalize}_order_{order}"
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
            if order == 'l2':
                denom = float(np.sum(self.mfcc**2) + EPS)
            elif order == 'l1':
                denom = float(np.sum(np.abs(self.mfcc)) + EPS)
            else:
                denom = float(np.sqrt(np.mean(self.mfcc**2)) + EPS)
            energy /= denom

        self._cache_mfcc[key] = energy
        return energy

    def _mfcc_high_order_energy(self, start_coeff=6, normalize=False, order='l2'):
        """Deprecated alias for _mfcc_high_order_energy_ratio."""
        return self._mfcc_high_order_energy_ratio(start_coeff=start_coeff,
                                                normalize=normalize, order=order)
    
    def _mfcc_timbre_complexity(self, start_coeff=6, normalize=True):
        """
        Renamed from _mfcc_noise_inharmonicity_proxy.
        Measures the relative energy in high-order MFCC coefficients —
        a proxy for timbral complexity, not acoustic inharmonicity.
        """
        key = f"mfcc_timbre_complexity_start_{start_coeff}_normalize_{normalize}"
        if key in self._cache_mfcc:
            return self._cache_mfcc[key]

        result = self._mfcc_high_order_energy_ratio(start_coeff=start_coeff,
                                                    normalize=normalize, order='l2')
        self._cache_mfcc[key] = result
        return result

    def _mfcc_noise_inharmonicity_proxy(self, coeff=0, normalize=True):
        """Deprecated alias for _mfcc_timbre_complexity."""
        return self._mfcc_timbre_complexity(start_coeff=coeff, normalize=normalize)

    # Envelope Features
    def _mfcc_attack_smoothness(self, attack_frames=None, normalize=True):
        """
        Delta-based attack smoothness: low first-order delta variance in the
        attack region = smooth attack; high = abrupt/percussive attack.
        """
        key = f"mfcc_attack_smoothness_frames_{attack_frames}_norm_{normalize}"
        if key in self._cache_mfcc:
            return self._cache_mfcc[key]

        if self.mfcc is None or self.mfcc.size == 0:
            self._cache_mfcc[key] = 0.0
            return 0.0

        T = self.mfcc.shape[1]
        if attack_frames is None:
            attack_frames = max(2, min(T // 5, 10))
        attack_frames = int(np.clip(attack_frames, 2, T))

        # Use timbral coefficients 1–4 only (exclude coeff 0 which tracks energy)
        attack_region = self.mfcc[1:5, :attack_frames]
        if attack_region.shape[1] < 2:
            self._cache_mfcc[key] = 1.0
            return 1.0

        # Smoothness = inverse of mean absolute frame-to-frame change during attack
        delta_attack = np.diff(attack_region, axis=1)
        mean_change  = float(np.mean(np.abs(delta_attack)))

        if normalize:
            # Normalise by mean absolute change over the full signal for comparability
            delta_full  = np.diff(self.mfcc[1:5, :], axis=1)
            global_scale = float(np.mean(np.abs(delta_full))) + EPS
            mean_change /= global_scale

        smoothness = float(1.0 / (1.0 + mean_change))
        smoothness = float(np.clip(smoothness, 0.0, 1.0))

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
            rms_sustain = np.sqrt(np.mean(sustain**2, axis=1))   # per-coefficient RMS
            cv = np.mean(sigma / (rms_sustain + EPS))
            stability = 1.0 / (1.0 + cv)
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
        """
        MFCC low-coefficient profile shape statistics.
        Returns centroid, spread, and slope of the mean profile across low coefficients.
        The old z-score → norm approach was nearly mathematically predetermined.
        """
        key = f"mfcc_profile_shape_coeffs_{coeffs}"
        if key in self._cache_mfcc:
            return self._cache_mfcc[key]

        if self.mfcc is None or self.mfcc.size == 0:
            result = {"score": 0.0, "centroid": 0.0, "spread": 0.0,
                    "slope": 0.0, "pattern": np.array([], dtype=float)}
            self._cache_mfcc[key] = result
            return result

        idx = np.array(coeffs, dtype=int)
        idx = idx[idx < self.mfcc.shape[0]]
        if idx.size < 2:
            result = {"score": 0.0, "centroid": 0.0, "spread": 0.0,
                    "slope": 0.0, "pattern": np.array([], dtype=float)}
            self._cache_mfcc[key] = result
            return result

        pattern = np.mean(np.abs(self.mfcc[idx, :]), axis=1).astype(float)
        total   = np.sum(pattern) + EPS

        # Centroid: energy-weighted average coefficient index
        centroid = float(np.dot(idx.astype(float), pattern) / total)

        # Spread: energy-weighted std of coefficient indices
        spread = float(np.sqrt(np.dot(pattern, (idx.astype(float) - centroid)**2) / total))

        # Slope: linear regression across the profile
        x = idx.astype(float) - np.mean(idx)
        slope = float(np.dot(x, pattern) / (np.dot(x, x) + EPS))

        # Score: spread normalised by number of coefficients (0=all in one coeff, 1=flat)
        score = float(np.clip(spread / (idx.size + EPS), 0.0, 1.0))

        result = {
            "score":    score,
            "centroid": centroid,
            "spread":   spread,
            "slope":    slope,
            "pattern":  pattern,
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

    def _mfcc_energy(self):
        """
        RMS of the full MFCC matrix (all coefficients, all frames) as a
        relative energy measure. No MFCC-0 dependence, no arbitrary constant.
        Returns a value in [0,1] via a sigmoid scaled to typical MFCC RMS values.
        """
        key = "mfcc_energy"
        if key in self._cache_mfcc:
            return self._cache_mfcc[key]

        if self.mfcc is None or self.mfcc.size == 0:
            self._cache_mfcc[key] = 0.0
            return 0.0

        # RMS across all coefficients and frames — true signal magnitude proxy
        rms = float(np.sqrt(np.mean(self.mfcc**2)))

        # Soft-normalise: sigmoid centred at rms=30 (typical for 13-coeff dB-scale MFCCs)
        # rms=10 → ~0.27,  rms=30 → 0.50,  rms=60 → ~0.73
        val = float(1.0 / (1.0 + np.exp(-(rms - 30.0) / 15.0)))
        val = float(np.clip(val, 0.0, 1.0))

        self._cache_mfcc[key] = val
        return val

    def _mfcc_relative_rms(self, normalize=True):
        """
        Renamed from _mfcc_rms_energy.
        RMS of MFCC matrix, optionally expressed relative to the 95th-percentile
        frame norm (more robust than max-normalisation).
        """
        key = f"mfcc_relative_rms_normalize_{normalize}"
        if key in self._cache_mfcc:
            return self._cache_mfcc[key]

        if self.mfcc is None or self.mfcc.size == 0:
            self._cache_mfcc[key] = 0.0
            return 0.0

        rms = float(np.sqrt(np.mean(self.mfcc**2)))

        if normalize:
            # 95th-percentile frame norm — robust against occasional loud frames
            frame_norms = np.linalg.norm(self.mfcc, axis=0)
            p95 = float(np.percentile(frame_norms, 95)) + EPS
            result = float(np.clip(rms / p95, 0.0, 1.0))
        else:
            result = rms

        self._cache_mfcc[key] = result
        return result

    def _mfcc_rms_energy(self, normalize=True):
        """Deprecated alias for _mfcc_relative_rms."""
        return self._mfcc_relative_rms(normalize=normalize)

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

        result = {
            "mean":  float(flux),
            "std":   float(np.std(np.linalg.norm(d, axis=0)) / (scale if normalize else 1.0 + EPS)),
            "p95":   float(np.percentile(np.linalg.norm(d, axis=0), 95) / (scale if normalize else 1.0 + EPS)),
        }

        self._cache_mfcc[key] = result

        # For backwards compatibility: callers that expect a float still work
        # because result["mean"] is the primary value. Update all callers to use
        # self._mfcc_flux()["mean"] instead of self._mfcc_flux() directly.
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

        v_high  = float(np.mean(np.var(X, axis=1)))
        v_total = float(np.mean(np.var(self.mfcc, axis=1)))

        if normalize:
            # Ratio of high-order variance to total variance: always in [0,1]
            v = float(np.clip(v_high / (v_total + EPS), 0.0, 1.0))
        else:
            v = v_high

        self._cache_mfcc[key] = v
        return v

    def _mfcc_coefficient_entropy(self, normalize=True):
        """
        Renamed from _mfcc_entropy.
        Shannon entropy of the mean absolute MFCC coefficient distribution.
        High = energy spread across many coefficients; low = concentrated in few.
        """
        key = f"mfcc_coefficient_entropy_normalize_{normalize}"
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
            ent /= np.log(len(p))

        self._cache_mfcc[key] = float(ent)
        return float(ent)

    def _mfcc_entropy(self, normalize=True):
        """Deprecated alias for _mfcc_coefficient_entropy."""
        return self._mfcc_coefficient_entropy(normalize=normalize)

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

        loud = self._mfcc_energy()

        self._cache_mfcc[key] = loud
        return loud

    def _energy_mfcc(self):
        key = "energy_mfcc"
        if key in self._cache_mfcc:
            return self._cache_mfcc[key]
        
        e0 = self._mfcc_spectral_slope_proxy(start_coeff=0, aggregate="rms")/100
        e1 = self._mfcc_rms_energy(normalize=True)
        e2 = self._mfcc_energy()

        val = float(np.clip(0.5*abs(e0) + 0.25*e1 + 0.25*e2, 0.0, 1.0))

        self._cache_mfcc[key] = val
        return val
    
    def _mfcc_temporal_complexity(self):
        """
        Renamed from _speechiness_mfcc.
        Measures MFCC temporal change rate and high-order spectral complexity.
        Not actual speechiness (which requires ZCR, formant tracking, etc.).
        """
        key = "mfcc_temporal_complexity"
        if key in self._cache_mfcc:
            return self._cache_mfcc[key]

        flux = self._mfcc_flux(normalize=True)["mean"]
        high = self._mfcc_high_order_energy_ratio(start_coeff=6, normalize=True, order="l2")
        var  = self._mfcc_high_order_variance(start_coeff=6, normalize=True)
        ent  = self._mfcc_coefficient_entropy(normalize=True)

        val = float(np.clip(
            0.40*flux +
            0.30*np.clip(high, 0.0, 1.0) +
            0.20*np.clip(var,  0.0, 1.0) +
            0.10*ent,
            0.0, 1.0
        ))

        self._cache_mfcc[key] = val
        return val

    def _speechiness_mfcc(self):
        """Deprecated alias for _mfcc_temporal_complexity."""
        return self._mfcc_temporal_complexity()
    
    def _mfcc_timbre_smoothness(self):
        """
        Renamed from _acousticness_mfcc.
        Measures timbral smoothness via MFCC flux and high-order energy.
        Not acousticness (which requires broadband noise floor analysis).
        """
        key = "mfcc_timbre_smoothness"
        if key in self._cache_mfcc:
            return self._cache_mfcc[key]

        smooth = self._mfcc_smoothness(normalize=True)
        ent    = self._mfcc_coefficient_entropy(normalize=True)
        high   = self._mfcc_high_order_energy_ratio(start_coeff=6, normalize=True, order="l2")
        flux   = self._mfcc_flux(normalize=True)["mean"]

        val = float(np.clip(
            0.35*smooth +
            0.25*(1.0 - ent) +
            0.25*(1.0 - np.clip(high, 0.0, 1.0)) +
            0.15*(1.0 - np.clip(flux, 0.0, 1.0)),
            0.0, 1.0
        ))

        self._cache_mfcc[key] = val
        return val

    def _acousticness_mfcc(self):
        """Deprecated alias for _mfcc_timbre_smoothness."""
        return self._mfcc_timbre_smoothness()

    def _valence_mfcc(self):
        """Deprecated compatibility method; MFCCs provide no valence estimate."""
        import logging
        logging.getLogger(__name__).warning(
            "_valence_mfcc() is deprecated and returns None. "
            "Use ChromagramFeatures._valence_chroma() for valence estimation."
        )
        return None

    def _mfcc_temporal_variability(self):
        """
        Renamed from _liveness_mfcc.
        Measures MFCC temporal variability (roughness, flux, high-order energy).
        Not liveness (which detects audience noise / reverb characteristics).
        """
        key = "mfcc_temporal_variability"
        if key in self._cache_mfcc:
            return self._cache_mfcc[key]

        rough = self._mfcc_transient_roughness(width=9, normalize=True, mode="interp")
        flux  = self._mfcc_flux(normalize=True)["mean"]
        high  = self._mfcc_high_order_energy_ratio(start_coeff=6, normalize=True, order="l2")
        ent   = self._mfcc_coefficient_entropy(normalize=True)

        val = float(np.clip(
            0.35*np.clip(rough / (1.0 + rough), 0.0, 1.0) +
            0.30*np.clip(flux, 0.0, 1.0) +
            0.20*np.clip(high, 0.0, 1.0) +
            0.15*ent,
            0.0, 1.0
        ))

        self._cache_mfcc[key] = val
        return val

    def _liveness_mfcc(self):
        """Deprecated alias for _mfcc_temporal_variability."""
        return self._mfcc_temporal_variability()

    def _instrumentalness_mfcc(self):
        """Deprecated compatibility method; MFCCs provide no reliable instrumentalness estimate."""
        import logging
        logging.getLogger(__name__).warning(
            "_instrumentalness_mfcc() is deprecated and returns None. "
            "Use a dedicated vocal activity detector for instrumentalness."
        )
        return None

    def mfcc_domain_evidence(self):
        """Return MFCC-derived evidence without claiming Spotify feature equivalence."""
        return {
            "rms_loudness_evidence": self._loudness_mfcc(),
            "timbral_energy_evidence": self._energy_mfcc(),
            "temporal_complexity": self._mfcc_temporal_complexity(),
            "timbre_smoothness": self._mfcc_timbre_smoothness(),
            "temporal_variability": self._mfcc_temporal_variability(),
        }

    def spotify_audio_features(self):
        """Deprecated compatibility alias for :meth:`mfcc_domain_evidence`."""
        return self.mfcc_domain_evidence()
