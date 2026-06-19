"""
MFCC (Mel-Frequency Cepstral Coefficients) features.
"""

import numpy as np
import librosa
from .utils import EPS, safe_clip01


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