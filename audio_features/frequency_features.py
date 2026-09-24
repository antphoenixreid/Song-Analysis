"""
Frequency-domain audio features used as evidence for Spotify-like audio features.

This class intentionally does NOT claim to reproduce Spotify's proprietary
models.  It extracts interpretable frequency-domain measurements that can be
combined later with Time, Chroma, Tempogram, and MFCC evidence.
"""

from __future__ import annotations

import numpy as np
import librosa
from scipy.signal import find_peaks

from .utils import EPS, safe_clip01
from audio_features.audio_signal import AudioSignal


TEMPERLEY_MAJOR = np.array(
    [5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0],
    dtype=float,
)

TEMPERLEY_MINOR = np.array(
    [5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0],
    dtype=float,
)


def _build_centered_freq_key_templates() -> np.ndarray:
    """Build mean-centered, L2-normalized major/minor key templates."""
    templates = np.zeros((24, 12), dtype=float)
    for tonic in range(12):
        templates[tonic] = np.roll(TEMPERLEY_MAJOR, tonic)
        templates[tonic + 12] = np.roll(TEMPERLEY_MINOR, tonic)

    templates -= np.mean(templates, axis=1, keepdims=True)
    norms = np.linalg.norm(templates, axis=1, keepdims=True)
    return templates / (norms + EPS)


KEY_TEMPLATES_FREQ_NORM = _build_centered_freq_key_templates()


class FrequencyFeatures:
    """Frequency-domain descriptors and Spotify-like frequency evidence."""

    def __init__(self, sig: AudioSignal):
        self.sig = sig
        self.y = np.asarray(sig.y, dtype=float)
        self.sr = int(sig.sr)
        self.N = int(sig.N)
        self.H = int(sig.H)

        # Keep the source signal unchanged.  The STFT was already produced by
        # AudioSignal, so padding self.y here would make time/F0/STFT lengths
        # inconsistent for short recordings.
        self.X = np.asarray(sig.stft)
        self.freqs = np.asarray(sig.fft_freqs, dtype=float)

        if self.X.ndim != 2:
            raise ValueError("sig.stft must be a 2-D array with shape (frequency, time)")
        if self.freqs.ndim != 1 or self.freqs.shape[0] != self.X.shape[0]:
            raise ValueError("sig.fft_freqs must match the frequency dimension of sig.stft")

        self._cache_freq: dict[str, object] = {}

        self._freq_resolution = float(self.sr) / float(self.N)
        self._one_sided_factor = self._build_one_sided_factor()

    # ------------------------------------------------------------------
    # Core spectral helpers
    # ------------------------------------------------------------------
    def _build_one_sided_factor(self) -> np.ndarray:
        """Return the Parseval correction for a real, one-sided FFT/STFT."""
        factor = np.ones(self.X.shape[0], dtype=float)
        if factor.size > 2:
            factor[1:-1] = 2.0
        elif factor.size == 2:
            factor[1] = 2.0
        return factor

    def _safe_band_mask(self, f_lo: float, f_hi: float) -> np.ndarray:
        lo = float(min(f_lo, f_hi))
        hi = float(max(f_lo, f_hi))
        return (self.freqs >= lo) & (self.freqs < hi)

    def _valid_frequency_mask(self, f_min: float = 20.0, f_max: float | None = None) -> np.ndarray:
        if f_max is None:
            f_max = float(self.freqs[-1]) if self.freqs.size else 0.0
        return (self.freqs >= float(f_min)) & (self.freqs <= float(f_max))

    def _magnitude_spectrum(self) -> np.ndarray:
        if "mag" not in self._cache_freq:
            self._cache_freq["mag"] = np.abs(self.X).astype(float)
        return self._cache_freq["mag"]  # type: ignore[return-value]

    def _power_spectrum(self) -> np.ndarray:
        """
        Parseval-consistent one-sided frame power.

        For a standard librosa Hann-windowed STFT, dividing by
        N * sum(w^2) converts the squared DFT magnitude into mean-square
        frame energy.  The one-sided correction doubles interior bins.
        """
        if "pow" not in self._cache_freq:
            mag = self._magnitude_spectrum()
            power = (mag ** 2) * self._one_sided_factor[:, None]
            # librosa's default Hann window has sum(w^2) approximately 0.375*N.
            # Keep this normalization explicit and isolated here.
            window_power = 0.375 * float(self.N)
            power /= max(window_power * float(self.N), EPS)
            self._cache_freq["pow"] = power
        return self._cache_freq["pow"]  # type: ignore[return-value]

    def _db_spectrum(self, ref: float | None = None, power: bool = False) -> np.ndarray:
        """Return dB spectrum with a cache key that includes the reference."""
        S = self._power_spectrum() if power else self._magnitude_spectrum()
        ref_key = "max" if ref is None else f"{float(ref):.12g}"
        key = f"db_{'pow' if power else 'mag'}_{ref_key}"
        if key in self._cache_freq:
            return self._cache_freq[key]  # type: ignore[return-value]

        if ref is None:
            ref = float(np.max(S)) if S.size else 1.0
        ref = max(float(ref), EPS)

        if power:
            db = 10.0 * np.log10(np.maximum(S, EPS) / ref)
        else:
            db = 20.0 * np.log10(np.maximum(S, EPS) / ref)

        self._cache_freq[key] = db
        return db

    def _frame_energy(self) -> np.ndarray:
        """Mean-square frame energy derived from the corrected power spectrum."""
        if "frame_energy" not in self._cache_freq:
            power = self._power_spectrum()
            self._cache_freq["frame_energy"] = np.sum(power, axis=0)
        return self._cache_freq["frame_energy"]  # type: ignore[return-value]

    def _frame_energy_db(self) -> np.ndarray:
        if "frame_energy_db" not in self._cache_freq:
            e = self._frame_energy()
            self._cache_freq["frame_energy_db"] = 10.0 * np.log10(np.maximum(e, EPS))
        return self._cache_freq["frame_energy_db"]  # type: ignore[return-value]

    def _dynamic_range(self) -> float:
        if "dynamic_range" not in self._cache_freq:
            e_db = self._frame_energy_db()
            if e_db.size == 0:
                value = 0.0
            else:
                valid = np.isfinite(e_db)
                value = (
                    float(np.percentile(e_db[valid], 90) - np.percentile(e_db[valid], 10))
                    if np.any(valid)
                    else 0.0
                )
            self._cache_freq["dynamic_range"] = value
        return float(self._cache_freq["dynamic_range"])

    def _track_activity_confidence(self) -> float:
        """Bounded confidence that the source contains meaningful audio energy."""
        key = "track_activity_confidence"
        if key in self._cache_freq:
            return float(self._cache_freq[key])

        energy = self._frame_energy()
        if energy.size == 0:
            confidence = 0.0
        else:
            positive = energy[energy > EPS]
            if positive.size == 0:
                confidence = 0.0
            else:
                peak = float(np.max(positive))
                median = float(np.median(positive))
                confidence = safe_clip01(median / (peak + EPS))
                if peak <= 1e-12:
                    confidence = 0.0

        self._cache_freq[key] = confidence
        return confidence

    def _band_energy(self, bands, use_power: bool = True) -> np.ndarray:
        """Absolute spectral energy in each requested frequency band."""
        key = f"band_energy_{tuple(bands)}_{use_power}"
        if key in self._cache_freq:
            return self._cache_freq[key]  # type: ignore[return-value]

        S = self._power_spectrum() if use_power else self._magnitude_spectrum()
        out = np.zeros((len(bands), S.shape[1]), dtype=float)
        for i, (f_lo, f_hi) in enumerate(bands):
            mask = self._safe_band_mask(f_lo, f_hi)
            if np.any(mask):
                out[i] = np.sum(S[mask, :], axis=0)

        self._cache_freq[key] = out
        return out

    def _band_energy_ratio(self, bands) -> np.ndarray:
        """Relative power in each requested band."""
        key = f"band_energy_ratio_{tuple(bands)}"
        if key not in self._cache_freq:
            band_energy = self._band_energy(bands, use_power=True)
            total = self._frame_energy()[None, :] + EPS
            self._cache_freq[key] = band_energy / total
        return self._cache_freq[key]  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Spectral shape
    # ------------------------------------------------------------------
    def _spectral_centroid(self, use_power: bool = True) -> np.ndarray:
        key = f"spectral_centroid_{'pow' if use_power else 'mag'}"
        if key in self._cache_freq:
            return self._cache_freq[key]  # type: ignore[return-value]

        S = self._power_spectrum() if use_power else self._magnitude_spectrum()
        mask = self._valid_frequency_mask()
        S_use = S[mask]
        f = self.freqs[mask, None]

        if S_use.size == 0:
            centroid = np.zeros(S.shape[1], dtype=float)
        else:
            denom = np.sum(S_use, axis=0) + EPS
            centroid = np.sum(f * S_use, axis=0) / denom

        self._cache_freq[key] = centroid
        return centroid

    def _spectral_bandwidth(self, use_power: bool = True) -> np.ndarray:
        key = f"spectral_bandwidth_{'pow' if use_power else 'mag'}"
        if key in self._cache_freq:
            return self._cache_freq[key]  # type: ignore[return-value]

        S = self._power_spectrum() if use_power else self._magnitude_spectrum()
        mask = self._valid_frequency_mask()
        S_use = S[mask]
        f = self.freqs[mask, None]
        centroid = self._spectral_centroid(use_power=use_power)[None, :]

        if S_use.size == 0:
            bw = np.zeros(S.shape[1], dtype=float)
        else:
            denom = np.sum(S_use, axis=0) + EPS
            var = np.sum(((f - centroid) ** 2) * S_use, axis=0) / denom
            bw = np.sqrt(np.maximum(var, 0.0))

        self._cache_freq[key] = bw
        return bw

    def _spectral_rolloff(self, roll_percent: float = 0.85, use_power: bool = True) -> np.ndarray:
        if not 0.0 < roll_percent < 1.0:
            raise ValueError("roll_percent must be between 0 and 1")

        key = f"spectral_roll_{roll_percent}_{'pow' if use_power else 'mag'}"
        if key in self._cache_freq:
            return self._cache_freq[key]  # type: ignore[return-value]

        S = self._power_spectrum() if use_power else self._magnitude_spectrum()
        mask = self._valid_frequency_mask()
        S_use = S[mask]
        freqs = self.freqs[mask]
        T = S.shape[1]
        rolloff = np.zeros(T, dtype=float)

        for t in range(T):
            spec = S_use[:, t]
            total = float(np.sum(spec))
            if total <= EPS:
                continue
            threshold = float(roll_percent) * total
            idx = int(np.searchsorted(np.cumsum(spec), threshold, side="left"))
            idx = min(idx, freqs.size - 1)
            rolloff[t] = freqs[idx]

        self._cache_freq[key] = rolloff
        return rolloff

    def _spectral_slope(self, use_power: bool = True, log_amp: bool = False) -> np.ndarray:
        key = f"spectral_slope_{'pow' if use_power else 'mag'}_{'log' if log_amp else 'lin'}"
        if key in self._cache_freq:
            return self._cache_freq[key]  # type: ignore[return-value]

        S = self._power_spectrum() if use_power else self._magnitude_spectrum()
        mask = self._valid_frequency_mask()
        x = self.freqs[mask].astype(float)
        if x.size < 2:
            slopes = np.zeros(S.shape[1], dtype=float)
            self._cache_freq[key] = slopes
            return slopes

        x_centered = x - np.mean(x)
        denom = np.sum(x_centered ** 2) + EPS
        slopes = np.zeros(S.shape[1], dtype=float)

        for t in range(S.shape[1]):
            y = S[mask, t].astype(float)
            if log_amp:
                y = np.log(np.maximum(y, EPS))
            y_centered = y - np.mean(y)
            slopes[t] = np.sum(x_centered * y_centered) / denom

        self._cache_freq[key] = slopes
        return slopes

    def _spectral_skewness(
        self,
        use_power: bool = True,
        energy_thresh: float = 1e-7,
        max_skew: float = 50.0,
    ) -> np.ndarray:
        key = f"spectral_skewness_{'pow' if use_power else 'mag'}_{energy_thresh}_{max_skew}"
        if key in self._cache_freq:
            return self._cache_freq[key]  # type: ignore[return-value]

        S = self._power_spectrum() if use_power else self._magnitude_spectrum()
        mask_f = self._valid_frequency_mask()
        S_use = S[mask_f]
        freqs = self.freqs[mask_f, None]
        sums = np.sum(S_use, axis=0)
        valid = sums > max(float(energy_thresh), EPS)

        skew = np.zeros(S.shape[1], dtype=float)
        if np.any(valid):
            p = S_use[:, valid] / (sums[valid][None, :] + EPS)
            centroid = np.sum(freqs * p, axis=0)
            diffs = freqs - centroid[None, :]
            mu2 = np.maximum(np.sum((diffs ** 2) * p, axis=0), 1.0)
            mu3 = np.sum((diffs ** 3) * p, axis=0)
            skew[valid] = mu3 / (mu2 ** 1.5)

        skew = np.clip(np.nan_to_num(skew, nan=0.0, posinf=max_skew, neginf=-max_skew), -max_skew, max_skew)
        self._cache_freq[key] = skew
        return skew

    def _spectral_kurtosis(
        self,
        use_power: bool = True,
        excess: bool = True,
        energy_thresh: float = 1e-7,
        max_kurt: float = 100.0,
    ) -> np.ndarray:
        key = f"spectral_kurtosis_{'pow' if use_power else 'mag'}_{excess}_{energy_thresh}_{max_kurt}"
        if key in self._cache_freq:
            return self._cache_freq[key]  # type: ignore[return-value]

        S = self._power_spectrum() if use_power else self._magnitude_spectrum()
        mask_f = self._valid_frequency_mask()
        S_use = S[mask_f]
        freqs = self.freqs[mask_f, None]
        sums = np.sum(S_use, axis=0)
        valid = sums > max(float(energy_thresh), EPS)

        kurt = np.zeros(S.shape[1], dtype=float)
        if np.any(valid):
            p = S_use[:, valid] / (sums[valid][None, :] + EPS)
            centroid = np.sum(freqs * p, axis=0)
            diffs = freqs - centroid[None, :]
            mu2 = np.maximum(np.sum((diffs ** 2) * p, axis=0), 1.0)
            mu4 = np.sum((diffs ** 4) * p, axis=0)
            kurt[valid] = mu4 / (mu2 ** 2)
            if excess:
                kurt[valid] -= 3.0

        kurt = np.clip(np.nan_to_num(kurt, nan=0.0, posinf=max_kurt, neginf=-max_kurt), -max_kurt, max_kurt)
        self._cache_freq[key] = kurt
        return kurt

    # ------------------------------------------------------------------
    # Harmonic / pitch structure
    # ------------------------------------------------------------------
    def _fundamental_freq_estimate(self, fmin: float = 50.0, fmax: float = 2000.0) -> np.ndarray:
        key = f"f0_estimate_{fmin}_{fmax}"
        if key in self._cache_freq:
            return self._cache_freq[key]  # type: ignore[return-value]

        if self.y.size < max(3, self.N):
            f0_raw = np.array([], dtype=float)
        else:
            f0_raw = librosa.yin(
                self.y,
                fmin=float(fmin),
                fmax=float(fmax),
                sr=self.sr,
                hop_length=self.H,
                frame_length=self.N,
                center=True,
            )

        f0 = np.asarray(f0_raw, dtype=float)
        f0[~np.isfinite(f0)] = 0.0
        f0[(f0 <= float(fmin) * 1.05) | (f0 >= float(fmax))] = 0.0

        T = self._magnitude_spectrum().shape[1]
        if f0.size > T:
            f0 = f0[:T]
        elif f0.size < T:
            f0 = np.pad(f0, (0, T - f0.size), constant_values=0.0)

        self._cache_freq[key] = f0
        return f0

    def _harmonic_bin_indices(self, f0: np.ndarray, fmax: float | None = None):
        """Return frequency-bin neighborhoods for integer harmonics."""
        freqs = self.freqs
        K, T = self._magnitude_spectrum().shape
        if fmax is None:
            fmax = float(freqs[-1])

        result = []
        for t in range(T):
            f0_t = float(f0[t]) if t < f0.size else 0.0
            if f0_t <= 0.0:
                result.append(np.array([], dtype=int))
                continue

            h = np.arange(1, int(np.floor(float(fmax) / f0_t)) + 1, dtype=int)
            harmonic_freqs = h.astype(float) * f0_t
            bins = np.rint(harmonic_freqs / self._freq_resolution).astype(int)
            bins = bins[(bins >= 0) & (bins < K)]
            result.append(np.unique(bins))

        return result

    def _harmonic_ratio(self) -> np.ndarray:
        """Harmonic-band power divided by total power per frame."""
        if "harmonic_ratio" in self._cache_freq:
            return self._cache_freq["harmonic_ratio"]  # type: ignore[return-value]

        S = self._power_spectrum()
        f0 = self._fundamental_freq_estimate()
        K, T = S.shape
        hr = np.zeros(T, dtype=float)
        tol_hz = max(self._freq_resolution, 4.0)

        for t in range(T):
            f0_t = float(f0[t])
            total = float(np.sum(S[:, t]))
            if f0_t <= 0.0 or total <= EPS:
                continue

            harmonic_mask = np.zeros(K, dtype=bool)
            h = 1
            while h * f0_t <= self.freqs[-1] + tol_hz:
                target = h * f0_t
                harmonic_mask |= np.abs(self.freqs - target) <= tol_hz
                h += 1

            harmonic_energy = float(np.sum(S[harmonic_mask, t]))
            hr[t] = np.clip(harmonic_energy / (total + EPS), 0.0, 1.0)

        self._cache_freq["harmonic_ratio"] = hr
        return hr

    def _inharmonicity(self, peak_height_factor: float = 0.2, tolerance_ratio: float = 0.06) -> np.ndarray:
        """Normalized deviation of accepted spectral peaks from harmonic locations."""
        key = f"inharmonicity_{peak_height_factor}_{tolerance_ratio}"
        if key in self._cache_freq:
            return self._cache_freq[key]  # type: ignore[return-value]

        mag = self._magnitude_spectrum()
        f0 = self._fundamental_freq_estimate()
        T = mag.shape[1]
        inh = np.zeros(T, dtype=float)

        for t in range(T):
            f0_t = float(f0[t])
            if f0_t <= 0.0:
                continue

            spec = mag[:, t]
            max_mag = float(np.max(spec)) if spec.size else 0.0
            if max_mag <= EPS:
                continue

            peak_idx, props = find_peaks(spec, height=peak_height_factor * max_mag)
            if peak_idx.size == 0:
                continue

            weighted_error = 0.0
            weight_sum = 0.0
            for idx, amp in zip(peak_idx, props.get("peak_heights", np.zeros(peak_idx.size))):
                f_peak = float(self.freqs[idx])
                if f_peak <= 0.0:
                    continue

                harmonic = max(1, int(np.rint(f_peak / f0_t)))
                ideal = harmonic * f0_t
                rel_error = abs(f_peak - ideal) / max(ideal, self._freq_resolution)
                if rel_error > tolerance_ratio:
                    # Peak is too far from a plausible harmonic: it is treated
                    # as residual/non-harmonic energy rather than assigned a huge
                    # harmonic deviation.
                    continue

                weight = max(float(amp), EPS)
                weighted_error += weight * rel_error
                weight_sum += weight

            if weight_sum > EPS:
                inh[t] = weighted_error / weight_sum

        self._cache_freq[key] = np.clip(inh, 0.0, 1.0)
        return self._cache_freq[key]  # type: ignore[return-value]

    def _spectral_peaks(self, height_factor: float = 0.2, max_peaks: int = 20):
        key = f"spectral_peaks_{height_factor}_{max_peaks}"
        if key in self._cache_freq:
            return self._cache_freq[key]

        mag = self._magnitude_spectrum()
        peak_freqs = []
        peak_mags = []

        for t in range(mag.shape[1]):
            spec = mag[:, t]
            if spec.size == 0 or np.max(spec) <= EPS:
                peak_freqs.append(np.array([], dtype=float))
                peak_mags.append(np.array([], dtype=float))
                continue

            idx, props = find_peaks(spec, height=float(height_factor) * float(np.max(spec)))
            if idx.size > max_peaks:
                heights = props.get("peak_heights", spec[idx])
                idx = idx[np.argsort(heights)[::-1][:max_peaks]]

            peak_freqs.append(self.freqs[idx].astype(float))
            peak_mags.append(spec[idx].astype(float))

        result = (peak_freqs, peak_mags)
        self._cache_freq[key] = result
        return result

    def _hnr(
        self,
        f0_hz=None,
        max_harmonics: int = 20,
        tol_bins: float = 1.0,
        use_power: bool = True,
    ) -> np.ndarray:
        """Spectral harmonic-to-residual energy ratio in dB."""
        f0_key = "auto" if f0_hz is None else f"array_{id(f0_hz)}"
        key = f"hnr_{f0_key}_{max_harmonics}_{tol_bins}_{use_power}"
        if key in self._cache_freq:
            return self._cache_freq[key]  # type: ignore[return-value]

        S = self._power_spectrum() if use_power else self._magnitude_spectrum()
        T = S.shape[1]
        if f0_hz is None:
            f0 = self._fundamental_freq_estimate()
        else:
            f0 = np.asarray(f0_hz, dtype=float)
            if f0.size != T:
                raise ValueError("f0_hz must have length equal to the number of STFT frames")

        tol_hz = max(float(tol_bins) * self._freq_resolution, self._freq_resolution)
        hnr_db = np.zeros(T, dtype=float)

        for t in range(T):
            f0_t = float(f0[t])
            if f0_t <= 0.0:
                continue

            mask = np.zeros(S.shape[0], dtype=bool)
            for h in range(1, max_harmonics + 1):
                target = h * f0_t
                if target > self.freqs[-1] + tol_hz:
                    break
                mask |= np.abs(self.freqs - target) <= tol_hz

            eh = float(np.sum(S[mask, t]))
            er = float(np.sum(S[~mask, t]))
            if eh <= EPS:
                continue
            hnr_db[t] = float(np.clip(10.0 * np.log10(eh / (er + EPS)), -60.0, 60.0))

        self._cache_freq[key] = hnr_db
        return hnr_db

    # ------------------------------------------------------------------
    # Spectral envelope / noise
    # ------------------------------------------------------------------
    def _spectral_envelope_bands(self, bands):
        return self._band_energy(bands, use_power=True)

    def _spectral_envelope_normalized(self, bands):
        band_energy = self._spectral_envelope_bands(bands)
        total = np.sum(band_energy, axis=0, keepdims=True) + EPS
        return band_energy / total

    def _spectral_flatness(self, use_power: bool = True) -> np.ndarray:
        key = f"spectral_flatness_{'pow' if use_power else 'mag'}"
        if key in self._cache_freq:
            return self._cache_freq[key]  # type: ignore[return-value]

        S = self._power_spectrum() if use_power else self._magnitude_spectrum()
        mask_f = self._valid_frequency_mask()
        S_use = S[mask_f]
        flatness = np.zeros(S.shape[1], dtype=float)
        if S_use.size:
            energy = np.sum(S_use, axis=0)
            valid = energy > EPS
            if np.any(valid):
                vals = np.maximum(S_use[:, valid], EPS)
                gm = np.exp(np.mean(np.log(vals), axis=0))
                am = np.mean(vals, axis=0) + EPS
                flatness[valid] = gm / am

        self._cache_freq[key] = np.clip(flatness, 0.0, 1.0)
        return self._cache_freq[key]  # type: ignore[return-value]

    def _spectral_entropy(self, use_power: bool = True, normalize: bool = True) -> np.ndarray:
        key = f"spectral_entropy_{'pow' if use_power else 'mag'}_{normalize}"
        if key in self._cache_freq:
            return self._cache_freq[key]  # type: ignore[return-value]

        S = self._power_spectrum() if use_power else self._magnitude_spectrum()
        mask_f = self._valid_frequency_mask()
        S_use = S[mask_f]
        entropy = np.zeros(S.shape[1], dtype=float)
        if S_use.size:
            total = np.sum(S_use, axis=0)
            valid = total > EPS
            if np.any(valid):
                p = S_use[:, valid] / (total[valid][None, :] + EPS)
                H = -np.sum(p * np.log(np.maximum(p, EPS)), axis=0)
                if normalize:
                    H /= np.log(max(S_use.shape[0], 2))
                entropy[valid] = H

        self._cache_freq[key] = np.clip(np.nan_to_num(entropy), 0.0, 1.0 if normalize else np.inf)
        return self._cache_freq[key]  # type: ignore[return-value]

    def _spectral_flux(
        self,
        use_power: bool = False,
        normalize: bool = True,
        half_wave_rectify: bool = False,
    ) -> np.ndarray:
        key = f"spectral_flux_{'pow' if use_power else 'mag'}_{normalize}_{half_wave_rectify}"
        if key in self._cache_freq:
            return self._cache_freq[key]  # type: ignore[return-value]

        S = self._power_spectrum() if use_power else self._magnitude_spectrum()
        S = S[self._valid_frequency_mask()]
        if S.shape[1] < 2:
            flux = np.zeros(S.shape[1], dtype=float)
            self._cache_freq[key] = flux
            return flux

        if normalize:
            norms = np.linalg.norm(S, axis=0, keepdims=True) + EPS
            S = S / norms

        diff = S[:, 1:] - S[:, :-1]
        if half_wave_rectify:
            diff = np.maximum(diff, 0.0)

        flux = np.sqrt(np.sum(diff ** 2, axis=0))
        flux = np.concatenate([[0.0], flux])
        self._cache_freq[key] = np.nan_to_num(flux)
        return self._cache_freq[key]  # type: ignore[return-value]

    def _band_ratios(self, bands, use_power: bool = True, relative: bool = True) -> np.ndarray:
        key = f"band_ratios_{tuple(bands)}_{use_power}_{relative}"
        if key in self._cache_freq:
            return self._cache_freq[key]  # type: ignore[return-value]

        B = self._band_energy(bands, use_power=use_power)
        if B.size == 0:
            result = np.zeros((len(bands), 0), dtype=float)
        elif relative:
            total = np.sum(self._power_spectrum() if use_power else self._magnitude_spectrum(), axis=0) + EPS
            result = B / total[None, :]
        else:
            result = B

        self._cache_freq[key] = result
        return result

    def _low_high_band_ratio(self, low_band, high_band) -> np.ndarray:
        key = f"low_high_band_ratio_{low_band}_{high_band}"
        if key in self._cache_freq:
            return self._cache_freq[key]  # type: ignore[return-value]

        e = self._band_energy([low_band, high_band], use_power=True)
        ratio = e[0] / (e[1] + EPS)
        self._cache_freq[key] = ratio
        return ratio

    # ------------------------------------------------------------------
    # Rhythm / transient evidence
    # ------------------------------------------------------------------
    @staticmethod
    def _parabolic_interpolation(f: np.ndarray, x: int) -> float:
        if x <= 0 or x >= len(f) - 1:
            return float(x)
        a, b, c = float(f[x - 1]), float(f[x]), float(f[x + 1])
        denom = a - 2.0 * b + c
        if abs(denom) < 1e-9:
            return float(x)
        return float(x - 0.5 * (a - c) / denom)

    def _rhythmic_autocorrelation(self) -> np.ndarray:
        key = "rhythmic_ac"
        if key in self._cache_freq:
            return self._cache_freq[key]  # type: ignore[return-value]

        flux = self._spectral_flux(use_power=False, normalize=True, half_wave_rectify=True)
        if flux.size < 3:
            ac = np.zeros(1, dtype=float)
        else:
            x = flux - np.mean(flux)
            ac = np.correlate(x, x, mode="full")[x.size - 1 :]
            if ac.size and ac[0] > EPS:
                ac = ac / ac[0]

        self._cache_freq[key] = np.nan_to_num(ac)
        return self._cache_freq[key]

    def _tempo_peak_candidates(self, bpm_min: float = 40.0, bpm_max: float = 240.0):
        """Return autocorrelation tempo candidates sorted by peak strength."""
        key = f"tempo_candidates_{bpm_min}_{bpm_max}"
        if key in self._cache_freq:
            return self._cache_freq[key]

        ac = self._rhythmic_autocorrelation()
        if ac.size < 3:
            self._cache_freq[key] = []
            return []

        fs_env = self.sr / float(self.H)
        lag_min = max(1, int(np.floor(fs_env * 60.0 / float(bpm_max))))
        lag_max = min(ac.size - 1, int(np.ceil(fs_env * 60.0 / float(bpm_min))))
        if lag_max <= lag_min:
            self._cache_freq[key] = []
            return []

        region = ac[lag_min : lag_max + 1]
        if region.size < 3:
            self._cache_freq[key] = []
            return []

        prominence = max(0.005, 0.1 * float(np.std(region)))
        peaks, props = find_peaks(region, prominence=prominence)
        if peaks.size == 0:
            peaks = np.array([int(np.argmax(region))], dtype=int)
            heights = region[peaks]
        else:
            heights = props.get("peak_heights", region[peaks])

        candidates = []
        for rel_idx, height in zip(peaks, heights):
            lag = lag_min + int(rel_idx)
            interp_lag = lag_min + self._parabolic_interpolation(region, int(rel_idx))
            interp_lag = max(float(interp_lag), 1.0)
            bpm = 60.0 * fs_env / interp_lag
            if bpm_min <= bpm <= bpm_max:
                candidates.append({"bpm": float(bpm), "lag": interp_lag, "strength": float(height)})

        candidates.sort(key=lambda item: item["strength"], reverse=True)
        self._cache_freq[key] = candidates
        return candidates

    def _pulse_clarity_ac(
        self,
        use_power: bool = False,
        normalize: bool = True,
        half_wave_rectify: bool = True,
        min_lag: int = 1,
        max_lag: int | None = None,
    ) -> float:
        key = f"pulse_clarity_{use_power}_{normalize}_{half_wave_rectify}_{min_lag}_{max_lag}"
        if key in self._cache_freq:
            return float(self._cache_freq[key])

        ac = self._rhythmic_autocorrelation()
        if ac.size < 3:
            self._cache_freq[key] = 0.0
            return 0.0

        lo = max(1, int(min_lag))
        hi = min(ac.size - 1, int(max_lag) if max_lag is not None else ac.size - 1)
        if hi <= lo:
            self._cache_freq[key] = 0.0
            return 0.0

        fs_env = self.sr / float(self.H)
        beat_lo = max(lo, int(np.floor(fs_env * 60.0 / 240.0)))
        beat_hi = min(hi, int(np.ceil(fs_env * 60.0 / 40.0)))
        if beat_hi <= beat_lo:
            return 0.0

        region = ac[beat_lo : beat_hi + 1]
        peaks, _ = find_peaks(region, prominence=max(0.005, 0.05 * float(np.std(region))))
        if peaks.size == 0:
            peak_val = float(np.max(region))
        else:
            peak_val = float(np.max(region[peaks]))

        self._cache_freq[key] = float(np.clip(peak_val, 0.0, 1.0))
        return float(self._cache_freq[key])

    def _beat_periodicity(
        self,
        use_power: bool = False,
        normalize: bool = True,
        half_wave_rectify: bool = True,
        min_lag: int = 1,
        max_lag: int | None = None,
    ) -> float:
        """Beat-range autocorrelation peak relative to the nonzero-lag baseline."""
        key = f"beat_periodicity_{use_power}_{normalize}_{half_wave_rectify}_{min_lag}_{max_lag}"
        if key in self._cache_freq:
            return float(self._cache_freq[key])

        ac = self._rhythmic_autocorrelation()
        if ac.size < 3:
            self._cache_freq[key] = 0.0
            return 0.0

        fs_env = self.sr / float(self.H)
        lo = max(int(min_lag), int(np.floor(fs_env * 60.0 / 240.0)))
        hi = min(
            int(max_lag) if max_lag is not None else ac.size - 1,
            ac.size - 1,
            int(np.ceil(fs_env * 60.0 / 40.0)),
        )
        if hi <= lo:
            self._cache_freq[key] = 0.0
            return 0.0

        region = ac[lo : hi + 1]
        if region.size == 0:
            self._cache_freq[key] = 0.0
            return 0.0

        peak = max(0.0, float(np.max(region)))
        baseline = float(np.median(np.abs(region))) + EPS
        ratio = peak / baseline
        periodicity = ratio / (1.0 + ratio)
        self._cache_freq[key] = float(np.clip(periodicity, 0.0, 1.0))
        return float(self._cache_freq[key])

    def _transient_counts(
        self,
        use_power: bool = False,
        normalize: bool = True,
        half_wave_rectify: bool = True,
        threshold_factor: float = 1.5,
    ) -> int:
        key = f"transient_counts_{use_power}_{normalize}_{half_wave_rectify}_{threshold_factor}"
        if key in self._cache_freq:
            return int(self._cache_freq[key])

        flux = self._spectral_flux(
            use_power=use_power,
            normalize=normalize,
            half_wave_rectify=half_wave_rectify,
        )
        if flux.size < 3:
            self._cache_freq[key] = 0
            return 0

        med = float(np.median(flux))
        mad = float(np.median(np.abs(flux - med))) + EPS
        threshold = med + float(threshold_factor) * mad
        fs_env = self.sr / float(self.H)
        min_distance = max(1, int(round(fs_env * 0.030)))
        peaks, _ = find_peaks(flux, height=threshold, distance=min_distance)
        count = int(peaks.size)
        self._cache_freq[key] = count
        return count

    def _transient_rate(
        self,
        use_power: bool = False,
        normalize: bool = True,
        half_wave_rectify: bool = True,
        threshold_factor: float = 1.5,
    ) -> float:
        key = f"transient_rate_{use_power}_{normalize}_{half_wave_rectify}_{threshold_factor}"
        if key in self._cache_freq:
            return float(self._cache_freq[key])

        count = self._transient_counts(
            use_power=use_power,
            normalize=normalize,
            half_wave_rectify=half_wave_rectify,
            threshold_factor=threshold_factor,
        )
        duration = max(len(self.y) / float(self.sr), EPS)
        rate = float(count) / duration
        self._cache_freq[key] = rate
        return rate

    def _percussive_spectral_slope(self, use_power: bool = True, log_amp: bool = True) -> float:
        key = f"percussive_spectral_slope_{use_power}_{log_amp}"
        if key in self._cache_freq:
            return float(self._cache_freq[key])

        S = self._power_spectrum() if use_power else self._magnitude_spectrum()
        flux = self._spectral_flux(use_power=False, normalize=True, half_wave_rectify=True)
        if S.shape[1] == 0 or flux.size == 0:
            self._cache_freq[key] = 0.0
            return 0.0

        threshold = float(np.median(flux) + np.std(flux))
        frames = np.where(flux > threshold)[0]
        if frames.size == 0:
            self._cache_freq[key] = 0.0
            return 0.0

        mask_f = self._valid_frequency_mask()
        x = self.freqs[mask_f].astype(float)
        xc = x - np.mean(x)
        denom = np.sum(xc ** 2) + EPS
        slopes = []

        for t in frames:
            y = S[mask_f, t].astype(float)
            if log_amp:
                y = np.log(np.maximum(y, EPS))
            yc = y - np.mean(y)
            slopes.append(float(np.sum(xc * yc) / denom))

        value = float(np.median(slopes)) if slopes else 0.0
        self._cache_freq[key] = value
        return value

    # ------------------------------------------------------------------
    # Phase features
    # ------------------------------------------------------------------
    def _phase(self) -> np.ndarray:
        if "phase" not in self._cache_freq:
            self._cache_freq["phase"] = np.unwrap(np.angle(self.X), axis=1)
        return self._cache_freq["phase"]  # type: ignore[return-value]

    def _group_delay(self) -> np.ndarray:
        """Group delay in seconds, masked in very low-energy bins."""
        if "group_delay" in self._cache_freq:
            return self._cache_freq["group_delay"]  # type: ignore[return-value]

        phi = np.unwrap(np.angle(self.X), axis=0)
        omega = 2.0 * np.pi * self.freqs / float(self.sr)
        gd = np.zeros_like(phi, dtype=float)

        if self.freqs.size >= 3:
            gd = -np.gradient(phi, omega, axis=0) / float(self.sr)

            mag = self._magnitude_spectrum()
            frame_max = np.max(mag, axis=0, keepdims=True) + EPS
            valid = mag >= (1e-3 * frame_max)
            gd[~valid] = 0.0

        gd = np.nan_to_num(gd, nan=0.0, posinf=0.0, neginf=0.0)
        self._cache_freq["group_delay"] = gd
        return gd

    def _instantaneous_freq(self) -> np.ndarray:
        if "instantaneous_frequency" in self._cache_freq:
            return self._cache_freq["instantaneous_frequency"]  # type: ignore[return-value]

        X = self.X
        K, T = X.shape
        if T == 0:
            out = np.zeros((K, 0), dtype=float)
            self._cache_freq["instantaneous_frequency"] = out
            return out
        if T == 1:
            out = (np.arange(K, dtype=float)[:, None] * self.sr / self.N)
            self._cache_freq["instantaneous_frequency"] = out
            return out

        phi_1 = np.angle(X[:, :-1]) / (2.0 * np.pi)
        phi_2 = np.angle(X[:, 1:]) / (2.0 * np.pi)
        ind_k = np.arange(K, dtype=float)[:, None]

        expected = ind_k * self.H / float(self.N)
        delta = phi_2 - phi_1 - expected
        delta = np.mod(delta + 0.5, 1.0) - 0.5
        kappa = (float(self.N) / float(self.H)) * delta
        inst = (ind_k + kappa) * float(self.sr) / float(self.N)
        inst = np.clip(np.nan_to_num(inst), 0.0, float(self.sr) / 2.0)

        out = np.concatenate([inst[:, :1], inst], axis=1)
        self._cache_freq["instantaneous_frequency"] = out
        return out

    def _phase_congruency(self, use_power: bool = True) -> np.ndarray:
        """Spectral phase concentration (not the classical multi-scale PC algorithm)."""
        key = f"spectral_phase_concentration_{use_power}"
        if key in self._cache_freq:
            return self._cache_freq[key]  # type: ignore[return-value]

        S = self._power_spectrum() if use_power else self._magnitude_spectrum()
        phase = self._phase()
        weights = np.maximum(S, EPS)
        vec = np.sum(weights * np.exp(1j * phase), axis=0)
        denom = np.sum(weights, axis=0) + EPS
        concentration = np.abs(vec) / denom
        self._cache_freq[key] = np.clip(concentration, 0.0, 1.0)
        return self._cache_freq[key]  # type: ignore[return-value]

    def _phase_coherence_time(self) -> np.ndarray:
        if "phase_coherence_time" in self._cache_freq:
            return self._cache_freq["phase_coherence_time"]  # type: ignore[return-value]

        phase = self._phase()
        if phase.shape[1] < 2:
            out = np.zeros(phase.shape[0], dtype=float)
        else:
            dphi = np.diff(phase, axis=1)
            out = np.abs(np.mean(np.exp(1j * dphi), axis=1))
            out = np.clip(out, 0.0, 1.0)

        self._cache_freq["phase_coherence_time"] = out
        return out

    def _phase_coherence_channels(self, phi_a, phi_b):
        phi_a = np.asarray(phi_a, dtype=float)
        phi_b = np.asarray(phi_b, dtype=float)
        if phi_a.shape != phi_b.shape or phi_a.size == 0:
            return 0.0
        diff = phi_a - phi_b
        return float(np.clip(np.abs(np.mean(np.exp(1j * diff))), 0.0, 1.0))

    # ------------------------------------------------------------------
    # Sub-band features
    # ------------------------------------------------------------------
    def _default_sub_bands(self, n_bands: int = 8, fmin: float = 20.0, fmax: float | None = None):
        """Log-spaced sub-bands, which better reflect audio perception than equal-Hz bands."""
        if n_bands < 1:
            raise ValueError("n_bands must be >= 1")
        if fmax is None:
            fmax = float(self.sr) / 2.0
        fmin = max(float(fmin), 1.0)
        fmax = max(float(fmax), fmin + 1e-6)

        edges = np.geomspace(fmin, fmax, n_bands + 1)
        return [
            (float(edges[i]), float(edges[i + 1]), float(np.sqrt(edges[i] * edges[i + 1])))
            for i in range(n_bands)
        ]

    def _sub_band_energy(self, n_bands: int = 8, use_power: bool = True, fmin: float = 20.0, fmax: float | None = None):
        key = f"sub_band_energy_{n_bands}_{use_power}_{fmin}_{fmax}"
        if key in self._cache_freq:
            return self._cache_freq[key]

        bands = self._default_sub_bands(n_bands=n_bands, fmin=fmin, fmax=fmax)
        result = self._band_energy([(lo, hi) for lo, hi, _ in bands], use_power=use_power)
        self._cache_freq[key] = result
        return result

    def _sub_band_energy_ratios(self, n_bands: int = 8, use_power: bool = True, fmin: float = 20.0, fmax: float | None = None):
        key = f"sub_band_energy_ratios_{n_bands}_{use_power}_{fmin}_{fmax}"
        if key in self._cache_freq:
            return self._cache_freq[key]

        band_energy = self._sub_band_energy(n_bands, use_power, fmin, fmax)
        total = np.sum(band_energy, axis=0, keepdims=True) + EPS
        result = band_energy / total
        self._cache_freq[key] = result
        return result

    def _sub_band_entropy(self, n_bands: int = 8, use_power: bool = True, fmin: float = 20.0, fmax: float | None = None, normalize: bool = True):
        key = f"sub_band_entropy_{n_bands}_{use_power}_{fmin}_{fmax}_{normalize}"
        if key in self._cache_freq:
            return self._cache_freq[key]

        ratios = self._sub_band_energy_ratios(n_bands, use_power, fmin, fmax)
        p = np.maximum(ratios, EPS)
        H = -np.sum(p * np.log(p), axis=0)
        if normalize:
            H /= np.log(max(n_bands, 2))
            H = np.clip(H, 0.0, 1.0)

        self._cache_freq[key] = H
        return H

    def _sub_band_centroid(self, n_bands: int = 8, use_power: bool = True, fmin: float = 20.0, fmax: float | None = None):
        key = f"sub_band_centroid_{n_bands}_{use_power}_{fmin}_{fmax}"
        if key in self._cache_freq:
            return self._cache_freq[key]

        bands = self._default_sub_bands(n_bands=n_bands, fmin=fmin, fmax=fmax)
        centers = np.array([c for _, _, c in bands], dtype=float)
        ratios = self._sub_band_energy_ratios(n_bands, use_power, fmin, fmax)
        centroid = np.sum(centers[:, None] * ratios, axis=0)
        self._cache_freq[key] = centroid
        return centroid

    def _sub_band_flatness(self, n_bands: int = 8, use_power: bool = True, fmin: float = 20.0, fmax: float | None = None):
        key = f"sub_band_flatness_{n_bands}_{use_power}_{fmin}_{fmax}"
        if key in self._cache_freq:
            return self._cache_freq[key]

        band_energy = np.maximum(self._sub_band_energy(n_bands, use_power, fmin, fmax), EPS)
        gm = np.exp(np.mean(np.log(band_energy), axis=0))
        am = np.mean(band_energy, axis=0) + EPS
        flat = np.clip(gm / am, 0.0, 1.0)
        self._cache_freq[key] = flat
        return flat

    def _sub_band_ratio(self, band_a: int, band_b: int, n_bands: int = 8, use_power: bool = True, fmin: float = 20.0, fmax: float | None = None):
        key = f"sub_band_ratio_{band_a}_{band_b}_{n_bands}_{use_power}_{fmin}_{fmax}"
        if key in self._cache_freq:
            return self._cache_freq[key]

        E = self._sub_band_energy(n_bands, use_power, fmin, fmax)
        if not (0 <= band_a < n_bands and 0 <= band_b < n_bands):
            result = np.zeros(E.shape[1], dtype=float)
        else:
            result = E[band_a] / (E[band_b] + EPS)
        self._cache_freq[key] = result
        return result

    def _sub_band_low_high_ratio(self, split_band: int = 4, n_bands: int = 8, use_power: bool = True, fmin: float = 20.0, fmax: float | None = None):
        key = f"sub_band_low_high_ratio_{split_band}_{n_bands}_{use_power}_{fmin}_{fmax}"
        if key in self._cache_freq:
            return self._cache_freq[key]

        E = self._sub_band_energy(n_bands, use_power, fmin, fmax)
        split = int(np.clip(split_band, 1, n_bands - 1))
        low = np.sum(E[:split], axis=0)
        high = np.sum(E[split:], axis=0)
        result = low / (high + EPS)
        self._cache_freq[key] = result
        return result

    # ------------------------------------------------------------------
    # Pitch-class / key evidence
    # ------------------------------------------------------------------
    def _pitch_class_profile(self, use_power: bool = True, f_min: float = 55.0, f_max: float = 4186.0) -> np.ndarray:
        """
        Frequency-derived pitch-class profile.

        This is a chroma-like frequency-domain profile, not a replacement for
        the dedicated ChromaFeatures class.  The frequency weighting is a broad
        log-frequency taper, deliberately not called A-weighting.
        """
        key = f"pitch_class_profile_{use_power}_{f_min:.3f}_{f_max:.3f}"
        if key in self._cache_freq:
            return self._cache_freq[key]  # type: ignore[return-value]

        S = self._power_spectrum() if use_power else self._magnitude_spectrum()
        profile = np.zeros(12, dtype=float)
        weight_sum = np.zeros(12, dtype=float)

        for k, f in enumerate(self.freqs):
            if f < f_min or f > f_max or f <= 0.0:
                continue

            # Broad perceptual taper across about five octaves.
            log_oct = np.log2(f / 440.0)
            weight = np.exp(-0.5 * (log_oct / 2.5) ** 2)

            midi = 12.0 * np.log2(f / 440.0) + 69.0
            pc = int(np.rint(midi)) % 12
            frame_energy = float(np.mean(S[k])) if S.shape[1] else 0.0
            profile[pc] += frame_energy * weight
            weight_sum[pc] += weight

        valid = weight_sum > EPS
        profile[valid] /= weight_sum[valid]
        total = float(np.sum(profile))
        if total > EPS:
            profile /= total

        self._cache_freq[key] = profile
        return profile

    def _mean_pitch_class_profile(self, n_bands: int = 12, use_power: bool = True):
        key = f"mean_pitch_class_profile_{n_bands}_{use_power}"
        if key not in self._cache_freq:
            self._cache_freq[key] = self._pitch_class_profile(use_power=use_power)
        return self._cache_freq[key]

    def _freq_key_templates(self):
        if "freq_key_templates_centered_norm" not in self._cache_freq:
            self._cache_freq["freq_key_templates_centered_norm"] = KEY_TEMPLATES_FREQ_NORM
        return self._cache_freq["freq_key_templates_centered_norm"]

    def _estimate_key_mode_freq(self, normalize: bool = True, use_power: bool = True, method: str = "correlation"):
        key = f"estimate_key_mode_freq_{normalize}_{use_power}_{method}"
        if key in self._cache_freq:
            return self._cache_freq[key]

        profile = self._pitch_class_profile(use_power=use_power).astype(float)
        if normalize:
            total = np.sum(profile)
            if total > EPS:
                profile /= total

        centered = profile - np.mean(profile)
        norm = np.linalg.norm(centered)
        if norm <= EPS:
            scores = np.zeros(24, dtype=float)
        else:
            scores = self._freq_key_templates() @ (centered / norm)

        key_idx = int(np.argmax(scores)) if scores.size else 0
        tonic = key_idx % 12
        mode = "major" if key_idx < 12 else "minor"
        score = float(scores[key_idx]) if scores.size else 0.0

        sorted_scores = np.sort(scores)[::-1] if scores.size else np.array([0.0])
        second = float(sorted_scores[1]) if sorted_scores.size > 1 else 0.0
        margin = max(0.0, score - second)
        confidence = safe_clip01(margin / 0.15)

        score_major = float(scores[tonic]) if scores.size >= 12 else 0.0
        score_minor = float(scores[tonic + 12]) if scores.size >= 24 else 0.0

        result = {
            "key_idx": key_idx,
            "tonic": tonic,
            "mode": mode,
            "score": score,
            "scores": scores,
            "score_major": score_major,
            "score_minor": score_minor,
            "delta_score": score_major - score_minor,
            "margin": margin,
            "confidence": confidence,
        }
        self._cache_freq[key] = result
        return result

    # ------------------------------------------------------------------
    # Frequency-domain feature evidence for later cross-domain fusion
    # ------------------------------------------------------------------
    def _frequency_weights(self):
        """Optional mild A-weighting-like gain for experiments, not Spotify energy."""
        if "frequency_weights" in self._cache_freq:
            return self._cache_freq["frequency_weights"]

        f = np.maximum(self.freqs.astype(float), 1.0)
        f2 = f ** 2
        ra_num = (12194.0 ** 2) * (f ** 4)
        ra_den = (
            (f2 + 20.6 ** 2)
            * np.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2))
            * (f2 + 12194.0 ** 2)
        )
        ra = ra_num / np.maximum(ra_den, EPS)
        A_db = 2.0 + 20.0 * np.log10(np.maximum(ra, EPS))
        gain = 10.0 ** (A_db / 20.0)
        gain /= max(float(np.max(gain)), EPS)
        self._cache_freq["frequency_weights"] = gain
        return gain

    def _rms_level_db(self) -> float:
        """Track RMS level proxy in dBFS-like units.

        This is a spectral reconstruction of signal energy, not Spotify loudness.
        """
        key = "rms_level_db"
        if key in self._cache_freq:
            return float(self._cache_freq[key])

        energy = self._frame_energy()
        if energy.size == 0:
            value = -80.0
        else:
            valid = energy > EPS
            if not np.any(valid):
                value = -80.0
            else:
                rms = float(np.sqrt(np.mean(energy[valid])))
                value = float(np.clip(20.0 * np.log10(rms + EPS), -80.0, 0.0))

        self._cache_freq[key] = value
        return value

    def _active_rms_level_db(self) -> float:
        """Upper-energy frame RMS level proxy; not perceptual loudness."""
        key = "active_rms_level_db"
        if key in self._cache_freq:
            return float(self._cache_freq[key])

        energy = self._frame_energy()
        if energy.size == 0 or not np.any(energy > EPS):
            value = -80.0
        else:
            positive = energy[energy > EPS]
            threshold = float(np.percentile(positive, 50.0))
            active = energy >= threshold
            rms = float(np.sqrt(np.mean(energy[active])))
            value = float(np.clip(20.0 * np.log10(rms + EPS), -80.0, 0.0))

        self._cache_freq[key] = value
        return value

    def _spectral_energy_evidence(self, weighted: bool = False) -> float:
        """Frequency-domain intensity/activity evidence in [0, 1]."""
        key = f"spectral_energy_evidence_{weighted}"
        if key in self._cache_freq:
            return float(self._cache_freq[key])

        if self._track_activity_confidence() <= 0.0:
            self._cache_freq[key] = 0.0
            return 0.0

        energy = self._frame_energy()
        if energy.size == 0:
            self._cache_freq[key] = 0.0
            return 0.0

        positive = energy[energy > EPS]
        if positive.size == 0:
            self._cache_freq[key] = 0.0
            return 0.0

        level_db = float(np.median(10.0 * np.log10(positive + EPS)))
        level_score = float(1.0 / (1.0 + np.exp(-(level_db + 24.0) / 6.0)))

        dr = self._dynamic_range()
        dynamic_score = float(1.0 - np.exp(-dr / 18.0))

        flux = self._spectral_flux(use_power=False, normalize=True, half_wave_rectify=True)
        flux_median = float(np.median(flux)) if flux.size else 0.0
        flux_p90 = float(np.percentile(flux, 90)) if flux.size else 0.0
        activity_score = float(np.clip(0.5 * (1.0 - np.exp(-5.0 * flux_median)) + 0.5 * (1.0 - np.exp(-3.0 * flux_p90)), 0.0, 1.0))

        transient = self._transient_rate(use_power=False)
        transient_score = safe_clip01(transient / (transient + 4.0))

        if weighted:
            S = self._power_spectrum()
            fw = self._frequency_weights()[:, None]
            weighted_energy = np.sum(S * fw, axis=0)
            weighted_db = float(np.median(10.0 * np.log10(weighted_energy + EPS)))
            weighted_level_score = float(1.0 / (1.0 + np.exp(-(weighted_db + 24.0) / 6.0)))
            level_component = 0.5 * level_score + 0.5 * weighted_level_score
        else:
            level_component = level_score

        value = (
            0.40 * level_component
            + 0.25 * activity_score
            + 0.20 * dynamic_score
            + 0.15 * transient_score
        )
        value = safe_clip01(value)
        self._cache_freq[key] = value
        return value

    def _speech_band_evidence(self) -> float:
        """Frequency evidence associated with speech-like spectral content.

        This is not a speech classifier; MFCC/vocal evidence should carry the
        primary decision during fusion.
        """
        key = "speech_band_evidence"
        if key in self._cache_freq:
            return float(self._cache_freq[key])

        if self._track_activity_confidence() <= 0.0:
            self._cache_freq[key] = 0.0
            return 0.0

        flat = float(np.mean(self._spectral_flatness()))
        entropy = float(np.mean(self._spectral_entropy()))
        centroid = float(np.mean(self._spectral_centroid()))
        nyq = float(self.sr) / 2.0
        mid_ratio = float(np.mean(self._band_ratios([(300.0, min(3400.0, nyq))])[0])) if nyq > 300 else 0.0
        harmonic = float(np.mean(self._harmonic_ratio()))

        centroid_score = float(np.exp(-0.5 * ((centroid - 1800.0) / 1500.0) ** 2))
        mid_score = safe_clip01(mid_ratio / 0.45)
        flat_score = safe_clip01(flat / 0.50)
        entropy_score = safe_clip01(entropy)
        harmonic_speech_score = 1.0 - safe_clip01(harmonic)

        value = (
            0.25 * mid_score
            + 0.20 * centroid_score
            + 0.20 * flat_score
            + 0.15 * entropy_score
            + 0.20 * harmonic_speech_score
        )
        self._cache_freq[key] = safe_clip01(value)
        return float(self._cache_freq[key])

    def _acoustic_timbre_evidence(self) -> float:
        """Spectral evidence associated with acoustic/instrument-like timbre.

        This is not a standalone acousticness classifier.
        """
        key = "acoustic_timbre_evidence"
        if key in self._cache_freq:
            return float(self._cache_freq[key])

        if self._track_activity_confidence() <= 0.0:
            self._cache_freq[key] = 0.0
            return 0.0

        flat = float(np.mean(self._spectral_flatness()))
        centroid = float(np.mean(self._spectral_centroid()))
        roll = float(np.mean(self._spectral_rolloff()))
        harmonic = float(np.mean(self._harmonic_ratio()))
        high_ratio = float(np.mean(self._band_ratios([(6000.0, float(self.sr) / 2.0)])[0])) if self.sr / 2.0 > 6000 else 0.0
        inharm = float(np.mean(self._inharmonicity()))

        nyq = max(float(self.sr) / 2.0, 1.0)
        low_brightness = 1.0 - safe_clip01(centroid / nyq)
        low_roll = 1.0 - safe_clip01(roll / nyq)
        low_high = 1.0 - safe_clip01(high_ratio)
        tonal = safe_clip01(harmonic)
        inharmonic_score = 1.0 - safe_clip01(inharm)
        non_noise = 1.0 - safe_clip01(flat)

        value = (
            0.20 * low_brightness
            + 0.15 * low_roll
            + 0.20 * low_high
            + 0.25 * tonal
            + 0.10 * inharmonic_score
            + 0.10 * non_noise
        )
        self._cache_freq[key] = safe_clip01(value)
        return float(self._cache_freq[key])

    def _rhythmic_spectral_evidence(self, tempo: float | None = None) -> float:
        """Frequency-domain rhythmic evidence for later danceability fusion."""
        key = f"rhythmic_spectral_evidence_{'none' if tempo is None else float(tempo):.6f}"
        if key in self._cache_freq:
            return float(self._cache_freq[key])

        if self._track_activity_confidence() <= 0.0:
            self._cache_freq[key] = 0.0
            return 0.0

        if tempo is None:
            tempo = self._spectral_tempo_evidence()

        pulse = self._pulse_clarity_ac()
        periodicity = self._beat_periodicity()
        flux = self._spectral_flux(use_power=False, normalize=True, half_wave_rectify=True)
        flux_consistency = 0.0
        if flux.size > 2:
            med = float(np.median(flux))
            mad = float(np.median(np.abs(flux - med))) + EPS
            flux_consistency = float(1.0 - np.clip(mad / (med + 0.1), 0.0, 1.0))

        # Broad tempo compatibility is intentionally low-weight.  The canonical
        # tempo/beat estimate should ultimately come from the Tempogram domain.
        if tempo > 0:
            tempo_score = float(np.exp(-0.5 * (np.log2(max(tempo, 1.0) / 120.0) / 0.9) ** 2))
        else:
            tempo_score = 0.0

        value = (
            0.35 * pulse
            + 0.30 * periodicity
            + 0.20 * flux_consistency
            + 0.15 * tempo_score
        )
        self._cache_freq[key] = safe_clip01(value)
        return float(self._cache_freq[key])

    def _brightness_valence_evidence(self) -> float:
        """Brightness/timbral evidence that may weakly support valence.

        Chroma/tonal information should carry the main valence signal.
        """
        key = "brightness_valence_evidence"
        if key in self._cache_freq:
            return float(self._cache_freq[key])

        if self._track_activity_confidence() <= 0.0:
            self._cache_freq[key] = 0.0
            return 0.0

        centroid = float(np.mean(self._spectral_centroid()))
        harmonic = float(np.mean(self._harmonic_ratio()))
        flat = float(np.mean(self._spectral_flatness()))
        roll = float(np.mean(self._spectral_rolloff()))

        bright = safe_clip01(centroid / (0.45 * max(self.sr / 2.0, 1.0)))
        roll_score = safe_clip01(roll / (0.55 * max(self.sr / 2.0, 1.0)))
        tonal = safe_clip01(harmonic)
        non_flat = 1.0 - safe_clip01(flat)

        value = 0.35 * bright + 0.25 * roll_score + 0.25 * tonal + 0.15 * non_flat
        self._cache_freq[key] = safe_clip01(value)
        return float(self._cache_freq[key])

    def _flux_implied_bpm(self, bpm_min: float = 40.0, bpm_max: float = 240.0) -> float:
        candidates = self._tempo_peak_candidates(bpm_min, bpm_max)
        if not candidates:
            return 0.0
        return float(candidates[0]["bpm"])

    def _spectral_tempo_evidence(self) -> float:
        """Frequency-domain tempo estimate used as supporting evidence.

        TempogramFeatures should own the final tempo estimate.
        """
        key = "spectral_tempo_evidence"
        if key in self._cache_freq:
            return float(self._cache_freq[key])

        candidates = self._tempo_peak_candidates(40.0, 240.0)
        if not candidates:
            self._cache_freq[key] = 0.0
            return 0.0

        # Evaluate octave/harmonic alternatives around the strongest candidate,
        # but choose based on actual local AC strength rather than a hard-coded
        # preference for 80-160 BPM.
        ac = self._rhythmic_autocorrelation()
        fs_env = self.sr / float(self.H)
        base = candidates[0]
        candidate_bpms = [
            base["bpm"],
            base["bpm"] / 2.0,
            base["bpm"] * 2.0,
            base["bpm"] * 2.0 / 3.0,
            base["bpm"] * 3.0 / 2.0,
        ]

        scored = []
        for bpm in candidate_bpms:
            if not 40.0 <= bpm <= 240.0:
                continue
            lag = fs_env * 60.0 / bpm
            if lag <= 1.0 or lag >= ac.size - 1:
                continue
            center = int(round(lag))
            radius = max(2, int(round(0.05 * lag)))
            lo = max(1, center - radius)
            hi = min(ac.size - 1, center + radius)
            strength = float(np.max(ac[lo : hi + 1]))
            scored.append((strength, float(bpm)))

        if not scored:
            value = float(base["bpm"])
        else:
            scored.sort(reverse=True)
            value = scored[0][1]

        self._cache_freq[key] = float(np.clip(value, 40.0, 240.0))
        return float(self._cache_freq[key])

    def _performance_variability_evidence(self) -> float:
        """Spectral variability associated with performance/noise variation.

        This does not detect an audience or establish liveness by itself.
        """
        key = "performance_variability_evidence"
        if key in self._cache_freq:
            return float(self._cache_freq[key])

        if self._track_activity_confidence() <= 0.0:
            self._cache_freq[key] = 0.0
            return 0.0

        flux = self._spectral_flux(use_power=False, normalize=True, half_wave_rectify=True)
        transient = self._transient_rate(use_power=False)
        flat = float(np.mean(self._spectral_flatness()))
        high_ratio = 0.0
        if self.sr / 2.0 > 4000.0:
            high_ratio = float(np.mean(self._band_ratios([(4000.0, self.sr / 2.0)])[0]))

        flux_var = float(np.std(flux)) if flux.size else 0.0
        flux_var_score = safe_clip01(flux_var / 0.20)
        transient_score = safe_clip01(transient / (transient + 5.0))

        value = (
            0.30 * transient_score
            + 0.20 * flux_var_score
            + 0.20 * safe_clip01(high_ratio / 0.25)
            + 0.15 * flat
            + 0.15 * self._beat_periodicity()
        )
        self._cache_freq[key] = safe_clip01(value)
        return float(self._cache_freq[key])

    def _non_vocal_spectral_evidence(self) -> float:
        """Spectral evidence that may support a non-vocal/instrumental hypothesis.

        It is not a vocal detector and should be subordinate to dedicated vocal
        evidence during fusion.
        """
        key = "non_vocal_spectral_evidence"
        if key in self._cache_freq:
            return float(self._cache_freq[key])

        if self._track_activity_confidence() <= 0.0:
            self._cache_freq[key] = 0.0
            return 0.0

        harmonic = float(np.mean(self._harmonic_ratio()))
        flat = float(np.mean(self._spectral_flatness()))
        inharm = float(np.mean(self._inharmonicity()))
        mid_ratio = float(np.mean(self._band_ratios([(300.0, min(3400.0, self.sr / 2.0))])[0])) if self.sr / 2.0 > 300 else 0.0

        tonal_score = safe_clip01(harmonic)
        non_noise = 1.0 - safe_clip01(flat)
        inharm_score = 1.0 - safe_clip01(inharm)
        non_speech_band = 1.0 - safe_clip01(mid_ratio / 0.45)

        value = (
            0.35 * tonal_score
            + 0.25 * non_noise
            + 0.15 * inharm_score
            + 0.25 * non_speech_band
        )
        self._cache_freq[key] = safe_clip01(value)
        return float(self._cache_freq[key])

    def _key_frequency_evidence(self):
        key = "key_frequency_evidence"
        if key not in self._cache_freq:
            self._cache_freq[key] = self._estimate_key_mode_freq()["tonic"]
        return int(self._cache_freq[key])

    def _mode_frequency_evidence(self):
        key = "mode_frequency_evidence"
        if key not in self._cache_freq:
            self._cache_freq[key] = self._estimate_key_mode_freq()["mode"]
        return self._cache_freq[key]

    def _meter_frequency_evidence(self) -> dict:
        """
        Conservative frequency-domain meter estimate.

        Candidates are 3/4 through 7/4.  The estimated tempo sets the beat
        duration; a weak/ambiguous estimate falls back to 4/4.
        """
        key = "meter_frequency_evidence"
        if key in self._cache_freq:
            return int(self._cache_freq[key])

        ac = self._rhythmic_autocorrelation()
        tempo = self._spectral_tempo_evidence()
        if ac.size < 4 or tempo <= 0.0:
            out = {"time_signature": 4, "confidence": 0.0, "structure_score": 0.0}
            self._cache_freq[key] = out
            return out

        fs_env = self.sr / float(self.H)
        beat_frames = fs_env * 60.0 / tempo
        candidates = range(3, 8)
        scores = {}

        for meter in candidates:
            lag = beat_frames * meter
            if lag <= 1.0 or lag >= ac.size - 1:
                scores[meter] = -np.inf
                continue

            center = int(round(lag))
            radius = max(2, int(round(0.04 * lag)))
            lo = max(1, center - radius)
            hi = min(ac.size - 1, center + radius)
            measure_peak = float(np.max(ac[lo : hi + 1]))

            # Compare the measure lag to the surrounding one-beat multiples.
            neighboring = []
            for multiple in range(max(1, meter - 1), min(7, meter + 1) + 1):
                lag_n = beat_frames * multiple
                c_n = int(round(lag_n))
                if 1 <= c_n < ac.size:
                    r_n = max(1, int(round(0.04 * lag_n)))
                    lo_n = max(1, c_n - r_n)
                    hi_n = min(ac.size - 1, c_n + r_n)
                    neighboring.append(float(np.max(ac[lo_n : hi_n + 1])))

            baseline = float(np.median(neighboring)) + EPS if neighboring else EPS
            scores[meter] = measure_peak / baseline

        finite_scores = {m: s for m, s in scores.items() if np.isfinite(s)}
        if not finite_scores:
            out = {"time_signature": 4, "confidence": 0.0, "structure_score": 0.0}
        else:
            ranked = sorted(finite_scores.items(), key=lambda item: item[1], reverse=True)
            best_meter, best_score = ranked[0]
            second_score = ranked[1][1] if len(ranked) > 1 else 0.0

            # A frequency-only meter estimate is weak unless the winning
            # periodicity clearly separates from alternatives.
            separation = max(0.0, float(best_score - second_score))
            confidence = safe_clip01(separation / (abs(float(best_score)) + EPS))

            if best_meter != 4 and best_score < 1.10 * max(second_score, EPS):
                meter = 4
            else:
                meter = int(best_meter)

            out = {
                "time_signature": meter,
                "confidence": float(confidence),
                "structure_score": float(max(best_score, 0.0)),
            }

        self._cache_freq[key] = out
        return out

    # ------------------------------------------------------------------
    # Public frequency-domain evidence exporter
    # ------------------------------------------------------------------
    def frequency_domain_evidence(
        self,
        primary_bpm: float | None = None,
        global_loudness_db: float | None = None,
    ) -> dict:
        """Return frequency-domain evidence for central feature fusion.

        The returned values are deliberately domain-specific.  They are not
        claimed to be Spotify's proprietary Audio Features outputs.  Features
        that frequency-domain analysis cannot establish reliably are exposed
        only as supporting evidence or with explicit confidence.
        """
        loudness_db = (
            float(global_loudness_db)
            if global_loudness_db is not None
            else self._rms_level_db()
        )

        spectral_tempo = (
            float(primary_bpm)
            if primary_bpm is not None and primary_bpm > 0.0
            else self._spectral_tempo_evidence()
        )

        key_mode = self._estimate_key_mode_freq()
        meter = self._meter_frequency_evidence()
        activity = float(self._track_activity_confidence())

        return {
            # Direct spectral level/intensity evidence.
            "rms_level_db": loudness_db,
            "active_rms_level_db": self._active_rms_level_db(),
            "spectral_energy_evidence": self._spectral_energy_evidence(),

            # Supporting evidence for later learned/model-based features.
            "speech_band_evidence": self._speech_band_evidence(),
            "acoustic_timbre_evidence": self._acoustic_timbre_evidence(),
            "rhythmic_spectral_evidence": self._rhythmic_spectral_evidence(
                tempo=spectral_tempo
            ),
            "brightness_valence_evidence": self._brightness_valence_evidence(),
            "performance_variability_evidence": self._performance_variability_evidence(),
            "non_vocal_spectral_evidence": self._non_vocal_spectral_evidence(),

            # Tempo is supporting evidence only; TempogramFeatures should be
            # authoritative for the final tempo estimate.
            "spectral_tempo_bpm": spectral_tempo,

            # Frequency-domain key/mode evidence is useful, but Chroma should
            # normally be the primary tonal source in the final fusion.
            "key_tonic": int(key_mode["tonic"]),
            "mode": key_mode["mode"],
            "key_confidence": float(key_mode["confidence"]),
            "key_margin": float(key_mode["margin"]),
            "key_score_major": float(key_mode["score_major"]),
            "key_score_minor": float(key_mode["score_minor"]),

            # Meter evidence is returned with confidence rather than a bare
            # integer so weak frequency evidence cannot masquerade as certainty.
            "meter": int(meter["time_signature"]),
            "meter_confidence": float(meter["confidence"]),
            "meter_structure_score": float(meter["structure_score"]),

            "activity_confidence": activity,
        }

    def spotify_audio_features(
        self,
        weights=None,
        primary_bpm=None,
        global_loudness_db=None,
    ) -> dict:
        """Backward-compatible alias for :meth:`frequency_domain_evidence`.

        ``weights`` is accepted for older callers but deliberately ignored.
        Cross-domain weighting belongs in the central fusion/model layer.
        """
        del weights
        return self.frequency_domain_evidence(
            primary_bpm=primary_bpm,
            global_loudness_db=global_loudness_db,
        )

    # ------------------------------------------------------------------
    # Backward-compatible aliases
    # ------------------------------------------------------------------
    # These aliases preserve existing callers while making the new public API
    # semantically explicit.  New code should use the evidence-oriented names.
    def _loudness_freq_db(self) -> float:
        return self._rms_level_db()

    def _loudness_freq_active_db(self) -> float:
        return self._active_rms_level_db()

    def _energy_freq(self, weighted: bool = False) -> float:
        return self._spectral_energy_evidence(weighted=weighted)

    def _speechiness_freq(self) -> float:
        return self._speech_band_evidence()

    def _acousticness_freq(self) -> float:
        return self._acoustic_timbre_evidence()

    def _danceability_freq(self, tempo: float | None = None) -> float:
        return self._rhythmic_spectral_evidence(tempo=tempo)

    def _valence_freq(self) -> float:
        return self._brightness_valence_evidence()

    def _tempo_freq(self) -> float:
        return self._spectral_tempo_evidence()

    def _liveness_freq(self) -> float:
        return self._performance_variability_evidence()

    def _instrumentalness_freq(self) -> float:
        return self._non_vocal_spectral_evidence()

    def _key_freq(self):
        return self._key_frequency_evidence()

    def _mode_freq(self):
        return self._mode_frequency_evidence()

    def _time_signature_freq(self) -> int:
        return int(self._meter_frequency_evidence()["time_signature"])
