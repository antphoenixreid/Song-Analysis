"""
Frequency domain audio features.
"""

import numpy as np
import librosa
from scipy.signal import find_peaks
from scipy.ndimage import maximum_filter
from .utils import EPS, safe_clip01
from audio_features.audio_signal import AudioSignal


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

    def _safe_band_mask(self, f_lo, f_hi):
        return (self.freqs >= f_lo) & (self.freqs < f_hi)

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
        return (self.freqs >= f_lo) & (self.freqs <= f_hi)
    
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