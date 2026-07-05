"""
Chromagram and pitch-class features.
"""

import numpy as np
import librosa
from .utils import EPS, safe_clip01
from .audio_signal import AudioSignal


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
        n_bins = len(self.freqs)
        fb_matr = np.zeros((n_pitches, n_bins), dtype=float)

        for p in range(n_pitches):
            center = self.midi_to_hz(p)
            lower, upper = self._pitch_bin_edges(p)

            mask = (self.freqs >= lower) & (self.freqs < upper)
            fb_matr[p, mask] = 1.0

        Y_LF = fb_matr @ self.X_mag
        return Y_LF         
    
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
        
        Y_LF = self._compute_spec_log_freq()
        C = self._compute_chromagram(Y_LF)

        self._cache_chroma[key] = C
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
        C = self._chroma()                          # always start linear
        if normalize:
            colsum = np.sum(C, axis=0, keepdims=True)
            zero_cols = colsum <= EPS
            C = C / (colsum + EPS)
            if np.any(zero_cols):
                C[:, zero_cols[0]] = 1.0 / 12.0
        if use_db:
            C = 10.0 * np.log10(C + EPS)           # convert after normalization
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
        abs_dist = np.minimum(np.abs(idx - centroid_idx[None, :]), 12 - np.abs(idx - centroid_idx[None, :]))
        signed_dist = (idx - centroid_idx[None, :] + 6) % 12 - 6  # wraps to [-6, 6)

        m2 = np.sum((abs_dist**2)*P, axis=0)
        m3 = np.sum((signed_dist**3)*P, axis=0)   # signed — allows negative skew

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
        H_norm = H / np.log(len(scores))

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

        if len(chord_labels) <= 1:
            label_list = list(chord_labels) if len(chord_labels) == 1 else []
            result = {
                "labels": label_list,
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
                "durations_sec": np.array([], dtype=float),
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
        key = f"tonal_stability_index_{'norm' if normalize else 'raw'}_{'db' if use_db else 'lin'}_{method}"
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
        T = self._chroma_template()

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

        # fit = self._harmonic_template_fit(normalize=normalize, use_db=use_db, method="dot")["best_score"]
        fit_res = self._harmonic_template_fit(normalize=normalize, use_db=use_db, method="dot")
        fit = fit_res["best_score"] if isinstance(fit_res, dict) else float(fit_res)

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
        templates = self._chroma_template()
        maj_score = float(key_res["scores"][tonic])
        min_score = float(key_res["scores"][tonic + 12])
        clarity = self._tonal_clarity(normalize=normalize, use_db=use_db, method="cosine")["tonal_clarity"]
        fit = self._harmonic_template_fit(normalize=normalize, use_db=use_db, method="cosine")["best_score"]
        bright = float((prof[tonic % 12] + prof[(tonic + 4)%12] + prof[(tonic + 7)%12] + prof[(tonic + 11)%12]) / (np.sum(prof) + EPS))
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
            "key": key_res["key_idx"],
            "mode": mode_res["mode"],
            "time_signature": self._time_signature_chroma(normalize=normalize, use_db=use_db),
        }