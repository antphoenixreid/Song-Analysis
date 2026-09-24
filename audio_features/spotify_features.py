from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np

from audio_features.audio_signal import AudioSignal
from audio_features.feature_extractor import FeatureExtractor


@dataclass
class SpotifyFusion:
    """Central fusion layer for the project's Spotify-style feature vector.

    The values produced here are custom DSP/model outputs. They are not claims
    to reproduce Spotify's proprietary Audio Features model.

    Feature ownership is intentionally centralized:
      - loudness: time + frequency
      - energy: time + frequency
      - speechiness: MFCC + frequency + time
      - acousticness: frequency + MFCC + time
      - danceability: tempogram + time
      - valence: chroma + frequency + tempogram
      - tempo: tempogram
      - liveness: frequency + time + MFCC + tempogram
      - instrumentalness: frequency + time + chroma support
      - key/mode: chroma
      - time_signature: tempogram
    """

    sig: AudioSignal
    compute_time: bool = True
    compute_frequency: bool = True
    compute_mfcc: bool = True
    compute_chroma: bool = True
    compute_tempogram: bool = True

    # These are ordinary evidence weights, not softmax logits. They are kept
    # explicit so the fusion policy can be inspected and tuned centrally.
    weights: Mapping[str, Mapping[str, float]] | None = None

    def __post_init__(self) -> None:
        self._features = FeatureExtractor(
            self.sig,
            compute_time=self.compute_time,
            compute_frequency=self.compute_frequency,
            compute_mfcc=self.compute_mfcc,
            compute_chroma=self.compute_chroma,
            compute_tempogram=self.compute_tempogram,
        )

        self._weights = self._merge_weights(self.weights)

    @staticmethod
    def from_audio(
        y: np.ndarray,
        sr: int,
        n_fft: int = 2048,
        hop_length: int = 512,
        **kwargs: Any,
    ) -> "SpotifyFusion":
        sig = AudioSignal(signal=y, sr=sr, N=n_fft, H=hop_length)
        return SpotifyFusion(sig, **kwargs)

    @staticmethod
    def _default_weights() -> Dict[str, Dict[str, float]]:
        """Return feature-specific domain weights.

        These weights encode ownership/support, not an imitation of Spotify's
        proprietary model. A domain that does not produce a field is simply
        omitted during fusion.
        """
        return {
            "loudness": {
                "time.loudness_db_time": 0.60,
                "frequency.rms_level_db": 0.40,
            },
            "energy": {
                "time.energy_time": 0.60,
                "frequency.spectral_energy_evidence": 0.40,
            },
            "speechiness": {
                "mfcc.temporal_complexity": 0.45,
                "frequency.speech_band_evidence": 0.35,
                "time.speechiness_time": 0.20,
            },
            "acousticness": {
                "frequency.acoustic_timbre_evidence": 0.45,
                "mfcc.timbre_smoothness": 0.35,
                "time.acousticness_time": 0.20,
            },
            "danceability": {
                "tempogram.rhythmic_drive": 0.55,
                "time.danceability_time": 0.30,
                "frequency.rhythmic_spectral_evidence": 0.15,
            },
            "valence": {
                "chroma.harmonic_valence_proxy": 0.60,
                "frequency.brightness_valence_evidence": 0.25,
                "tempogram.rhythmic_coherence": 0.15,
            },
            "liveness": {
                "frequency.performance_variability_evidence": 0.45,
                "time.performance_variability_time": 0.30,
                "mfcc.temporal_variability": 0.15,
                "tempogram.performance_variability": 0.10,
            },
            "instrumentalness": {
                "frequency.non_vocal_spectral_evidence": 0.60,
                "time.instrumentalness_time": 0.25,
                "chroma.tonal_focus_evidence": 0.15,
            },
        }

    @classmethod
    def _merge_weights(
        cls,
        custom: Mapping[str, Mapping[str, float]] | None,
    ) -> Dict[str, Dict[str, float]]:
        merged = {
            name: dict(values)
            for name, values in cls._default_weights().items()
        }

        if custom is not None:
            for feature, values in custom.items():
                if feature not in merged:
                    merged[feature] = {}
                merged[feature].update(
                    {str(key): float(value) for key, value in values.items()}
                )

        for feature, values in merged.items():
            values = {
                key: max(float(weight), 0.0)
                for key, weight in values.items()
            }
            total = sum(values.values())
            if total > 0.0:
                merged[feature] = {
                    key: weight / total for key, weight in values.items()
                }
            else:
                merged[feature] = values

        return merged

    @staticmethod
    def _finite(value: Any) -> float | None:
        if value is None or isinstance(value, (dict, list, tuple)):
            return None
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return result if np.isfinite(result) else None

    @classmethod
    def _weighted_mean(
        cls,
        evidence: Mapping[str, Any],
        weights: Mapping[str, float],
        confidence: Mapping[str, Any] | None = None,
        clip01: bool = True,
    ) -> float | None:
        """Fuse available evidence while skipping missing/non-finite values."""
        numerator = 0.0
        denominator = 0.0

        for key, base_weight in weights.items():
            value = cls._finite(evidence.get(key))
            if value is None or base_weight <= 0.0:
                continue

            conf = 1.0
            if confidence is not None and key in confidence:
                conf_value = cls._finite(confidence[key])
                if conf_value is not None:
                    conf = float(np.clip(conf_value, 0.0, 1.0))

            effective_weight = float(base_weight) * conf
            if effective_weight <= 0.0:
                continue

            numerator += effective_weight * value
            denominator += effective_weight

        if denominator <= 0.0:
            return None

        result = numerator / denominator
        if clip01:
            result = float(np.clip(result, 0.0, 1.0))
        return float(result)

    @staticmethod
    def _clip01(value: Any) -> float | None:
        value = SpotifyFusion._finite(value)
        return None if value is None else float(np.clip(value, 0.0, 1.0))

    @staticmethod
    def _confidence_from_margin(margin: Any) -> float:
        value = SpotifyFusion._finite(margin)
        if value is None:
            return 0.0
        return float(np.clip(value, 0.0, 1.0))

    @staticmethod
    def _select_categorical(
        candidates: Sequence[tuple[Any, float]],
        valid: Iterable[int] | None = None,
    ) -> int | None:
        valid_set = set(valid) if valid is not None else None
        scored: list[tuple[int, float]] = []
        for value, confidence in candidates:
            if value is None:
                continue
            try:
                ivalue = int(value)
            except (TypeError, ValueError):
                continue
            if valid_set is not None and ivalue not in valid_set:
                continue
            conf = float(np.clip(confidence, 0.0, 1.0))
            scored.append((ivalue, conf))

        if not scored:
            return None
        return max(scored, key=lambda item: item[1])[0]

    def _fuse_loudness(self, evidence: Mapping[str, Any]) -> float | None:
        return self._weighted_mean(
            evidence,
            self._weights["loudness"],
            clip01=False,
        )

    def _fuse_energy(self, evidence: Mapping[str, Any]) -> float | None:
        return self._weighted_mean(evidence, self._weights["energy"])

    def _fuse_speechiness(self, evidence: Mapping[str, Any]) -> float | None:
        return self._weighted_mean(evidence, self._weights["speechiness"])

    def _fuse_acousticness(self, evidence: Mapping[str, Any]) -> float | None:
        return self._weighted_mean(evidence, self._weights["acousticness"])

    def _fuse_danceability(self, evidence: Mapping[str, Any]) -> float | None:
        return self._weighted_mean(evidence, self._weights["danceability"])

    def _fuse_valence(self, evidence: Mapping[str, Any]) -> float | None:
        return self._weighted_mean(evidence, self._weights["valence"])

    def _fuse_liveness(self, evidence: Mapping[str, Any]) -> float | None:
        return self._weighted_mean(evidence, self._weights["liveness"])

    def _fuse_instrumentalness(self, evidence: Mapping[str, Any]) -> float | None:
        return self._weighted_mean(evidence, self._weights["instrumentalness"])

    def _fuse_tempo(self, evidence: Mapping[str, Any]) -> float | None:
        """Use Tempogram as the authoritative tempo source.

        The secondary candidate is retained as octave evidence, not averaged
        into BPM. If primary confidence is weak, the stronger candidate is used
        only when it has a measurable strength advantage.
        """
        bpm = self._finite(evidence.get("tempogram.tempo_bpm"))
        confidence = self._clip01(evidence.get("tempogram.tempo_confidence"))
        strength = self._finite(evidence.get("tempogram.tempo_strength"))

        secondary = self._finite(evidence.get("tempogram.tempo_secondary_bpm"))
        secondary_strength = self._finite(
            evidence.get("tempogram.tempo_secondary_strength")
        )

        if bpm is None or bpm <= 0.0:
            if secondary is not None and secondary > 0.0:
                return float(secondary)
            return None

        if secondary is None or secondary <= 0.0:
            return float(bpm)

        # Do not replace a clearly supported primary tempo.
        if confidence is None or confidence >= 0.50:
            return float(bpm)

        if strength is not None and secondary_strength is not None:
            ratio = secondary_strength / max(strength, 1e-12)
            if ratio >= 1.05:
                return float(secondary)

        # If candidates are an octave apart, preserve the primary estimate.
        # This avoids arbitrary doubling/halving rules in the fusion layer.
        return float(bpm)

    def _fuse_key(self, evidence: Mapping[str, Any]) -> int | None:
        key = evidence.get("chroma.key_tonic")
        confidence = self._clip01(evidence.get("chroma.key_confidence"))
        if key is None or confidence is None or confidence <= 0.0:
            return None
        try:
            key = int(key) % 12
        except (TypeError, ValueError):
            return None
        return key

    def _fuse_mode(self, evidence: Mapping[str, Any]) -> int | None:
        mode = self._finite(evidence.get("chroma.mode_value"))
        confidence = self._clip01(evidence.get("chroma.mode_confidence"))
        if mode is None or confidence is None or confidence <= 0.0:
            return None
        return int(1 if mode >= 0.5 else 0)

    def _fuse_time_signature(self, evidence: Mapping[str, Any]) -> int | None:
        meter = evidence.get("tempogram.time_signature")
        confidence = self._clip01(evidence.get("tempogram.time_signature_confidence"))
        if meter is None or confidence is None or confidence <= 0.0:
            return None
        try:
            meter = int(meter)
        except (TypeError, ValueError):
            return None
        return meter if meter > 0 else None

    def extract(self) -> Dict[str, Any]:
        """Return the final custom Spotify-style feature vector."""
        evidence = self._features.extract()

        out: Dict[str, Any] = {
            "danceability": self._fuse_danceability(evidence),
            "energy": self._fuse_energy(evidence),
            "key": self._fuse_key(evidence),
            "loudness": self._fuse_loudness(evidence),
            "mode": self._fuse_mode(evidence),
            "speechiness": self._fuse_speechiness(evidence),
            "acousticness": self._fuse_acousticness(evidence),
            "instrumentalness": self._fuse_instrumentalness(evidence),
            "liveness": self._fuse_liveness(evidence),
            "valence": self._fuse_valence(evidence),
            "tempo": self._fuse_tempo(evidence),
            "time_signature": self._fuse_time_signature(evidence),
        }

        # Keep probability-like outputs in their documented range without
        # manufacturing values when evidence is unavailable.
        for name in (
            "danceability",
            "energy",
            "speechiness",
            "acousticness",
            "instrumentalness",
            "liveness",
            "valence",
        ):
            if out[name] is not None:
                out[name] = float(np.clip(out[name], 0.0, 1.0))

        if out["tempo"] is not None:
            out["tempo"] = float(np.clip(out["tempo"], 30.0, 300.0))

        if out["loudness"] is not None:
            out["loudness"] = float(np.clip(out["loudness"], -80.0, 0.0))

        return out

    def spotify_audio_features(self) -> Dict[str, Any]:
        """Compatibility alias for :meth:`extract`."""
        return self.extract()
