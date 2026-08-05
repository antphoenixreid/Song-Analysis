from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

import numpy as np

from audio_features.audio_signal import AudioSignal
from audio_features.feature_extractor import FeatureExtractor
from audio_features.utils import EPS, safe_clip01, _safe_float

@dataclass
class SpotifyFusion:
    sig: AudioSignal
    compute_time: bool = True
    compute_frequency: bool = True
    compute_mfcc: bool = True
    compute_chroma: bool = True
    compute_tempogram: bool = True
    temperature: float = 1.0

    def __post_init__(self):
        self._features = FeatureExtractor(
            self.sig,
            compute_time=self.compute_time,
            compute_frequency=self.compute_frequency,
            compute_mfcc=self.compute_mfcc,
            compute_chroma=self.compute_chroma,
            compute_tempogram=self.compute_tempogram
        )

    @staticmethod
    def from_audio(y: np.ndarray, sr: int, n_fft: int = 2048, hop_length: int = 512, **kwargs):
        sig = AudioSignal(signal=y, sr=sr, N=n_fft, H=hop_length)
        return SpotifyFusion(sig, **kwargs)
    
    @staticmethod
    def _first_existing(d: Dict[str, Any], keys: list[str]) -> Any:
        for k in keys:
            if k in d and d[k] is not None:
                return d[k]
        return None
    
    @staticmethod
    def _softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        t = max(float(temperature), 1e-8)
        z = x/t
        z = z - np.max(z)
        e = np.exp(z)

        return e/(np.sum(e))
    
    @staticmethod
    def _weighted_softmax_fuse(values: Sequence[Any], logits: Sequence[float], temperature: float = 1.0, clip01: bool = True):
        vals = np.array([_safe_float(v, default=np.nan) for v in values], dtype=float)
        logits = np.array(logits, dtype=float)

        mask = np.isfinite(vals) & np.isfinite(logits)
        if not np.any(mask):
            return 0.0, np.array([], dtype=float)
        
        vals = vals[mask]
        logits = logits[mask]
        w = SpotifyFusion._softmax(logits, temperature=temperature)
        fused = float(np.sum(w*vals))
        if clip01:
            fused = _safe_float(safe_clip01(fused), default=0.0)

        return fused, w
    
    def extract(self) -> Dict[str, Any]:
        feats = self._features.extract()

        sources = {
            "loudness": [
                ("loudness", 3.0),
                ("time.loudness", 2.5),
                ("frequency.loudness", 2.2),
                ("tempogram.loudness", 1.0),
                ("mfcc.loudness", 1.4)
            ],
            "energy": [
                ("energy", 3.0),
                ("time.energy", 2.6),
                ("frequency.energy", 2.4),
                ("mfcc.energy", 1.6),
                ("chroma.energy", 1.0)
            ],
            "speechiness": [
                ("speechiness", 3.0),
                ("mfcc.speechiness", 2.7),
                ("time.speechiness", 1.8),
                ("frequency.speechiness", 1.7),
                ("chroma.speechiness", 1.0)
            ],
            "acousticness": [
                ("acousticness", 3.0),
                ("frequency.acousticness", 2.5),
                ("mfcc.acousticness", 2.0),
                ("time.acousticness", 1.5),
                ("chroma.acousticness", 1.2)
            ],
            "danceability": [
                ("danceability", 3.0),
                ("time.danceability", 2.8),
                ("tempogram.danceability", 2.4),
                ("frequency.danceability", 1.8),
                ("chroma.danceability", 1.2)
            ],
            "valence": [
                ("valence", 3.0),
                ("chroma.valence", 2.8),
                ("tempogram.valence", 2.0),
                ("frequency.valence", 1.8),
                ("mfcc.valence", 1.3)
            ],
            "tempo": [
                ("tempo", 3.0),
                ("tempogram.tempo", 2.9),
                ("time.tempo", 2.5),
                ("frequency.tempo", 1.2),
                ("chroma.tempo", 1.0)
            ],
            "liveness": [
                ("liveness", 3.0),
                ("frequency.liveness", 2.7),
                ("time.liveness", 2.0),
                ("mfcc.liveness", 1.8),
                ("tempogram.liveness", 1.0)
            ],
            "instrumentalness": [
                ("instrumentalness", 3.0),
                ("frequency.instrumentalness", 2.7),
                ("mfcc.instrumentalness", 2.3),
                ("time.instrumentalness", 1.8),
                ("chroma.instrumentalness", 1.2)
            ]
        }

        out = {}

        for name, opts in sources.items():
            keys = [k for k, _ in opts]
            vals = [feats.get(k) for k in keys]
            logits = [w for _, w in opts]
            fused, weights = self._weighted_softmax_fuse(vals, logits, temperature=self.temperature, clip01=(name != "tempo"))
            if name == "tempo":
                out[name] = float(fused*240.0)
            elif name == "loudness":
                out[name] = float(fused*80.0 - 80.0)
            else:
                out[name] = float(fused)

        key_val = self._first_existing(feats, ["key", "chroma.key", "frequency.key"])
        mode_val = self._first_existing(feats, ["mode", "chroma.mode", "frequency.mode", "tempogram.mode"])
        ts_val = self._first_existing(feats, ["time_signature", "chroma.time_signature", "frequency.time_signature", "tempogram.time_signature"])

        out["key"] = int(key_val) if key_val is not None else -1
        _MODE_MAP = {"maj": 1, "major": 1, "min": 0, "minor": 0}
        out["mode"] = _MODE_MAP.get(str(mode_val).lower(), -1) if mode_val is not None else -1
        out["time_signature"] = int(ts_val) if ts_val is not None else 4

        out["spotify_score"] = float(np.clip(
            0.15 * out["energy"] +
            0.15 * out["danceability"] +
            0.12 * out["valence"] +
            0.12 * out["acousticness"] +
            0.10 * out["speechiness"] +
            0.10 * out["liveness"] +
            0.08 * out["instrumentalness"] +
            0.08 * np.clip(out["tempo"] / 240.0, 0.0, 1.0) +
            0.10 * np.clip((out["loudness"] + 80.0) / 80.0, 0.0, 1.0),
            0.0, 1.0
        ))

        return out