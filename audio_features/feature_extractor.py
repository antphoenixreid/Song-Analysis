from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from audio_features.audio_signal import AudioSignal
from audio_features.time_features import TimeFeatures
from audio_features.frequency_features import FrequencyFeatures
from audio_features.chromagram_features import ChromagramFeatures
from audio_features.tempogram_features import TempogramFeatures
from audio_features.mfcc_features import MFCCFeatures
from audio_features.utils import EPS, safe_clip01, _safe_float

@dataclass
class FeatureExtractor:
    sig: AudioSignal
    compute_time: bool = True
    compute_frequency: bool = True
    compute_mfcc: bool = True
    compute_chroma: bool = True
    compute_tempogram: bool = True

    def __post_init__(self):
        self._time = TimeFeatures(self.sig) if self.compute_time else None
        self._freq = FrequencyFeatures(self.sig) if self.compute_frequency else None
        self._mfcc = MFCCFeatures(self.sig) if self.compute_mfcc else None
        self._chroma = ChromagramFeatures(self.sig) if self.compute_chroma else None
        self._temp = TempogramFeatures(self.sig) if self.compute_tempogram else None

    @staticmethod
    def from_audio(y: np.ndarray, sr: int, n_fft: int = 2048, hop_length: int = 512, **kwargs):
        sig = AudioSignal(signal=y, sr=sr, N=n_fft, H=hop_length)
        return FeatureExtractor(sig, **kwargs)

    def extract(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}

        if self._time is not None:
            out.update(self._extract_time())
        if self._freq is not None:
            out.update(self._extract_frequency())
        if self._mfcc is not None:
            out.update(self._extract_mfcc())
        if self._chroma is not None:
            out.update(self._extract_chroma())
        if self._temp is not None:
            out.update(self._extract_tempogram())

        out = self._add_unified_aliases(out)
        return out

    @staticmethod
    def _flatten(d: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in d.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    out[f"{k}.{kk}"] = vv
            else:
                out[k] = v
        return out
    
    @staticmethod
    def _first_existing(d: Dict[str, Any], keys: list[str]) -> Any:
        for k in keys:
            if k in d:
                return d[k]
            
        return None

    @staticmethod
    def _add_unified_aliases(d: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(d)
        aliases = {
            "loudness": ["time.loudness", "frequency.loudness", "mfcc.loudness", "tempogram.loudness"],
            "energy": ["time.energy", "frequency.energy", "mfcc.energy", "chroma.energy"],
            "speechiness": ["mfcc.speechiness", "time.speechiness", "frequency.speechiness", "chroma.speechiness"],
            "acousticness": ["frequency.acousticness", "mfcc.acousticness", "time.acousticness", "chroma.acousticness"],
            "danceability": ["time.danceability", "frequency.danceability", "tempogram.danceability", "chroma.danceability"],
            "valence": ["chroma.valence", "frequency.valence", "mfcc.valence", "tempogram.valence"], 
            "tempo": ["tempogram.tempo", "time.tempo", "frequency.tempo", "chroma.tempo"],
            "liveness": ["frequency.liveness", "time.liveness", "mfcc.liveness", "tempogram.liveness"],
            "instrumentalness": ["frequency.instrumentalness", "mfcc.instrumentalness", "time.instrumentalness", "chroma.instrumentalness"],
            "key": ["chroma.key", "frequency.key"],
            "mode": ["chroma.mode", "frequency.mode", "tempogram.mode"],
            "time_signature": ["time.time_signature", "tempogram.time_signature", "frequency.time_signature", "chroma.time_signature"]
        }

        for target, sources in aliases.items():
            for src in sources:
                if src not in out:
                    continue
                val = out[src]
                if val is None:
                    continue
                if isinstance(val, float) and np.isnan(val):
                    continue
                out[target] = val
                out[f"{target}.__source__"] = src
                break

        return out
    
    def _extract_time(self) -> Dict[str, Any]:
        t = self._time

        return {
            "time.loudness": t._spotify_loudness(),
            "time.energy": t._spotify_energy(),
            "time.speechiness": t._spotify_speechiness(),
            "time.acousticness": t._spotify_acousticness(),
            "time.danceability": t._spotify_danceability(),
            "time.tempo": t._spotify_tempo(),
            "time.liveness": t._spotify_liveness(),
            "time.instrumentalness": t._spotify_instrumentalness(),
            "time.time_signature": t._spotify_time_signature()
        }
    
    def _extract_frequency(self) -> Dict[str, Any]:
        f = self._freq

        return {
            "frequency.loudness": f._loudness_freq_active_db(),
            "frequency.energy": f._energy_freq(),
            "frequency.speechiness": f._speechiness_freq(),
            "frequency.acousticness": f._acousticness_freq(),
            "frequency.danceability": f._danceability_freq(),
            "frequency.valence": f._valence_freq(),
            "frequency.tempo": f._tempo_freq(),
            "frequency.liveness": f._liveness_freq(),
            "frequency.instrumentalness": f._instrumentalness_freq(),
            "frequency.key": f._key_freq(),
            "frequency.mode": f._mode_freq(),
            "frequency.time_signature": f._time_signature_freq()
        }
    
    def _extract_chroma(self) -> Dict[str, Any]:
        c = self._chroma
        key_est = c._key_estimation()

        return {
            "chroma.energy": c._energy_chroma(),
            "chroma.speechiness": c._speechiness_chroma(),
            "chroma.acousticness": c._acousticness_chroma(),
            "chroma.danceability": c._danceability_chroma(),
            "chroma.valence": c._valence_chroma(),
            "chroma.tempo": c._tempo_chroma(),
            "chroma.instrumentalness": c._instrumentalness_chroma(),
            "chroma.key": key_est['key_idx'],
            "chroma.mode": key_est['mode'],
            "chroma.time_signature": c._time_signature_chroma()
        }
    
    def _extract_tempogram(self) -> Dict[str, Any]:
        t = self._temp

        return {
            "tempogram.loudness": float(np.mean(t._loudness_tempogram_per_beat())),
            "tempogram.danceability": t._danceability_tempogram(),
            "tempogram.valence": t._valence_tempogram(),
            "tempogram.tempo": float(t._global_bpm()["bpm"]),
            "tempogram.liveness": t._liveness_tempogram(),
            "tempogram.mode": t._mode_tempogram()["mode"],
            "tempogram.time_signature": t._time_signature_tempogram()["time_signature"]
        }
    
    def _extract_mfcc(self) -> Dict[str, Any]:
        m = self._mfcc

        return {
            "mfcc.loudness": m._loudness_mfcc(),
            "mfcc.energy": m._energy_mfcc(),
            "mfcc.speechiness": m._speechiness_mfcc(),
            "mfcc.acousticness": m._acousticness_mfcc(),
            "mfcc.valence": m._valence_mfcc(),
            "mfcc.liveness": m._liveness_mfcc(),
            "mfcc.instrumentalness": m._instrumentalness_mfcc()
        }