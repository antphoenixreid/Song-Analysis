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


@dataclass
class FeatureExtractor:
    """Collect domain-specific DSP evidence for the central fusion layer.

    This class deliberately does not produce unified Spotify-style features.
    Each enabled DSP domain retains its own namespace so downstream fusion can
    apply feature-specific ownership, confidence, and weighting.
    """

    sig: AudioSignal
    compute_time: bool = True
    compute_frequency: bool = True
    compute_mfcc: bool = True
    compute_chroma: bool = True
    compute_tempogram: bool = True

    def __post_init__(self) -> None:
        self._time = TimeFeatures(self.sig) if self.compute_time else None
        self._freq = FrequencyFeatures(self.sig) if self.compute_frequency else None
        self._mfcc = MFCCFeatures(self.sig) if self.compute_mfcc else None
        self._chroma = ChromagramFeatures(self.sig) if self.compute_chroma else None
        self._temp = TempogramFeatures(self.sig) if self.compute_tempogram else None

    @staticmethod
    def from_audio(
        y: np.ndarray,
        sr: int,
        n_fft: int = 2048,
        hop_length: int = 512,
        **kwargs,
    ) -> "FeatureExtractor":
        sig = AudioSignal(signal=y, sr=sr, N=n_fft, H=hop_length)
        return FeatureExtractor(sig, **kwargs)

    def extract(self) -> Dict[str, Any]:
        """Return namespaced evidence from all enabled DSP domains."""
        out: Dict[str, Any] = {}

        if self._time is not None:
            out.update(self._extract_time())
        if self._freq is not None:
            out.update(self._extract_frequency())
        if self._chroma is not None:
            out.update(self._extract_chroma())
        if self._temp is not None:
            out.update(self._extract_tempogram())
        if self._mfcc is not None:
            out.update(self._extract_mfcc())

        return out

    @staticmethod
    def _finite_or_none(value: Any) -> Any:
        """Convert non-finite scalar values to None; preserve other values."""
        if value is None:
            return None
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return None
        return value

    def _extract_time(self) -> Dict[str, Any]:
        """Collect amplitude/time-domain evidence only."""
        evidence = self._time.time_domain_evidence()
        return {f"time.{key}": self._finite_or_none(value) for key, value in evidence.items()}

    def _extract_frequency(self) -> Dict[str, Any]:
        """Collect spectral/frequency-domain evidence only."""
        evidence = self._freq.frequency_domain_evidence()
        return {f"frequency.{key}": self._finite_or_none(value) for key, value in evidence.items()}

    def _extract_chroma(self) -> Dict[str, Any]:
        """Collect harmonic/key/mode evidence only."""
        evidence = self._chroma.chroma_domain_evidence()
        return {f"chroma.{key}": self._finite_or_none(value) for key, value in evidence.items()}

    def _extract_tempogram(self) -> Dict[str, Any]:
        """Collect rhythm/tempo/meter evidence only."""
        evidence = self._temp.tempogram_domain_evidence()
        return {f"tempogram.{key}": self._finite_or_none(value) for key, value in evidence.items()}

    def _extract_mfcc(self) -> Dict[str, Any]:
        """Collect MFCC/timbral evidence only."""
        evidence = self._mfcc.mfcc_domain_evidence()
        return {f"mfcc.{key}": self._finite_or_none(value) for key, value in evidence.items()}
