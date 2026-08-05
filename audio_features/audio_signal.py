"""
AudioSignal class for loading and basic audio signal processing.
"""

import os
import logging
import numpy as np
from typing import Dict, Optional
from audio_features.utils import safe_load

logger = logging.getLogger(__name__)

# Configuration/Constants
DEFAULT_SR: int = 22050
CLIP_THRESHOLD: float = 1.5
STD_EPS: float = 1e-6

class AudioSignal:
    def __init__(self, audio_path: Optional[str] = None, signal: Optional[np.ndarray] = None, sr: int = 22050, N: int = 2048, H: int = 512):
        self.audio_path = audio_path
        self.sr = sr
        self.N = N
        self.H = H

        # ALWAYS initialize the cache and invalid flag first
        self._cache: Dict[str, object] = {}
        self._invalid: bool = False

        if signal is not None:
            self.y = np.asarray(signal, dtype=np.float32)
            if sr is not None:
                self.sr = sr
            invalid = False
        elif audio_path is not None:
            if not isinstance(audio_path, str):
                raise ValueError("audio_path must be a string path")
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            self.y, self.sr, invalid = safe_load(audio_path)
        else:
            raise ValueError("Must provide either 'audio_path' or 'signal'")

        if invalid:
            self._invalid = True
            return

        # If multi-channel, convert to mono by averaging channels
        if getattr(self.y, "ndim", 0) > 1:
            try:
                # handle common layouts: (channels, samples) or (samples, channels)
                if self.y.shape[0] <= 8 and self.y.shape[1] > self.y.shape[0]:
                    # (channels, samples)
                    self.y = np.mean(self.y, axis=0)
                else:
                    # (samples, channels) or fallback
                    self.y = np.mean(self.y, axis=1)
            except Exception:
                # safe fallback
                self.y = np.mean(self.y, axis=-1)

        # ensure dtype
        self.y = np.asarray(self.y, dtype=np.float32)

        # -----------------------------
        # GLOBAL VALIDITY CHECKS
        # -----------------------------

        # Check for empty array safely
        if self.y is None or self.y.size == 0:
            self._invalid = True
            return

        # Detect clipping / malformed decode
        if np.max(np.abs(self.y)) > CLIP_THRESHOLD:
            logger.warning("Clipped / malformed audio: %s", self.audio_path)
            self._invalid = True
            return

        # near-constant (all zeros or DC)
        if np.std(self.y) < STD_EPS:
            self._invalid = True
            return

    # Backwards-compatible property
    @property
    def invalid(self) -> bool:
        """Backward-compatible read-only view of invalid state."""
        return self._invalid

    @property
    def is_valid(self) -> bool:
        """Readable convenience: True if signal passed validation."""
        return not self._invalid
    
    # Add these properties inside the AudioSignal class in audio_features/audio_signal.py
    @property
    def stft(self) -> np.ndarray:
        """Computes the STFT matrix once and caches it globally for all modules."""
        if "stft" not in self._cache:
            import librosa
            self._cache["stft"] = librosa.stft(
                self.y,
                n_fft=self.N,
                hop_length=self.H,
                win_length=self.N,
                window="hann",
                center=True
            )
        return self._cache["stft"]

    @property
    def stft_mag(self) -> np.ndarray:
        """Caches the magnitude spectrum so modules don't re-run np.abs() on massive arrays."""
        if "stft_mag" not in self._cache:
            self._cache["stft_mag"] = np.abs(self.stft)
        return self._cache["stft_mag"]

    @property
    def fft_freqs(self) -> np.ndarray:
        """Caches the FFT frequency bins."""
        if "fft_freqs" not in self._cache:
            import librosa
            self._cache["fft_freqs"] = librosa.fft_frequencies(sr=self.sr, n_fft=self.N)
        return self._cache["fft_freqs"]