"""
AudioSignal class for loading and basic audio signal processing.
"""

import os
import logging
import numpy as np
from typing import Dict, Optional
from .utils import safe_load

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

        if audio_path is not None:
            # load
            if not isinstance(audio_path, str):
                raise ValueError("audio_path must be a string path")
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            self.y, self.sr, invalid = safe_load(audio_path)
        elif signal is not None:
            # accept provided signal; prefer float32 for memory/compat
            self.y = np.asarray(signal, dtype=np.float32)
            self.sr = DEFAULT_SR
            invalid = False
        else:
            raise ValueError("Must provide either 'audio_path' or 'signal'")

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
        self._invalid: bool = False

        if invalid:
            self._invalid = True
            return

        # failed load / empty file
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

        # compute once lazily — requires cache initialized
        self._cache: Dict[str, object] = {}

    # Backwards-compatible property
    @property
    def invalid(self) -> bool:
        """Backward-compatible read-only view of invalid state."""
        return self._invalid

    @property
    def is_valid(self) -> bool:
        """Readable convenience: True if signal passed validation."""
        return not self._invalid