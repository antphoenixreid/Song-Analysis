"""
AudioSignal class for loading and basic audio signal processing.
"""

import os
import numpy as np
from typing import Dict
from .utils import safe_load


class AudioSignal:
    def __init__(self, audio_path: str = None, signal: np.ndarray = None, N: int = 2048, H: int = 512):
        self.audio_path = audio_path
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
            self.y = np.asarray(signal, dtype=float)
            self.sr = 22050
            invalid = False
        else:
            raise ValueError("Must provide either 'path' or 'signal'")

        # -----------------------------
        # GLOBAL VALIDITY CHECKS
        # -----------------------------
        self.invalid = False

        if invalid:
            self.invalid = True
            return

        # failed load / empty file
        if self.y is None or self.y.size == 0:
            self.invalid = True
            return

        # near-constant (all zeros or DC)
        if np.std(self.y) < 1e-6:
            self.invalid = True
            return

        # Detect clipping / malformed decode
        if np.max(np.abs(self.y)) > 1.5:
            print(f"[WARN] Clipped / malformed audio: {audio_path}")
            self.invalid = True
            return

        # compute once lazily — requires cache initialized
        self._cache: Dict[str, object] = {}