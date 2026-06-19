"""
Audio Feature Extraction Package

This package provides classes for extracting various audio features from audio signals.
"""

from .utils import *
from .audio_signal import AudioSignal
from .time_features import TimeFeatures
from .frequency_features import FrequencyFeatures
from .chromagram_features import ChromagramFeatures
from .tempogram_features import TempogramFeatures
from .mfcc_features import MFCCFeatures

__all__ = [
    'AudioSignal',
    'TimeFeatures',
    'FrequencyFeatures',
    'ChromagramFeatures',
    'TempogramFeatures',
    'MFCCFeatures',
    # Utility functions
    'EPS',
    'safe_median',
    'safe_clip01',
    'ensure_1d',
    'robust_normalize',
    'min_length',
    'fft_autocorr',
    'squash',
    'safe_load',
    'confidence_weighted'
]