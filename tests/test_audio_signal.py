import logging
import numpy as np
import pytest
import math

from audio_features.audio_signal import AudioSignal, DEFAULT_SR, CLIP_THRESHOLD

def test_signal_validity_and_properties():
    # 1s sine wave at 440 Hz
    sr = DEFAULT_SR
    t = np.linspace(0, 1.0, sr, endpoint=False)
    y = 0.1*np.sin(2*np.pi*440.0*t)

    sig = AudioSignal(signal=y)

    assert sig.is_valid is True
    assert sig.invalid is False
    assert sig.sr == DEFAULT_SR
    assert sig.y.dtype == np.float32
    assert sig.y.ndim == 1

def test_silent_signal():
    y = np.zeros(1024)
    sig = AudioSignal(signal=y)

    assert sig.is_valid is False
    assert sig.invalid is True

def test_clipped_signal_logs_and_invalid(caplog):
    # create an obviously clipped signal exceeding CLIP_THRESHOLD
    y = np.ones(2048, dtype=np.float32) * (CLIP_THRESHOLD + 0.5)

    caplog.set_level(logging.WARNING)
    sig = AudioSignal(signal=y)

    assert sig.invalid is True
    # ensure the module emitted a warning about clipping
    assert any("Clipped" in r.message for r in caplog.records)

def test_multi_channel_conversion_channels_first():
    # shape (channels,, samples)
    ch0 = np.zeros(512)
    ch1 = np.ones(512)*0.4
    y = np.vstack([ch0, ch1])

    sig = AudioSignal(signal=y)

    assert sig.y.ndim == 1
    assert sig.y.shape[0] == 512
    assert np.allclose(sig.y, np.mean(y, axis=0), atol=1e-6)

def test_multi_channel_conversion_samples_first():
    # shape (samples, channels)
    ch0 = np.zeros(256)
    ch1 = np.ones(256)*0.8
    y = np.column_stack([ch0, ch1])

    sig = AudioSignal(signal=y)

    assert sig.y.ndim == 1
    assert sig.y.shape[0] == 256
    assert np.allclose(sig.y, np.mean(y, axis=1), atol=1e-6)