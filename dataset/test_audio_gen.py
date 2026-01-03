import numpy as np
import soundfile as sf

def write_wav(path, y, sr=44100):
    sf.write(path, y, sr, subtype="PCM_16")


# ---------------------------------------------------
# 1) CONSTANT SINE
# ---------------------------------------------------
def sine_tone(freq=440.0, duration=5.0, sr=44100, amplitude=0.5):
    t = np.linspace(0, duration, int(sr*duration), endpoint=False)
    y = amplitude * np.sin(2*np.pi*freq*t)
    return y, sr


# ---------------------------------------------------
# 2) AMPLITUDE-MODULATED (TREMOLO) SINE
# ---------------------------------------------------
def am_sine(
    carrier_freq=440.0,
    mod_freq=4.0,
    duration=5.0,
    sr=44100,
    depth=0.8,
):
    t = np.linspace(0, duration, int(sr*duration), endpoint=False)
    mod = (1.0 - depth) + depth * np.sin(2*np.pi*mod_freq*t)
    y = 0.5 * mod * np.sin(2*np.pi*carrier_freq*t)
    return y, sr


# ---------------------------------------------------
# 3) CLICK TRACK (SIMULATED DRUM)
# ---------------------------------------------------
def click_track(bpm=120, duration=10.0, sr=44100, click_amp=0.9):
    y = np.zeros(int(sr*duration))
    beats_per_sec = bpm / 60.0
    spacing = int(sr / beats_per_sec)

    for i in range(0, len(y), spacing):
        y[i:i+5] = click_amp  # short impulse

    return y, sr


# ---------------------------------------------------
# 4) WHITE NOISE
# ---------------------------------------------------
def white_noise(duration=5.0, sr=44100, amplitude=0.4):
    y = amplitude * np.random.randn(int(sr*duration))
    return y, sr


# ---------------------------------------------------
# 5) SILENCE
# ---------------------------------------------------
def silence(duration=5.0, sr=44100):
    y = np.zeros(int(sr*duration))
    return y, sr


# ---------------------------------------------------
# WRAPPERS TO SAVE PREMADE TEST FILES
# ---------------------------------------------------
def make_all(out_dir="tests_audio", sr=44100):
    import os
    os.makedirs(out_dir, exist_ok=True)

    signals = {
        "sine.wav": sine_tone(),
        "am_sine.wav": am_sine(),
        "click_120.wav": click_track(120),
        "white_noise.wav": white_noise(),
        "silence.wav": silence(),
    }

    for name, (y, sr) in signals.items():
        write_wav(f"{out_dir}/{name}", y, sr)
        print(f"Saved {out_dir}/{name}")


if __name__ == "__main__":
    make_all()