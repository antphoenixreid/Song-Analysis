# Frequency Features Test Results

**Date:** 2026-04-28T23:57:31.439645

## Summary

- **Total Tests:** 43
- **Passed:** 39
- **Failed:** 0
- **Warnings:** 4

## Test Details

### ✅ magnitude_spectrum

**Status:** PASS

**Metrics:**

- success_rate: 1.000000
- avg_magnitude: 0.583579

**Details:**

- signals_tested: 10
- successful: 10

### ✅ power_spectrum

**Status:** PASS

**Metrics:**

- max_error: 0.000000
- avg_error: 0.000000

**Details:**

- test_cases: 4
- validation: power = magnitude²

### ✅ band_energy_basic

**Status:** PASS

**Metrics:**

- accuracy: 1.000000

**Details:**

- total_signals: 3
- correct_predictions: 3

### ✅ spectral_centroid_basic

**Status:** PASS

**Metrics:**

- accuracy: 1.000000
- avg_error: 16.076327

**Details:**

- total_signals: 5
- correct: 5
- results: [{'name': 'low_100hz', 'expected': 100, 'actual': np.float64(99.92284197895793), 'error': np.float64(0.0771580210420666), 'correct': np.True_}, {'name': 'mid_1000hz', 'expected': 1000, 'actual': np.float64(999.9359853461352), 'error': np.float64(0.06401465386477412), 'correct': np.True_}, {'name': 'high_5000hz', 'expected': 5000, 'actual': np.float64(4999.991675169914), 'error': np.float64(0.008324830086166912), 'correct': np.True_}, {'name': 'two_tones', 'expected': 1000, 'actual': np.float64(999.8821859991696), 'error': np.float64(0.11781400083043536), 'correct': np.True_}, {'name': 'weighted_tones', 'expected': 400, 'actual': np.float64(319.88567793140106), 'error': np.float64(80.11432206859894), 'correct': np.True_}]

### ✅ spectral_bandwidth

**Status:** PASS

**Metrics:**

- narrow_bandwidth: 8.263719
- wide_bandwidth: 3160.538157
- ratio: 382.459547

**Details:**

- signals_tested: 5
- validation: white noise BW >> pure sine BW

### ✅ spectral_rolloff

**Status:** PASS

**Metrics:**

- bass_rolloff_85: 215.332031
- treble_rolloff_85: 4995.950633

**Details:**

- signals_tested: 3
- percentiles_tested: [0.5, 0.85, 0.95]
- validation: rolloff increases with percentage

### ⚠️ spectral_slope

**Status:** WARN

**Metrics:**

- pink_slope: -0.006108
- white_slope: -0.000028
- high_slope: -0.119007

**Details:**

- signals_tested: 3
- slopes: {'pink_noise': {'linear': np.float64(-0.006107581904550365), 'log': np.float64(-0.0004747023514683449)}, 'white_noise': {'linear': np.float64(-2.751559216723904e-05), 'log': np.float64(-4.8864522874372934e-06)}, 'high_emphasis': {'linear': np.float64(-0.11900655109080659), 'log': np.float64(-0.001069484795598958)}}

### ✅ spectral_skewness

**Status:** PASS

**Metrics:**

- symmetric: 1.731223
- right_skewed: 3.749731
- left_skewed: -3.740094

**Details:**

- signals_tested: 3

### ✅ spectral_kurtosis

**Status:** PASS

**Metrics:**

- tone_kurtosis: 417.493669
- noise_kurtosis: -1.207930

**Details:**

- signals_tested: 3
- validation: single tone > white noise kurtosis

### ✅ comprehensive_spectral_shape

**Status:** PASS

**Metrics:**

- Pure 440 Hz_centroid: 439.930946
- Pure 440 Hz_bandwidth: 7.490374
- Pure 440 Hz_rolloff_85: 441.801926
- Pure 440 Hz_slope: -0.047400
- Pure 440 Hz_skewness: 1.675730
- Pure 440 Hz_kurtosis: 505.779267
- White Noise_centroid: 361.158554
- White Noise_bandwidth: 253.628140
- White Noise_rolloff_85: 656.515187
- White Noise_slope: -0.071789
- White Noise_skewness: 2.065179
- White Noise_kurtosis: 4.262889

**Details:**

- signals_tested: 6
- features_tested: 6

### ✅ fundamental_frequency

**Status:** PASS

**Metrics:**

- accuracy: 1.000000
- avg_error: 3.249023

**Details:**

- total_signals: 4
- correct: 4
- results: [{'name': 'A110', 'expected': 110, 'actual': np.float64(107.666015625), 'error': np.float64(2.333984375), 'correct': np.True_}, {'name': 'A220', 'expected': 220, 'actual': np.float64(215.33203125), 'error': np.float64(4.66796875), 'correct': np.True_}, {'name': 'A440', 'expected': 440, 'actual': np.float64(441.4306640625), 'error': np.float64(1.4306640625), 'correct': np.True_}, {'name': 'C523', 'expected': 523, 'actual': np.float64(527.5634765625), 'error': np.float64(4.5634765625), 'correct': np.True_}]

### ⚠️ harmonic_ratio

**Status:** WARN

**Metrics:**

- harmonic_hr: 0.229389
- noise_hr: 0.153891
- ratio: 1.490596

**Details:**

- signals_tested: 4
- validation: harmonic HR > noise HR
- note: Method has typo: _harmonnic_ratio (should be _harmonic_ratio)

### ✅ inharmonicity

**Status:** PASS

**Metrics:**

- perfect_inh: 0.009134
- inharmonic_inh: 0.053532

**Details:**

- signals_tested: 3
- validation: inharmonic > perfect harmonics

### ✅ spectral_peaks

**Status:** PASS

**Metrics:**

- avg_peaks_per_frame: 5.000000

**Details:**

- signals_tested: 2
- note: Visual validation required

### ✅ hnr

**Status:** PASS

**Metrics:**

- pure_hnr: 11.639005
- noise_hnr: -14.254700
- difference: 25.893705

**Details:**

- signals_tested: 3
- validation: pure harmonic HNR > noise HNR + 10 dB

### ✅ spectral_envelope

**Status:** PASS

**Details:**

- signals_tested: 3
- bands: 5

### ✅ comprehensive_harmonic_timbre

**Status:** PASS

**Details:**

- signals_tested: 4
- features_tested: 4

### ⚠️ spectral_flatness

**Status:** WARN

**Metrics:**

- sine_flatness: 0.000002
- noise_flatness: 0.560443
- ratio: 291843.974880

**Details:**

- signals_tested: 5
- in_range: 3
- ordered_correctly: True

### ✅ spectral_entropy

**Status:** PASS

**Metrics:**

- tone_entropy: 0.129349
- noise_entropy: 0.938589
- ratio: 7.256263

**Details:**

- signals_tested: 4
- validation: noise entropy > tone entropy

### ✅ spectral_flux

**Status:** PASS

**Metrics:**

- constant_flux: 0.004180
- noise_flux: 0.525157
- sweep_flux: 0.641201

**Details:**

- signals_tested: 4
- validation: dynamic signals > constant

### ⚠️ band_ratios

**Status:** WARN

**Metrics:**

- accuracy: 0.750000

**Details:**

- signals_tested: 4
- correct_predictions: 3

### ✅ low_high_band_ratio

**Status:** PASS

**Metrics:**

- bass_ratio: 21615125643070.414062
- treble_ratio: 0.000011
- difference: 1892188132138288128.000000

**Details:**

- signals_tested: 3
- ordered_correctly: True

### ✅ comprehensive_noise_dynamics

**Status:** PASS

**Details:**

- signals_tested: 6
- features_tested: 4

### ✅ pulse_clarity

**Status:** PASS

**Metrics:**

- regular_clarity: 0.713701
- irregular_clarity: 0.281434
- sustained_clarity: 0.426642

**Details:**

- signals_tested: 4
- validation: regular > irregular clarity

### ✅ beat_periodicity

**Status:** PASS

**Metrics:**

- regular_periodicity: 0.887418
- random_periodicity: 0.855912

**Details:**

- signals_tested: 3
- validation: regular > random periodicity

### ✅ transient_counts

**Status:** PASS

**Metrics:**

- accuracy: 1.000000
- many_count: 23
- few_count: 6

**Details:**

- signals_tested: 4
- correct_counts: 4
- results: [{'name': 'many_transients', 'count': 23, 'rate': 7.619441105769231, 'expected_range': (15, 30), 'valid': True}, {'name': 'few_transients', 'count': 6, 'rate': 1.9876802884615383, 'expected_range': (3, 8), 'valid': True}, {'name': 'sustained', 'count': 1, 'rate': 0.3312800480769231, 'expected_range': (0, 3), 'valid': True}, {'name': 'crescendo', 'count': 1, 'rate': 0.3312800480769231, 'expected_range': (0, 4), 'valid': True}]

### ✅ percussive_spectral_slope

**Status:** PASS

**Metrics:**

- bass_percussion: -0.000924
- treble_percussion: -0.000548
- broadband_percussion: 0.000022

**Details:**

- signals_tested: 3
- note: Slope values computed on transient frames

### ✅ comprehensive_rhythmic

**Status:** PASS

**Details:**

- signals_tested: 3
- features_tested: 5

### ✅ phase

**Status:** PASS

**Details:**

- signals_tested: 2
- note: Phase unwrapped correctly

### ✅ group_delay

**Status:** PASS

**Metrics:**

- pure_tone: {'mean': np.float64(-0.0028921044397998085), 'std': np.float64(0.0002696286942461355)}
- am_signal: {'mean': np.float64(-0.0027382075640506377), 'std': np.float64(0.0008222072745120317)}
- chirp: {'mean': np.float64(0.00035478712560363607), 'std': np.float64(0.0009098406352104539)}

**Details:**

- signals_tested: 3
- note: Group delay computed

### ✅ instantaneous_frequency

**Status:** PASS

**Metrics:**

- avg_error: 0.016667
- results: [{'name': '440hz_tone', 'expected': 440, 'actual': np.float64(440.01594889717876), 'error': np.float64(0.01594889717875958), 'bin_used': np.int64(41), 'bin_freq': np.float64(441.4306640625)}, {'name': '1000hz_tone', 'expected': 1000, 'actual': np.float64(1000.0149154852635), 'error': np.float64(0.014915485263486516), 'bin_used': np.int64(93), 'bin_freq': np.float64(1001.2939453125)}, {'name': 'between_bins', 'expected': 450.5, 'actual': np.float64(450.5191365338857), 'error': np.float64(0.019136533885728113), 'bin_used': np.int64(42), 'bin_freq': np.float64(452.197265625)}]

**Details:**

- signals_tested: 4
- tones_tested: 3
- implementation: Phase vocoder bin offset method

### ✅ phase_congruency

**Status:** PASS

**Metrics:**

- tone_congruency: 0.296100
- noise_congruency: 0.008438
- ratio: 35.091816

**Details:**

- signals_tested: 4
- validation: tonal > noise congruency

### ✅ phase_coherence_time

**Status:** PASS

**Metrics:**

- stable_tone: {'mean': np.float64(0.7443526629987884), 'max': np.float64(0.9999072821616651), 'std': np.float64(0.13690872594841585)}
- frequency_modulated: {'mean': np.float64(0.18122585578101807), 'max': np.float64(0.6938214805455724), 'std': np.float64(0.15514893665803378)}
- white_noise: {'mean': np.float64(0.561952445229303), 'max': np.float64(0.7576733893955411), 'std': np.float64(0.0546387926438525)}

**Details:**

- signals_tested: 3
- note: Phase coherence over time

### ✅ phase_coherence_channels

**Status:** PASS

**Metrics:**

- identical: 1.000000
- phase_shifted: 0.831222
- different: 0.000124

**Details:**

- pairs_tested: 3
- ordered_correctly: True

### ✅ comprehensive_phase

**Status:** PASS

**Details:**

- signals_tested: 4
- features_tested: 5

### ✅ sub_band_energy

**Status:** PASS

**Details:**

- signals_tested: 4
- n_bands: 8
- results: [{'name': 'low_freq', 'dominant_band': np.int64(0), 'band_freq': '0-1378 Hz', 'energy_distribution': array([1.32312329e+05, 8.60032365e-02, 9.30149880e-03, 2.63398902e-03,
       1.16000194e-03, 6.74926777e-04, 4.83303981e-04, 4.14885533e-04])}, {'name': 'mid_freq', 'dominant_band': np.int64(0), 'band_freq': '0-1378 Hz', 'energy_distribution': array([9.72114181e+04, 2.42980958e+04, 1.08008412e+04, 4.38869905e-01,
       1.46461290e-01, 7.70232296e-02, 5.27033440e-02, 4.44125977e-02])}, {'name': 'high_freq', 'dominant_band': np.int64(3), 'band_freq': '4134-5512 Hz', 'energy_distribution': array([1.64982534e+00, 2.30582409e+00, 5.65011814e+00, 9.72049193e+04,
       9.71975547e+04, 9.72041398e+04, 4.98930361e+00, 3.01489671e+00])}, {'name': 'broadband', 'dominant_band': np.int64(0), 'band_freq': '0-1378 Hz', 'energy_distribution': array([1003.31351413,  961.7553166 ,  989.69559083,  977.18084784,
        971.14889704,  975.92862463,  954.8423337 ,  997.66396486])}]

### ✅ sub_band_energy_ratios

**Status:** PASS

**Details:**

- signals_tested: 3
- n_bands: 8
- validation: sum of ratios ≈ 1.0

### ✅ sub_band_entropy

**Status:** PASS

**Metrics:**

- single_tone_entropy: 0.000169
- noise_entropy: 0.996804

**Details:**

- signals_tested: 4
- n_bands: 8
- ordered_correctly: True

### ✅ sub_band_centroid

**Status:** PASS

**Metrics:**

- bass: 689.064721
- mid: 1167.167217
- treble: 6201.509929

**Details:**

- signals_tested: 3
- n_bands: 8
- ordered_correctly: True

### ✅ sub_band_flatness

**Status:** PASS

**Metrics:**

- single_band: 0.000017
- few_bands: 0.000745
- white_noise: 0.993133

**Details:**

- signals_tested: 3
- n_bands: 8
- ordered_correctly: True

### ✅ sub_band_ratio

**Status:** PASS

**Details:**

- n_bands: 8
- ratios_tested: 3

### ✅ sub_band_low_high_ratio

**Status:** PASS

**Metrics:**

- bass_heavy: 949082591104841.625000
- balanced: 2.999990
- treble_heavy: 0.333424

**Details:**

- signals_tested: 3
- n_bands: 8
- split_band: 4
- ordered_correctly: True

### ✅ comprehensive_sub_band

**Status:** PASS

**Details:**

- signals_tested: 4
- n_bands: 8
- features_tested: 4

