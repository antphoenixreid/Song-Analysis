# Frequency Features Test Results

**Date:** 2026-04-28T23:58:02.426364

## Summary

- **Total Tests:** 25
- **Passed:** 22
- **Failed:** 0
- **Warnings:** 3

## Test Details

### ✅ chromagram

**Status:** PASS

**Details:**

- signals_tested: 3
- note: Visual validation of pitch class detection

### ✅ pitch_class_profile

**Status:** PASS

**Details:**

- signals_tested: 3
- validation: sum of profile ≈ 1.0

### ⚠️ pitch_class_deviation

**Status:** WARN

**Metrics:**

- pure_note: 0.386244
- perfect_fifth: 12.378427
- chromatic: 12.056431

**Details:**

- signals_tested: 3
- ordered_correctly: False

### ✅ chroma_centroid

**Status:** PASS

**Details:**

- note: Centroid tracks dominant pitch class over time

### ✅ chroma_spread

**Status:** PASS

**Metrics:**

- single_note: 0.341982
- major_triad: 3.068783
- chromatic: 3.439702

**Details:**

- signals_tested: 3
- ordered_correctly: True

### ✅ chroma_skewness_kurtosis

**Status:** PASS

**Details:**

- signals_tested: 3
- results: {'single_note': {'skewness': np.float64(11.284900379488036), 'kurtosis': np.float64(201.22995749510702)}, 'major_chord': {'skewness': np.float64(1.3539065546107054), 'kurtosis': np.float64(-1.0553451597145906)}, 'chromatic': {'skewness': np.float64(1.3034652975726948), 'kurtosis': np.float64(-1.186358542466954)}}

### ✅ comprehensive_chroma

**Status:** PASS

**Details:**

- signals_tested: 4
- features_tested: 6

### ✅ key_estimation

**Status:** PASS

**Metrics:**

- accuracy: 1.000000

**Details:**

- signals_tested: 5
- correct_detections: 5
- results: [{'name': 'C_major', 'expected': 'C maj', 'detected': 'C maj', 'correct': True, 'score': 0.7388869917330414}, {'name': 'A_minor', 'expected': 'A min', 'detected': 'A min', 'correct': True, 'score': 0.7176828368594014}, {'name': 'G_major', 'expected': 'G maj', 'detected': 'G maj', 'correct': True, 'score': 0.7281392615029051}, {'name': 'E_minor', 'expected': 'E min', 'detected': 'E min', 'correct': True, 'score': 0.7225594506460482}, {'name': 'F_major', 'expected': 'F maj', 'detected': 'F maj', 'correct': True, 'score': 0.7292164898920289}]

### ✅ mode_classification

**Status:** PASS

**Metrics:**

- accuracy: 1.000000

**Details:**

- signals_tested: 6
- correct_detections: 6
- results: [{'name': 'C_major', 'expected': 'maj', 'detected': 'maj', 'correct': True, 'delta_score': np.float64(0.12571197262271228)}, {'name': 'A_minor', 'expected': 'min', 'detected': 'min', 'correct': True, 'delta_score': np.float64(-0.08090066341257895)}, {'name': 'G_major', 'expected': 'maj', 'detected': 'maj', 'correct': True, 'delta_score': np.float64(0.12658911284913643)}, {'name': 'E_minor', 'expected': 'min', 'detected': 'min', 'correct': True, 'delta_score': np.float64(-0.08002938195469134)}, {'name': 'D_major', 'expected': 'maj', 'detected': 'maj', 'correct': True, 'delta_score': np.float64(0.12554365524997646)}, {'name': 'D_minor', 'expected': 'min', 'detected': 'min', 'correct': True, 'delta_score': np.float64(-0.07804298836020274)}]

### ✅ tonal_clarity

**Status:** PASS

**Metrics:**

- strong_C_major: 0.040881
- weak_key: 0.015014
- chromatic: 0.000206
- dominant_7th: 0.039908

**Details:**

- signals_tested: 4
- ordered_correctly: True

### ⚠️ harmonic_entropy

**Status:** WARN

**Metrics:**

- clear_key: 0.998372
- moderate_ambiguity: 0.998980
- high_ambiguity: 0.998185
- chromatic: 0.999991

**Details:**

- signals_tested: 4
- ordered_correctly: False

### ✅ consonance_dissonance

**Status:** PASS

**Metrics:**

- major_consonance: 0.743088
- dim_consonance: 0.660474

**Details:**

- signals_tested: 5
- validation: major > diminished consonance
- results: [{'name': 'C_major', 'consonance': 0.7430878042359085, 'dissonance': 0.2569121957640915, 'type': 'consonant'}, {'name': 'A_minor', 'consonance': 0.7194549532522135, 'dissonance': 0.28054504674778646, 'type': 'consonant'}, {'name': 'G7', 'consonance': 0.7241804539062444, 'dissonance': 0.2758195460937556, 'type': 'moderate'}, {'name': 'Cdim', 'consonance': 0.6604744779007852, 'dissonance': 0.3395255220992148, 'type': 'dissonant'}, {'name': 'Caug', 'consonance': 0.6332506895934897, 'dissonance': 0.36674931040651026, 'type': 'dissonant'}]

### ✅ comprehensive_tonality

**Status:** PASS

**Details:**

- signals_tested: 6
- features_tested: 5

### ✅ chord_detection

**Status:** PASS

**Metrics:**

- accuracy: 1.000000

**Details:**

- signals_tested: 4
- correct_detections: 4
- results: [{'name': 'C_major', 'expected': 'Cmaj', 'detected': 'Cmaj', 'correct': True, 'detection_rate': 0.9770114942528736, 'avg_score': np.float64(0.9922154014921404)}, {'name': 'A_minor', 'expected': 'Amin', 'detected': 'Amin', 'correct': True, 'detection_rate': 1.0, 'avg_score': np.float64(0.9969726156926958)}, {'name': 'G7', 'expected': 'G7', 'detected': 'G7', 'correct': True, 'detection_rate': 1.0, 'avg_score': np.float64(0.9951069634455453)}, {'name': 'D_minor', 'expected': 'Dmin', 'detected': 'Dmin', 'correct': True, 'detection_rate': 1.0, 'avg_score': np.float64(0.9947740907048603)}]

### ✅ chord_progression

**Status:** PASS

**Details:**

- progressions_tested: 3
- note: Transition matrices computed

### ✅ harmonic_rhythm

**Status:** PASS

**Metrics:**

- fast_changes: {'change_rate': 4.9787752888934005, 'avg_duration': 0.19128819781881004, 'num_changes': 20}
- moderate_changes: {'change_rate': 1.3302364864643674, 'avg_duration': 0.6682186948853617, 'num_changes': 8}
- slow_changes: {'change_rate': 0.7497053312316658, 'avg_duration': 1.2004716553287982, 'num_changes': 9}
- static: {'change_rate': 0.33255912161609186, 'avg_duration': 2.004656084656084, 'num_changes': 2}

**Details:**

- signals_tested: 4
- ordered_correctly: True

### ✅ root_motion

**Status:** PASS

**Metrics:**

- stepwise: 0.108696
- fifths: 0.086957
- tritone: 0.069767

**Details:**

- progressions_tested: 3
- note: Root interval histograms computed

### ✅ tonal_stability

**Status:** PASS

**Metrics:**

- stable_static: 0.882760
- stable_progression: 0.730307
- moderate_variation: 0.713025
- unstable: 0.595242

**Details:**

- signals_tested: 4
- ordered_correctly: True

### ✅ comprehensive_harmonic

**Status:** PASS

**Details:**

- progressions_tested: 4
- features_tested: 5

### ✅ chroma_autocorrelation

**Status:** PASS

**Details:**

- signals_tested: 3
- note: Autocorrelation computed for temporal patterns

### ⚠️ chroma_variability

**Status:** WARN

**Metrics:**

- static: 0.010023
- slow_changes: 0.089007
- fast_changes: 0.085569

**Details:**

- signals_tested: 3
- ordered_correctly: False

### ✅ chroma_smoothness

**Status:** PASS

**Metrics:**

- static: 0.995892
- slow_progression: 0.991286
- fast_progression: 0.965686

**Details:**

- signals_tested: 3
- ordered_correctly: True

### ✅ dominant_pitch_track

**Status:** PASS

**Details:**

- note: Dominant pitch tracked over time

### ✅ tuning_deviation

**Status:** PASS

**Metrics:**

- avg_error: 4.407380

**Details:**

- signals_tested: 4
- results: [{'name': 'A440_perfect', 'expected_cents': 0.0, 'detected_cents': 0.5242459140539779, 'error': 0.5242459140539779}, {'name': 'A440_sharp', 'expected_cents': 8.0, 'detected_cents': 0.5242459140539779, 'error': 7.475754085946022}, {'name': 'A440_flat', 'expected_cents': -8.0, 'detected_cents': 0.5242459140539779, 'error': 8.524245914053978}, {'name': 'chord_perfect', 'expected_cents': 0.0, 'detected_cents': -1.1052759651548172, 'error': 1.1052759651548172}]

### ✅ comprehensive_temporal

**Status:** PASS

**Details:**

- signals_tested: 3
- features_tested: 3

