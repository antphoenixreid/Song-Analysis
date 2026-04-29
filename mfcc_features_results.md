# Frequency Features Test Results

**Date:** 2026-04-29T09:07:33.515347

## Summary

- **Total Tests:** 5
- **Passed:** 5
- **Failed:** 0
- **Warnings:** 0

## Test Details

### ✅ mfcc_mean

**Status:** PASS

**Metrics:**

- pure_440hz: {'c0_mean': -470.6946973721358}
- harmonic_220hz: {'c0_mean': -371.32037728562403}
- white_noise: {'c0_mean': -36.20317284447089}
- chirp: {'c0_mean': -467.9986917059543}

**Details:**

- signals_tested: 4
- n_mfcc: 13
- note: MFCC mean computed for each coefficient

### ✅ mfcc_variance

**Status:** PASS

**Metrics:**

- static_tone: {'total_variance': 876.237365331015, 'mean_variance': 67.40287425623193}
- frequency_modulated: {'total_variance': 1026.4601455433046, 'mean_variance': 78.95847273410035}
- chirp: {'total_variance': 9434.778241910151, 'mean_variance': 725.752172454627}
- white_noise: {'total_variance': 37.250870773033256, 'mean_variance': 2.865451597925635}

**Details:**

- signals_tested: 4
- ordered_correctly: True

### ✅ mfcc_skewness

**Status:** PASS

**Metrics:**

- pure_tone: {'mean_skewness': 2.901352705964535, 'max_abs_skewness': 6.214268147282682}
- harmonic: {'mean_skewness': 2.100501997801239, 'max_abs_skewness': 6.106751877820617}
- noise: {'mean_skewness': -0.3388232043832216, 'max_abs_skewness': 3.748971317598411}

**Details:**

- signals_tested: 3
- note: Skewness measures distribution asymmetry

### ✅ mfcc_kurtosis

**Status:** PASS

**Metrics:**

- pure_tone: {'mean_kurtosis': 30.118568176654303, 'max_kurtosis': 37.70815035547259}
- harmonic: {'mean_kurtosis': 30.243388235431862, 'max_kurtosis': 50.292017613169}
- white_noise: {'mean_kurtosis': 2.3221044298250506, 'max_kurtosis': 29.24909826637014}

**Details:**

- signals_tested: 3
- note: Kurtosis measures distribution peakedness

### ✅ comprehensive_mfcc_statistics

**Status:** PASS

**Details:**

- signals_tested: 4
- features_tested: 4

