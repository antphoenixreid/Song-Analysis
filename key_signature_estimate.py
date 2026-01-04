import librosa
import numpy as np

# Load the audio file
filename = 'E:\Engineering\Signal Processing\Personal Projects\Song Analysis\Perish Song (Extended).wav'
y, sr = librosa.load(filename)

# Compute the chromagram of the audio signal
chroma = librosa.feature.chroma_stft(y=y, sr=sr)

# Compute the average chroma vector across time
mean_chroma = np.mean(chroma, axis=1)

# Define a function to map chroma vectors to key labels
def chroma_to_key(chroma_vector):
    # Define the 12 possible key labels
    key_labels = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    # Compute the correlation between the chroma vector and each key's template
    corr = []
    for i in range(12):
        key_template = np.roll(np.array([1, 0, 0.5, 0, 0.5, 1, 0, 0.5, 0, 0.5, 0.5, 0]), i)
        corr.append(np.correlate(chroma_vector, key_template))
    
    # Find the key label with the highest correlation
    max_corr_idx = np.argmax(corr)
    key_label = key_labels[max_corr_idx]
    
    return key_label

# Define a function to determine the mode (major or minor) of a key signature
def determine_mode(key_label, chroma_vector):
    # Define the templates for major and minor keys
    major_template = np.array([1, 0.5, 0.5, 0, 1, 0.5, 0.5, 1, 0.5, 0.5, 0.5, 0])
    minor_template = np.array([1, 0.5, 0, 1, 0.5, 0.5, 0.5, 1, 0.5, 0, 0.5, 0.5])
    
    # Compute the correlation between the chroma vector and the major and minor templates for the detected key
    major_corr = np.correlate(chroma_vector, major_template)
    minor_corr = np.correlate(chroma_vector, minor_template)
    
    # Determine whether the key is major or minor based on the correlation values
    if major_corr > minor_corr:
        mode = 'major'
    else:
        mode = 'minor'
    
    return mode

# Compute the key label and mode of the audio signal
key_label = chroma_to_key(mean_chroma)
mode = determine_mode(key_label, mean_chroma)

# Print the results
print('The key signature of', filename, 'is', key_label, mode)