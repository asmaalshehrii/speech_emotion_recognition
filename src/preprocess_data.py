import librosa
import librosa.display
import numpy as np
import os
import pickle

# Paths
data_dir = "../data/organized_data"
output_file = "../data/mel_spectrograms.pkl"

# Parameters for mel-spectrogram extraction
SAMPLE_RATE = 16000  # Standard sample rate
N_MELS = 128  # Number of mel bands

# Initialize data containers
spectrograms = []
labels = []

# Loop through all emotion folders
for emotion_label in os.listdir(data_dir):
    emotion_path = os.path.join(data_dir, emotion_label)
    if os.path.isdir(emotion_path):
        for file in os.listdir(emotion_path):
            if file.endswith(".wav"):
                file_path = os.path.join(emotion_path, file)

                # Load the audio file
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                # Generate Mel-spectrogram
                mel_spec = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=N_MELS, fmax=8000)
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

                # Store spectrogram and label
                spectrograms.append(mel_spec_db)
                labels.append(emotion_label)

# Convert to numpy arrays and save
data = {'spectrograms': np.array(spectrograms, dtype=object), 'labels': np.array(labels)}
with open(output_file, 'wb') as f:
    pickle.dump(data, f)

print("Preprocessing complete. Data saved to:", output_file)
