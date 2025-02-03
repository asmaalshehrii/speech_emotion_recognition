import librosa
import os
import pickle

# Paths
data_dir = "../data/organized_data"
output_file = "../data/mel_spectrograms.pkl"

# Parameters for mel-spectrogram extraction
SAMPLE_RATE = 16000
N_MELS = 128

# Initialize data containers
spectrograms = []
labels = []

# Loop through all emotion folders
for emotion_label in os.listdir(data_dir):
    emotion_path = os.path.join(data_dir, emotion_label)
    if os.path.isdir(emotion_path):
        print(f"Processing emotion: {emotion_label}")
        
        for i, file in enumerate(os.listdir(emotion_path)):
            if file.endswith(".wav"):
                file_path = os.path.join(emotion_path, file)
                print(f"  Processing file: {file} ({i+1}/{len(os.listdir(emotion_path))})")
                
                try:
                    # Load the audio file
                    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                    # Generate Mel-spectrogram
                    mel_spec = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=N_MELS, fmax=8000)
                    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

                    # Store spectrogram and label
                    spectrograms.append(mel_spec_db)
                    labels.append
