import librosa
import numpy as np


def extract_audio_features(file_path):
    # Load audio file
    y, sr = librosa.load(file_path)

    # Extract features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
    chromagram = librosa.feature.chroma_stft(y=y, sr=sr)

    # Compile features into a dictionary
    features = {
        'mfcc': mfcc,
        'mel_spectrogram': mel_spectrogram,
        'zero_crossing_rate': zero_crossing_rate,
        'spectral_centroid': spectral_centroid,
        'spectral_rolloff': spectral_rolloff,
        'chromagram': chromagram
    }

    return features


# Example usage
# file_path = 'path/to/audio/file.wav'
# features = extract_audio_features(file_path)
# print(features)