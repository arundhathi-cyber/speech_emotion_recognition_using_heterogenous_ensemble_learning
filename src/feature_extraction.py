import numpy as np
import librosa

from src.config import SAMPLE_RATE, DURATION


def extract_features(file_path: str) -> np.ndarray:
    """Extract a fixed-size feature vector from one audio file."""
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)

    # Pad or trim to a fixed length
    target_length = int(SAMPLE_RATE * DURATION)
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))
    else:
        y = y[:target_length]

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)

    features = np.hstack([
        np.mean(mfcc.T, axis=0),
        np.mean(chroma.T, axis=0),
        np.mean(mel.T, axis=0),
        np.mean(contrast.T, axis=0),
        np.mean(tonnetz.T, axis=0),
    ])

    return features