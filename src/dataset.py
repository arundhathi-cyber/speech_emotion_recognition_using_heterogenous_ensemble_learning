from pathlib import Path

import numpy as np

from src.feature_extraction import extract_features

EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}


def load_dataset(data_dir: str):
    X, y, files = [], [], []

    for file_path in Path(data_dir).rglob("*.wav"):
        emotion_id = file_path.name.split("-")[2]

        if emotion_id not in EMOTIONS:
            continue

        features = extract_features(str(file_path))

        X.append(features)
        y.append(EMOTIONS[emotion_id])
        files.append(file_path.name)

    return np.array(X), np.array(y), files