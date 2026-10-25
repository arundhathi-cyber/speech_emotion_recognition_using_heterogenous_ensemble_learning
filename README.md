# Speech Emotion Recognition

This project classifies speech recordings into eight emotions using the RAVDESS dataset.

## Emotions

- Neutral
- Calm
- Happy
- Sad
- Angry
- Fearful
- Disgust
- Surprised

## Features

The model uses audio features including:

- MFCCs
- Chroma
- Mel spectrogram
- Spectral contrast
- Tonnetz

## Models

- Support Vector Machine
- Random Forest
- K-Nearest Neighbors
- Deep Neural Network

## Setup

```bash
git clone https://github.com/arundhathi-cyber/speech_emotion_recognition_using_heterogenous_ensemble_learning.git
cd speech_emotion_recognition_using_heterogenous_ensemble_learning
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Download Data

Download the RAVDESS speech dataset and place the `.wav` files in:

```text
data/ravdess/
```

## Train Classical Models

```bash
python -m src.train_classical
```

## Train DNN

```bash
python -m src.train_dnn
```

## Evaluate

```bash
python -m src.evaluate
```

## Results

| Model | Accuracy |
|---|---:|
| SVM | 95.07% |
| Random Forest | 91.94% |
| KNN | 73.85% |
| DNN | ~95–96% |
