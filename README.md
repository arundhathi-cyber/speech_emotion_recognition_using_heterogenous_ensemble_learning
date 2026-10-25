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
git clone [https://github.com/your-username/speech-emotion-recognition.git](https://github.com/your-username/speech-emotion-recognition.git)
cd speech-emotion-recognition
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

## Notes

The current evaluation uses random splits, so results may be optimistic because clips from the same speaker can appear in both training and validation data.