# Speech Emotion Recognition using Heterogeneous Ensemble Learning

This project performs speech emotion recognition on the RAVDESS dataset using audio features such as MFCCs, chroma, Mel-spectrograms, spectral contrast, and tonnetz. It compares classical machine-learning models and a deep neural network for eight-class emotion classification.

## Emotions Recognized

- Neutral
- Calm
- Happy
- Sad
- Angry
- Fearful
- Disgust
- Surprised

## Dataset

This project uses the [RAVDESS Emotional Speech Audio dataset](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio).

Download the dataset from Kaggle and extract the `.wav` files into:

```text
data/ravdess/
```

The expected structure is:

```text
data/
└── ravdess/
    ├── Actor_01/
    ├── Actor_02/
    ├── Actor_03/
    └── ...
```

The dataset is not included in this repository because of its size.

## Features Extracted

Each audio file is converted into a fixed-size feature vector using:

- MFCCs
- Chroma features
- Mel spectrogram features
- Spectral contrast
- Tonnetz features

## Models Used

- Support Vector Machine (SVM)
- Random Forest
- K-Nearest Neighbors (KNN)
- Deep Neural Network (DNN)

## Project Structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
└── src/
    ├── config.py
    ├── feature_extraction.py
    ├── dataset.py
    ├── train_classical.py
    ├── train_dnn.py
    └── evaluate.py
```

## Installation

```bash
git clone [https://github.com/arundhathi-cyber/speech_emotion_recognition_using_heterogenous_ensemble_learning.git](https://github.com/arundhathi-cyber/speech_emotion_recognition_using_heterogenous_ensemble_learning.git)
cd speech_emotion_recognition_using_heterogenous_ensemble_learning
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Training

After placing the RAVDESS dataset in `data/ravdess/`, train the classical models:

```bash
python -m src.train_classical
```

Train the deep neural network:

```bash
python -m src.train_dnn
```

Evaluate the trained model and generate plots:

```bash
python -m src.evaluate
```

## Results

| Model | Accuracy |
|---|---:|
| SVM | 95.07% |
| Random Forest | 91.94% |
| KNN | 73.85% |
| DNN | 95–96% |

These results were obtained from the original Kaggle notebook.
