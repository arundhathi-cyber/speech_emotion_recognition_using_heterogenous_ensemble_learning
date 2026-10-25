import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

from src.config import FIGURE_DIR
from src.dataset import load_dataset


def plot_confusion_matrix(y_true, y_pred, labels, filename):
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Normalized Confusion Matrix")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename, dpi=300)
    plt.close()


def main():
    X, y, _ = load_dataset("data/ravdess")

    model = joblib.load("models/svm.joblib")

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred))
    plot_confusion_matrix(y_test, y_pred, sorted(np.unique(y)), "confusion_matrix.png")


if __name__ == "__main__":
    main()