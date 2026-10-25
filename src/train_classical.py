import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate, StratifiedKFold

from src.config import MODEL_DIR
from src.dataset import load_dataset


def get_models():
    return {
        "svm": make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced")
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
        "knn": make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=5)
        ),
    }


def main():
    X, y, _ = load_dataset("data/ravdess")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in get_models().items():
        scores = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=["accuracy", "f1_weighted"],
            n_jobs=-1,
        )

        print(name)
        print("Accuracy:", np.mean(scores["test_accuracy"]))
        print("F1:", np.mean(scores["test_f1_weighted"]))

        model.fit(X, y)
        joblib.dump(model, MODEL_DIR / f"{name}.joblib")


if __name__ == "__main__":
    main()