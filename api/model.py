"""Utility script to train a simple model for the starter API.

This uses a small synthetic dataset so the project can run end-to-end
without external data dependencies.
"""

from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_PATH = Path(__file__).parent / "model.pkl"


def _generate_data(random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Create a simple binary classification dataset."""

    features, labels = make_classification(
        n_samples=200,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        class_sep=1.5,
        random_state=random_state,
    )
    return features, labels


def train_and_save(model_path: Path = MODEL_PATH) -> Path:
    """Train a logistic regression model and persist it to disk."""

    features, labels = _generate_data()
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=200)),
        ]
    )
    pipeline.fit(x_train, y_train)

    model_path = Path(model_path)
    joblib.dump(pipeline, model_path)
    return model_path


if __name__ == "__main__":
    saved_path = train_and_save()
    print(f"Model saved to {saved_path}")
