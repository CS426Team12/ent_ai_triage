from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from preprocessing import (
    load_dataset,
    build_preprocessor,
    preprocess_features,
    encode_labels,
)

def train_random_forest(
    data_path: str,
    output_model: str,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    cv_folds: int = 5,
    n_estimators_grid: Iterable[int] | None = None,
    max_depth_grid: Iterable[int | None] | None = None,
) -> None:
    
    if n_estimators_grid is None:
        n_estimators_grid = [50, 100, 200]
    if max_depth_grid is None:
        max_depth_grid = [None, 10, 20]

    # Load and split the data
    df: pd.DataFrame = load_dataset(data_path)
    X = df.drop(columns=["urgency"])
    y = df["urgency"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # encode the labels into integers and build the feature preprocessor
    y_train_enc, label_mapping = encode_labels(y_train)
    y_test_enc = y_test.map(label_mapping)
    preprocessor = build_preprocessor(include_text=False)

    X_train_trans, X_test_trans = preprocess_features(preprocessor, X_train, X_test)
    X_train_dense = X_train_trans.toarray() if hasattr(X_train_trans, "toarray") else X_train_trans
    X_test_dense = X_test_trans.toarray() if hasattr(X_test_trans, "toarray") else X_test_trans

    param_grid = {
        "n_estimators": list(n_estimators_grid),
        "max_depth": list(max_depth_grid),
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }
    base_clf = RandomForestClassifier(class_weight="balanced", random_state=random_state, n_jobs=-1)
    grid = GridSearchCV(
        base_clf,
        param_grid,
        cv=cv_folds,
        scoring="f1_macro",
        n_jobs=-1,
    )
    grid.fit(X_train_dense, y_train_enc)
    best_clf: RandomForestClassifier = grid.best_estimator_

    y_pred = best_clf.predict(X_test_dense)
    report = classification_report(y_test_enc, y_pred, target_names=list(label_mapping.keys()))
    print("Classification report for Random Forest:")
    print(report)
    print("Best parameters:", grid.best_params_)

    X_full, _ = preprocess_features(preprocessor, X, X)
    X_full_dense = X_full.toarray() if hasattr(X_full, "toarray") else X_full
    final_clf = RandomForestClassifier(**grid.best_params_, class_weight="balanced", random_state=random_state, n_jobs=-1)
    final_clf.fit(X_full_dense, encode_labels(y)[0])

    joblib.dump(
        {
            "preprocessor": preprocessor,
            "model": final_clf,
            "label_mapping": label_mapping,
        },
        output_model,
    )
    print(f"Model saved to {output_model}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Random Forest classifier on the ENT triage dataset."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="../data/ent_triage_dataset.csv",
        help="Path to the input CSV data.",
    )
    parser.add_argument(
        "--output_model",
        type=str,
        default="random_forest_model.pkl",
        help="Where to save the trained model.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Proportion of the dataset to include in the test split.",
    )
    parser.add_argument(
        "--cv_folds",
        type=int,
        default=5,
        help="Number of cross‑validation folds for hyper‑parameter search.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_random_forest(
        data_path=args.data,
        output_model=args.output_model,
        test_size=args.test_size,
        random_state=42,
        cv_folds=args.cv_folds,
    )