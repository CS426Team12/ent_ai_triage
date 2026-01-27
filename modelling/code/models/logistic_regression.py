import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report
import joblib

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from preprocessing import load_dataset, build_preprocessor, preprocess_features, encode_labels


def train_logistic_regression(
    data_path: str,
    output_model: str,
    test_size: float = 0.2,
    random_state: int = 42,
    cv_folds: int = 5,
    C_grid: list[float] | None = None,
    solvers: list[str] | None = None
) -> None:
    if C_grid is None:
        C_grid = [0.1, 1.0, 10.0]
    if solvers is None:
        solvers = ['liblinear', 'lbfgs']

    # Load dataset
    df = load_dataset(data_path)
    X = df.drop(columns=['urgency'])
    y = df['urgency']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Encode labels
    y_train_enc, label_mapping = encode_labels(y_train)
    y_test_enc = y_test.map(label_mapping)

    # Preprocess
    preprocessor = build_preprocessor(include_text=False)
    X_train_trans, X_test_trans = preprocess_features(preprocessor, X_train, X_test)
    X_train_dense = X_train_trans.toarray() if hasattr(X_train_trans, 'toarray') else X_train_trans
    X_test_dense = X_test_trans.toarray() if hasattr(X_test_trans, 'toarray') else X_test_trans

    # Hyper‑parameter search
    param_grid = {
        'C': C_grid,
        'solver': solvers,
        'max_iter': [200]
    }
    base_clf = LogisticRegression(multi_class='ovr', class_weight='balanced')
    grid = GridSearchCV(base_clf, param_grid, cv=cv_folds, scoring='f1_macro', n_jobs=-1)
    grid.fit(X_train_dense, y_train_enc)
    best_clf = grid.best_estimator_

    # Evaluation
    y_pred = best_clf.predict(X_test_dense)
    report = classification_report(y_test_enc, y_pred, target_names=list(label_mapping.keys()))
    print('Classification report for Logistic Regression:')
    print(report)
    print('Best parameters:', grid.best_params_)

    # Fit on full data
    X_full, _ = preprocess_features(preprocessor, X, X)
    X_full_dense = X_full.toarray() if hasattr(X_full, 'toarray') else X_full
    final_clf = LogisticRegression(**grid.best_params_, multi_class='ovr', class_weight='balanced')
    final_clf.fit(X_full_dense, encode_labels(y)[0])

    joblib.dump({
        'preprocessor': preprocessor,
        'model': final_clf,
        'label_mapping': label_mapping
    }, output_model)
    print(f'Model saved to {output_model}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train a logistic regression model on the ENT triage dataset.')
    parser.add_argument('--data', type=str, default='../data/ent_triage_dataset.csv', help='Path to the input CSV data.')
    parser.add_argument('--output_model', type=str, default='logistic_regression_model.pkl', help='Where to save the trained model.')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of the dataset to include in the test split.')
    parser.add_argument('--cv_folds', type=int, default=5, help='Number of cross‑validation folds for hyper‑parameter search.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train_logistic_regression(
        data_path=args.data,
        output_model=args.output_model,
        test_size=args.test_size,
        random_state=42,
        cv_folds=args.cv_folds
    )