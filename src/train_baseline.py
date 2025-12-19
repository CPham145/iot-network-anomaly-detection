import numpy as np

import json
from pathlib import Path

import pandas as pd
from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


DATA_PATH = Path("data/iot_sample_clean.csv")
MODEL_OUT = Path("models/baseline_logreg.joblib")
METRICS_OUT = Path("models/baseline_metrics.json")

LABEL_COL = "Label"


def main():
    df = pd.read_csv(DATA_PATH)
    df = df.replace([np.inf, -np.inf], np.nan)

    y = df[LABEL_COL]
    X = df.drop(columns=[LABEL_COL])

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ])

    model = Pipeline([
        ("preprocess", preprocessor),
        ("clf", LogisticRegression(max_iter=200, class_weight="balanced")),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")
    cm = confusion_matrix(y_test, preds).tolist()
    report = classification_report(y_test, preds, output_dict=True)

    MODEL_OUT.parent.mkdir(exist_ok=True)
    dump(model, MODEL_OUT)

    METRICS_OUT.write_text(json.dumps({
        "accuracy": acc,
        "f1_weighted": f1,
        "confusion_matrix": cm,
        "classification_report": report,
        "rows": len(df),
        "features": len(X.columns),
    }, indent=2))

    print(f"Accuracy: {acc:.4f}")
    print(f"F1 (weighted): {f1:.4f}")
    print("Saved model →", MODEL_OUT)
    print("Saved metrics →", METRICS_OUT)


if __name__ == "__main__":
    main()
