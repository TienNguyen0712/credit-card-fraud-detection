# src/models/evaluate.py

import joblib
from sklearn.metrics import roc_auc_score, classification_report

from src.data.preprocess import load_data, split_data, scale_data


def evaluate():
    df = load_data("data/creditcard.csv")

    X_train, X_test, y_train, y_test = split_data(df)
    X_train, X_test, scaler = scale_data(X_train, X_test)

    model = joblib.load("models/model.pkl")

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    print("ROC-AUC:", roc_auc_score(y_test, y_pred_proba))
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    evaluate()
