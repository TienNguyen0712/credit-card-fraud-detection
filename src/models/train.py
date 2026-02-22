# src/models/train.py

import joblib
from sklearn.linear_model import LogisticRegression

from src.data.preprocess import load_data, split_scale_data


def train():
    df = load_data("link_data")

    X_train, X_test, y_train, y_test = split_data(df)

    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, "models/model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    print("Model trained and saved.")

if __name__ == "__main__":
    train()

