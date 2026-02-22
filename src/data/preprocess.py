# src/data/preprocess.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(path: str):
    df = pd.read_csv(path)
    return df


def split_data(df, test_size=0.2, random_state=42):
    X = df.drop("Class", axis=1)
    y = df["Class"]

    return train_test_split(X, y,
                            test_size=test_size,
                            stratify=y,
                            random_state=random_state)


def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler
