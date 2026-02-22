# src/data/preprocess.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


def load_data(path: str):
    df = pd.read_csv(path)
    return df


def split_scale_data_(df, test_size=0.2, random_state=42):
    # Tách dữ liệu 
    features = df.drop(columns=["Class", "Time", "Amount"])
    target = df["Class"]
    # Scale dữ liệu 
    scaler = RobustScaler()
    features['Amount_log'] = scaler.fit_transform(features['Amount_log'].values.reshape(-1, 1))

    return train_test_split(features, target,
                            test_size=test_size,
                            stratify=y,
                            random_state=random_state)


