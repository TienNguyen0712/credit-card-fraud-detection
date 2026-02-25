import numpy as np
import pandas as pd

def add_features(df, cfg):
    df = df.copy()

    if cfg.features.log_amount.enabled:
        col = cfg.features.log_amount.input_col
        out = cfg.features.log_amount.output_col
        df[out] = np.log1p(df[col])
        df.drop(columns=[col], inplace=True)

    if cfg.features.extract_hour.enabled:
        col = cfg.features.extract_hour.input_col
        out = cfg.features.extract_hour.output_col
        df[out] = (df[col] // 3600) % 24
        df.drop(columns=[col], inplace=True)

    return df