from hydra.utils import to_absolute_path
from pathlib import Path
import pandas as pd

def load_data(cfg):
    file_path = to_absolute_path(cfg.data.path)

    if Path(file_path).exists():
        print(f"---Đang tải dữ liệu từ {file_path}---")
        df = pd.read_csv(file_path)
        return df
    else:
        raise FileNotFoundError(f"Không thấy file: {file_path}")
    