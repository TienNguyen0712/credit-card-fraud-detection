import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

def load_data(path):
    # Đọc file từ thư mục data
    return pd.read_csv(f"data/{path}.csv")

def outlier_removal(df, outlier_report):
    """
    Hàm này xử lý các giá trị ngoại lai của lớp đa số (0).
    Lớp giao dịch bình thường    
    """
    df_final = df.copy()
    for index, row in outlier_report.iterrows():
        feature = row['Feature']
        # Tính lại ngưỡng của feature đó
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        # Chỉ loại bỏ những dòng là Outlier VÀ đồng thời là Class 0
        indices_to_drop = df[(df['Class'] == 0) & ((df[feature] < lower) | (df[feature] > upper))].index
        df_final = df_final.drop(indices_to_drop, errors='ignore')

    return df_final


def preprocess_features(df):
    """
    Hàm này xử lý logic tạo đặc trưng (Feature Engineering) 
    trước khi chia dữ liệu.
    """
    # Tạo bản sao để tránh cảnh báo SettingWithCopy
    df = df.copy()
    
    # Tạo đặc trưng mới đổi thời gian qua giờ
    df['Hour'] = (df['Time'] // 3600) % 24
    # Thêm 1e-9 để tránh lỗi log(0)
    df['Amount_log'] = np.log1p(df['Amount']) 
    
    features = df.drop(columns=["Class", "Time", "Amount"])
    target = df['Class']
    return features, target

def split_scale_data(features, target):
    """
    Chia dữ liệu và thực hiện Scaling. 
    Lưu ý: Chỉ fit scaler trên tập Train để tránh rò rỉ dữ liệu (Data Leakage).
    """
    # 1. Chia dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )
    
    # 2. Khởi tạo Scaler
    scaler = RobustScaler()
    
    # 3. Scale cột Amount_log (giả sử nó nằm ở một vị trí cụ thể hoặc toàn bộ X)
    # Ở đây tôi ví dụ fit trên toàn bộ features nếu tất cả là số
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
