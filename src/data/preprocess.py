import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

def load_data(path):
    # Giả sử bạn đọc file csv
    return pd.read_csv(f"data/{path}.csv")

def split_scale_data(df):
    features = df.drop('target', axis=1) # Thay 'target' bằng tên cột mục tiêu của bạn
    target = df['Class']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, statify=y)
    
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
