import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

def load_data(path):
    # Giả sử bạn đọc file csv
    return pd.read_csv(f"data/{path}.csv")

def split_scale_data(df):
    features = df.drop(columns=["Class", "Time", "Amount"])
    target = df['Class']
    
    scaler = RobustScaler()
    features['Amount_log'] = scaler.fit_transform(features['Amount_log'].values.reshape(-1, 1))
    
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42, stratify=target)
    
    return X_train, X_test, y_train, y_test, scaler

