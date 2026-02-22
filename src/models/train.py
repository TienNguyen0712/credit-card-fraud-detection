import joblib
import os
from sklearn.linear_model import LogisticRegression
from src.data.preprocess import load_data, split_scale_data

def train():
    # 1. Đảm bảo thư mục models tồn tại
    if not os.path.exists("models"):
        os.makedirs("models")

    # 2. Tải và xử lý dữ liệu
    print("Đang tải dữ liệu...")
    df = load_data("link_data") 
    X_train, X_test, y_train, y_test, scaler = split_scale_data(df)

    # 3. Huấn luyện mô hình
    print("Đang huấn luyện mô hình...")
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)

    # 4. Lưu kết quả
    joblib.dump(model, "models/model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    print("Hoàn tất! Model và Scaler đã được lưu vào thư mục models/.")

if __name__ == "__main__":
    train()
