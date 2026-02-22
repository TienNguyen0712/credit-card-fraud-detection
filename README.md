![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Machine Learning](https://img.shields.io/badge/Field-Machine%20Learning-purple)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-ML-yellow)

# 🛡 Credit Card Fraud Detection – End-to-End ML Pipeline

## 🎯 Project Overview (Tổng quan dự án)

- **Vấn đề:** Sự phát triển không ngừng của các ngành thương mại, sàn điện tử đòi hỏi con người phải minh bạch tinh vi hơn trong các hợp đồng giao dịch, việc xảy ra các gian lận là điều hiển nhiên khi không có bát kỳ một hệ thống nào có thể nhận diện chính xác. Bởi tính phức tạp của giao dịch cũng như quy mô triển khai khá lớn. 
- **Thách thức:** Lợi dụng những kẽ hở đó mà những người xấu sử dụng chúng để thực hiện những hành động phạm pháp như:
  - Sửa đổi giao dịch
  - Tạo nhiều giao dịch giả
  - Tạo giao dịch với tên người đặt hàng khác nhằm lợi dụng
- **Mục tiêu:** Xây dựng một mô hình phân biệt được các giao dịch có dấu hiệu giả và demo triển khai trong thực tế

---

## 🏗️ Business Understanding (Hiểu doanh nghiệp)

Một giao dịch được xem là một bản hợp đồng đối với doanh nghiệp:

Tuy nhiên nếu nó làm giả hoặc bị lợi dung nó sẽ được xem là một giao dịch gian lận. Điều này ảnh hưởng nặng nề tới doanh thu của một doanh nghiệp
  - **_Dự đoán sai giao dịch gian lận:_** Thiệt hại từ giao dịch, kẻ xấu có thể lợi dụng để chuộc lợi cho bản thân, khách hàng mất lòng tin dẫn đến chi phí khách hàng cao
    - Gian lận không là gian lận: Mất chi phí của một khách hàng, tuy nhiên không ảnh hưởng đến lợi ích lâu dài
  - **_Dự đoán đúng giao dịch gian lận:_** Khắc phục thiệt hại, cũng như loại bỏ các thành phần gây xâm phạm đến quyền lợi người tiêu dùng 
    - Gian lận là gian lận: Giảm chi phí khắc phục sự cố, bảo toàn quyền lợi khách hàng, gia tăng lòng tin khách hàng

--- 

## 📂 Dataset Description (Mô tả bộ dữ liệu)

- **Tên:** Credit Card Fraud Detection 
- **Nguồn:** Public dataset ([Kaggle – dữ liệu nghiên cứu học thuật](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Số dòng:** gồm **~284.807 khách hàng**
- **Số cột:** 31
- **Đối tượng:** **Khả năng** một giao dịch được xem là gian lận
### Một số thuộc tính quan trọng
- **Thông tin giao dịch:** `V1` -> `V28`
- **Thông tin thời gian:** `Time`, `Amount`
- **Thông tin giao dịch** `Class`

---

## 🧠 Data Science Perspective (Góc nhìn Khoa học dữ liệu)

### Challenges (Thách thức)

- Dữ liệu bị mất cân bằng cực kỳ nghiêm trọng
- Đặc trưng dã được PCA để đảm bảo tính bảo mật

### Modeling Strategy (Phương pháp chọn mô hình)

- Logistic Regressio (baseline)
- Random Forest
- Class imbalance handling (Xử lý mất cân bằng dữ liệu)
  - SMOTE
  - Class weight
- Threshold tuning (Tối ưu ngưỡng)
 - Maximize F1
 - Minimize loss    

--- 

## 📈 Evaluation Metrics

- Recall & Precision 
- PR-AUC (Do dữ liệu bị mất cân bằng)
- Confusion Maxtix

---

##  🏗 System Architecture (Kiến trúc hệ thống)
 
--- 

## 🗂️ Project Structure (Cấu trúc dự án)

```
credit-fraud-system/
│
├── configs/
│   └── config.yaml
│
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── api/
│   └── utils/
│
├── tests/
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🚀 API Deployment (Triển khai API)

---

## 🔎 Production Considerations (Các yếu tố cần xem xét trong sản phảm)

--- 

## 🔑 Key Learnings (Học hỏi)

- Thử nghiệm các phương pháp sử lý dữ liệu mất cân bằng nặng
- Tối ưu ngưỡng sao cho khả nặng nhận diện giao dịch cao mà còn hạn chế giao dịch báo động giả
- Thiết kê một ML-pipeline
- Xây dựng hướng sản phầm API
- Chuẩn bị huấn luyện trực tuyến lẫn ngoại tuyến

---

## 🔮Future Improvements (Cải thiện trong tương lai)

- Thêm model drift 
- Thêm CI/CD
- Deploy sử dụng Kubernetes 

---

## 👨‍🎓 Author (Tác giả)

- Sản phẩm **là bài làm gốc**
- Tên: **Nguyễn Đăng Tiến**
