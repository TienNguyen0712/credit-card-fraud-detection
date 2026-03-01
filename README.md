![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Machine Learning](https://img.shields.io/badge/Field-Machine%20Learning-purple)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-ML-yellow)

[Tiếng Việt](#README_vi.md)

# 🛡 Credit Card Fraud Detection – End-to-End ML Pipeline

## 🎯 Project Overview

- **Problem:** Financial transactions are highly vulnerable to fraud, especially in large-scale e-commerce systems. 
- **Challenges:** Fraud detection presents three keys challenges:
  - Extreme class imbalance (~0.17% fraud)
  - High cost of false negatives (missed fraud)
  - Business trade-off between customer friction and fraud loss
- **Goals:** This project builds an end-to-end machine learning pipeline to detect fraudulent transactions with a strong focus on: **Threshold tuning**, **Production-ready structure**, **Reproducible experiments**, **Recall optimization**

---

## 📂 Dataset Description

- **Name:** Credit Card Fraud Detection 
- **Source:** Public dataset ([Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud))
- **Total transactions:** about **~284.807**
- **Total target:** 30 anonymized features (PCA transformed) + `Time` + `Amount`
- **Target:** `Class` (1 = fraud, 0 = unfraud)
> Note: In the dataset feature V1 -> V28 are PCA transformed
---

## 🧠 Business Framing

| Scenario                             | Business Impact                     |
| ------------------------------------ | ----------------------------------- |
| False Negative (Missed Fraud)        | Direct financial loss, trust damage |
| False Positive (Flag legit as fraud) | Customer friction, operational cost |
| True Positive                        | Prevent fraud loss                  |
| True Negative                        | Normal operation                    |

**Therefore (Metrics to evaluate)**: 
- Recall & Precision is critical
- F1 - Score
- PR-AUC (imbalance data)
- Confusion Maxtix

--- 

## 🏗 System Architecture

```
Raw Data ( Hydra config )
   ↓
Data Loader
   ↓
Preprocessing + Feature Engineering
   ↓
Imbalance Handling ( Class Weight)
   ↓
Model Training 
   ↓
Threshold Optimization
   ↓
Evaluation
   ↓
Model Registry (MLflow)
   ↓
API Deployment
```

---

## ⚙ Modeling Strategy

### Baseline
- Logistic Regression

### Tree-based Models
- Random Forest
- XGBoost

### Imbalance Handling
- Class weighting
- Threshold tuning (F1 optimization, Recall for cost business)

--- 

## 🗂️ Project Structure 

```
credit-card-fraud-detection/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing_and_baseline_model.ipynb
│   └── 03_threshold_error_analysis.ipynb
│
├── src/
│   ├── data/
│   │   ├── load_data.py
│   │   └── split_data.py
│   │
│   ├── features/
│   │   ├── build_features.py
│   │   └── preprocess.py
│   │
│   ├── models/
│   │   ├── model_factory.py
│   │   └── trainer.py
│   │
│   ├── utils/
│   │   ├── evaluate.py
│   │   ├── threshold.py
│   │   └── helper.py
│   │
│   └── __init__.py
│
├── train.py
├── predict.py
│
├── experiments/
│   └── experiment_logs/
│
├── models/
│   └── saved_models/
│
├── configs/
│   ├── data/
│   │   └── default.yaml
|   |
│   ├── features/
│   │   └── default.yaml
|   | 
│   ├── mlflow/
│   │   └── default.yaml
|   |
│   ├── paths/
│   │   └── default.yaml
|   |
│   ├── training/
│   │   └── default.yaml
|   |
│   ├── model/
│   │   ├── logistic_regression.yaml
│   │   ├── random_forest.yaml
│   │   └── xgboost.yaml
│   │
│   ├── tuning.yaml
│   └── config.yaml
│
├── tests/
├── requirements.txt
└── README.md

```

---

## 🚀 How to Run

```bash
git clone https://github.com/yourusername/credit-card-fraud-detection
cd credit-card-fraud-detection
pip install -r requirements.txt

python train.py --config configs/config.yaml
python predict.py --input sample.csv
```

---

## 🧪 Experiment Tracking

- MLflow for:
  - Parameter logging
  - Metric tracking
  - Model versioning
-Config-driven experiments

---

## 🔎 Production Considerations

- Config-driven pipeline
- Modular architecture
- Threshold separated from model
- Ready for API deployment
- Online vs offline inference consideration
- Drift monitoring (planned)

---

## 🔮Future Improvements
- Add model drift detection
- CI/CD pipeline
- Dockerization
- Kubernetes deployment- 
- Real-time fraud streaming detection

---

## 👨‍🎓 Author

- Name: **Nguyễn Đăng Tiến**
- Role: **AI Engineer Candidate**
