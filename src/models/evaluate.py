import joblib
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score, precision_recall_curve
from src.data.preprocess import load_data, split_scale_data


def plot_pr_curve():
        
    model = joblib.load("models/model.pkl")
    
    y_probs = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    plt.plot(recall, precision, label=f'{label} (AP={average_precision_score(y_test, y_probs):.2f})')
    plt.figure(figsize=(8, 6))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('So sánh Precision-Recall Curve')
    plt.legend()
    plt.show()

def evaluate():
    df = load_data("data/creditcard.csv")

    X_train, X_test, y_train, y_test = split_data(df)
    X_train, X_test, scaler = scale_data(X_train, X_test)

    model = joblib.load("models/model.pkl")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

def cross_validation(k):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    
    model = joblib.load("models/model.pkl")
    results = cross_val_score(model, X_train, y_train, cv=skf, scoring='average_precision')

    print(f"AP trung bình: {results.mean():.2f} (+/- {results.std():.2f})")
    
    
    


if __name__ == "__main__":
    evaluate()
    plot_pr_curve()




