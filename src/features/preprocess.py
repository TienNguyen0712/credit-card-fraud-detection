from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib


def clean_data(df):
    df.drop_duplicates(inplace=True)
    return df

def handle_outlier(df, cfg, outlier_report, multiplier=3.0):
    """
    Hàm này xử lý giá trị ngoại lai của lớp 
    đa số, Lớp giao dịch bình thường
    """
    df_final = df.copy()
    for index, row in outlier_report.iterrows():
        feature = row[cfg.features.summary_outlier.columns_outlier]

        Q1 = df_final[feature].quantile(0.25)
        Q3 = df_final[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR


        # Điều kiện: Là Class 0 VÀ nằm ngoài khoảng (lower_bound, upper_bound)
        indices_to_drop = df[(df[cfg.data.target_column] == 0) & ((df[feature] < lower_bound) | (df[feature] > upper_bound))].index
                
        # Loại bỏ các dòng thỏa mãn điều kiện trên
        df_final = df_final.drop(indices_to_drop, errors='ignore')
                
    return df_final

def get_preprocessor(cfg):
    """
    Khởi tạo ColumnTransformer dựa theo cấu hình config.yaml
    """

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")), 
        ("scaler", RobustScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_transformer,
                [cfg.features.log_amount_col]
            )
        ],
        remainder="passthrough"
    )

    return preprocessor
