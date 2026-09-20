"""
Preprocessing pipeline (feature scaling / encoding) for the credit
default prediction project.
"""
 
 
from __future__ import annotations
 
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
 
 
 
# NOTE: OpenML exposes the columns as anonymized names x1..x23
CATEGORICAL_COLS = ["x2", "x3", "x4", "x6", "x7", "x8", "x9", "x10", "x11"]
NUMERIC_COLS = [ "x1", "x5", "x12", "x13", "x14", "x15", "x16",
                "x17", "x18", "x19", "x20", "x21", "x22", "x23",]
 
 
def build_preprocessor() -> ColumnTransformer:
    """
    Build the preprocessing pipeline: OneHotEncoder for the
    categorical columns, RobustScaler for the numeric columns.
 
    Must be fit on the training split ONLY, then used to transform the
    validation and test splits, to avoid data leakage:
        preprocessor = build_preprocessor()
        X_train_scaled = preprocessor.fit_transform(X_train)
        X_val_scaled = preprocessor.transform(X_val)
        X_test_scaled = preprocessor.transform(X_test)
    """
    
    ohe = OneHotEncoder(drop="first")
    rs = RobustScaler()
    
    return make_column_transformer(
        (ohe, CATEGORICAL_COLS),
        (rs, NUMERIC_COLS),
    )