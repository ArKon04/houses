import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import (
    FunctionTransformer,
    StandardScaler,
    OneHotEncoder,
)


def _safe_log1p(X):
    """
    log1p, безопасный для отрицательных значений.
    Отрицательные значения (например, отрицательный Age) обрезаются до 0.
    """
    X = np.asarray(X, dtype=np.float64)
    return np.log1p(np.clip(X, 0, None))


def create_standart_numeric_pipeline(config):
    steps = [
        ('imputer', SimpleImputer(strategy=config.preprocessing.missing_values)),
       
        ('log1p', FunctionTransformer(_safe_log1p, validate=False)),
        ("scaler", StandardScaler()),
    ]
    return Pipeline(steps=steps)


def create_standart_categorical_pipeline(config):
    steps = [
        
        ('imputer', SimpleImputer(strategy="constant", fill_value="None")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
    return Pipeline(steps=steps)


def create_tree_numeric_pipeline(config):
    steps = [
        ('imputer', SimpleImputer(strategy=config.preprocessing.missing_values)),
        
    ]
    return Pipeline(steps=steps)


def create_tree_categorical_pipeline(config):
    steps = [
        
        ('imputer', SimpleImputer(strategy="constant", fill_value="None")),
        ('encoder', OneHotEncoder(handle_unknown="ignore")),
    ]
    return Pipeline(steps=steps)


def create_standart_preprocessor(X, config):
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include=["object"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", create_standart_numeric_pipeline(config), numeric_features),
            ("cat", create_standart_categorical_pipeline(config), categorical_features),
        ]
    )
    return preprocessor


def create_tree_preprocessor(X, config):
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include=["object"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", create_tree_numeric_pipeline(config), numeric_features),
            ("cat", create_tree_categorical_pipeline(config), categorical_features),
        ]
    )
    return preprocessor


def prepare_catboost(X, config):
    params = config.models.catboost

    # Удаляем drop_columns (Id)
    drop_cols = [c for c in params.drop_columns if c in X.columns]
    X = X.drop(columns=drop_cols)

    X = X.copy()

    # все object-колонки — категориальные
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    for col in categorical_features:
        X[col] = X[col].fillna("None").astype(str)

    # Числовые колонки: заполняем NaN медианой
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
    for col in numeric_cols:
        X[col] = X[col].fillna(X[col].median())

    return X, categorical_features


def get_preprocessor(X, model_name, config):
    method = config.models[model_name].preprocessing

    if method == "standart":
        return create_standart_preprocessor(X, config)

    if method == "tree":
        return create_tree_preprocessor(X, config)