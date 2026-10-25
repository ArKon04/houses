import numpy as np

from sklearn.pipeline import Pipeline

from src.preprocessing import get_preprocessor, prepare_catboost
from src.models import get_models, create_model
from src.validation import evaluate_model


def train_single_model(model_name, params, X, y, config):
    model = create_model(model_name, params, config.project.random_state)

    if model_name == "catboost":
        X_catboost, cat_features = prepare_catboost(X, config)
        result = evaluate_model(model, X_catboost, y, config, cat_features)
    else:
        preprocessor = get_preprocessor(X, model_name, config)
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model),
        ])
        result = evaluate_model(pipeline, X, y, config)

    return result


def train_models(X, y, config):
    models = get_models(config)
    results = {}

    for name, params in models.items():
        metrics = train_single_model(
            model_name=name,
            params=params,
            X=X,
            y=y,
            config=config,
        )
        results[name] = metrics

    return results


def predict_model(model_name, params, X_train, y_train, X_valid, config):
    model = create_model(model_name, params, config.project.random_state)

    if model_name == "catboost":
        X_train, cat_features = prepare_catboost(X_train, config)
        X_valid, _ = prepare_catboost(X_valid, config)

        model.fit(
            X_train,
            y_train,
            cat_features=cat_features,
        )
    else:
        preprocessor = get_preprocessor(X_train, model_name, config)
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ])
        pipeline.fit(X_train, y_train)
        model = pipeline

    
    predictions = model.predict(X_valid)

    
    if getattr(config.data, "log_target", False):
        predictions = np.expm1(predictions)

    return predictions