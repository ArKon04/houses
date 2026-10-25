import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error

from src.models import get_models, create_model
from src.preprocessing import get_preprocessor, prepare_catboost
from src.validation import get_cv
from sklearn.linear_model import Ridge


def get_ensemble_models(X, config):
    """
    Создает модели для ансамбля.
    """
    all_models = get_models(config)
    models = {}
    ensemble_models = config.ensemble.weights.keys()

    for name in ensemble_models:
        params = all_models[name]
        model = create_model(name, params, config.project.random_state)

        if name == "catboost":
            X_cat, cat_features = prepare_catboost(X, config)
            models[name] = {
                "model": model,
                "X": X_cat,
                "cat_features": cat_features,
            }
        else:
            preprocessor = get_preprocessor(X, name, config)
            pipeline = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ])
            models[name] = {
                "model": pipeline,
                "X": X,
                "cat_features": None,
            }

    return models


def get_oof_predictions(models, y, config):
    """
    Получает честные out-of-fold predictions.
    """
    cv = get_cv(config)
    predictions = {}

    for name, data in models.items():
        model = data["model"]
        X = data["X"]

        if name == "catboost":
        
            preds = cross_val_predict(
                model,
                X,
                y,
                cv=cv,
                method="predict",
                params={"cat_features": data["cat_features"]},
            )
        else:
            preds = cross_val_predict(
                model,
                X,
                y,
                cv=cv,
                method="predict",
            )

     
        predictions[name] = preds

    return predictions


def blend_predictions(predictions, y, weights, config):
    """
    Weighted average предсказаний (регрессия).
    """
    final_prediction = np.zeros(len(y))

    for name, pred in predictions.items():
        final_prediction += pred * weights[name]


    rmse = np.sqrt(mean_squared_error(y, final_prediction))
    return rmse


def run_ensemble(X, y, config):
    models = get_ensemble_models(X, config)
    predictions = get_oof_predictions(models, y, config)

    weighted_score = blend_predictions(
        predictions, y, config.ensemble.weights, config
    )
    voting_score = voting_predictions(predictions, y, config)
    stacking_score = stacking_predictions(predictions, y, config)

    return {
        "weighted_average": {
           
            "rmse_mean": weighted_score,
           
            "rmse_std": 0,
            "config": config.ensemble.weights,
        },
        "voting": {
            "rmse_mean": voting_score,
            "rmse_std": 0,
            "config": "simple average",
        },
        "stacking": {
            "rmse_mean": stacking_score,
            "rmse_std": 0,
            "config": "ridge",
        },
    }


def voting_predictions(predictions, y, config):
    """
    ИЗМЕНЕНО: hard voting -> простое усреднение.
    В регрессии нет классов, поэтому вместо голосования — среднее.
    """
    all_preds = np.array([pred for pred in predictions.values()])
    final_prediction = np.mean(all_preds, axis=0)

    rmse = np.sqrt(mean_squared_error(y, final_prediction))
    return rmse


def stacking_predictions(predictions, y, config):
    """
    Stacking через Ridge (регрессия).
    """
    X_stack = np.column_stack([
        predictions[name] for name in predictions
    ])

    cv = get_cv(config)


    meta_model = Ridge(random_state=config.project.random_state)

  
    meta_predictions = cross_val_predict(
        meta_model,
        X_stack,
        y,
        cv=cv,
        method="predict",
    )

    
    rmse = np.sqrt(mean_squared_error(y, meta_predictions))
    return rmse