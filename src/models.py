from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from sklearn.neural_network import MLPRegressor


def get_models(config):
    models = {}

    if config.models.mlp.enabled:
        params = config.models.mlp

        models["mlp"] = {
            "hidden_layer_sizes": tuple(params.hidden_layer_sizes),
            "activation": params.activation,
            "solver": params.solver,
            "learning_rate_init": params.learning_rate_init,
            "batch_size": params.batch_size,
            "max_iter": params.max_iter,
            "alpha": params.alpha,
            "early_stopping": params.early_stopping,
        }

    if config.models.ridge.enabled:
        models["ridge"] = {
            "alpha": config.models.ridge.alpha,
            "max_iter": config.models.ridge.max_iter,
        }

    if config.models.knn.enabled:
        params = config.models.knn
        models["knn"] = {
            "n_neighbors": params.n_neighbors,
            "weights": params.weights,
            "metric": params.metric,
        }

    if config.models.decision_tree.enabled:
        params = config.models.decision_tree
        models["decision_tree"] = {
            "max_depth": params.max_depth,
        }

    if config.models.random_forest.enabled:
        params = config.models.random_forest
        models["random_forest"] = {
            "n_estimators": params.n_estimators,
            "max_depth": params.max_depth,
        }

    if config.models.catboost.enabled:
        params = config.models.catboost
        models["catboost"] = {
            "iterations": params.iterations,
            "learning_rate": params.learning_rate,
            "depth": params.depth,
            "loss_function": "RMSE",    
            "eval_metric": "RMSE",      
        }

    if config.models.xgboost.enabled:
        params = config.models.xgboost
        models["xgboost"] = {
            "n_estimators": params.n_estimators,
            "max_depth": params.max_depth,
            "learning_rate": params.learning_rate,
            "objective": "reg:squarederror",  
            "eval_metric": "rmse",           
        }

    return models


def create_model(model_name, params, random_state):

    if model_name == "mlp":
        return MLPRegressor(
            **params,
            random_state=random_state
        )

    if model_name == "ridge":
        return Ridge(**params, random_state=random_state)

    elif model_name == "knn":
        return KNeighborsRegressor(**params)

    elif model_name == "decision_tree":
        return DecisionTreeRegressor(**params, random_state=random_state)

    elif model_name == "random_forest":
        return RandomForestRegressor(**params, random_state=random_state)

    elif model_name == "catboost":
        params = dict(params)  # не мутируем исходный словарь
        params.setdefault("loss_function", "RMSE")
        params.setdefault("eval_metric", "RMSE")
        return CatBoostRegressor(
            **params,
            verbose=0,
            random_state=random_state,
        )

    elif model_name == "xgboost":
        params = dict(params)
        params.setdefault("objective", "reg:squarederror")
        params.setdefault("eval_metric", "rmse")
        return XGBRegressor(
            **params,
            random_state=random_state,
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")