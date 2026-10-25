import optuna

from src.tuning.search_space import catboost_space, xgboost_space


def optimize_catboost(X, y, config, train_function):

    def objective(trial):
        params = catboost_space(trial, config)

        result = train_function(
            model_name="catboost",
            params=params,
            X=X,
            y=y,
            config=config,
        )

        return result["rmse_mean"]

    study = optuna.create_study(direction="minimize")

    study.optimize(
        objective,
        n_trials=config.tuning.n_trials,
    )

    return study


def optimize_xgboost(X, y, config, train_function):

    def objective(trial):
        params = xgboost_space(trial, config)

        result = train_function(
            model_name="xgboost",
            params=params,
            X=X,
            y=y,
            config=config,
        )

        
        return result["rmse_mean"]

   
    study = optuna.create_study(direction="minimize")

    study.optimize(
        objective,
        n_trials=config.tuning.n_trials,
    )

    return study