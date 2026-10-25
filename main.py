import numpy as np

from src.data_loader import load_data
from src.config import load_config
from src.train import train_models, train_single_model
from src.logger import save_results
from src.features import apply_features
from src.tuning.optuna_runner import optimize_catboost, optimize_xgboost
from src.ensemble import run_ensemble


def main():
    config = load_config("configs/config.yaml")

    # загрузка данных
    train, test = load_data(config)

    # подготовка данных
    X = train.drop(config.data.target, axis=1)
    X = apply_features(X, config)

    y = train[config.data.target]

    
    if getattr(config.data, "log_target", False):
        y = np.log1p(y)

 
    if config.experiment.mode == "tuning":

        if config.tuning.models.catboost.enabled:
            study = optimize_catboost(
                X,
                y,
                config,
                train_function=train_single_model,  
            )
        elif config.tuning.models.xgboost.enabled:
            study = optimize_xgboost(
                X,
                y,
                config,
                train_function=train_single_model,  
            )
        else:
            raise ValueError("No tuning model enabled")

       
        results = {
            "optuna": {
                "rmse_mean": study.best_value,
                "rmse_std": 0,
                "config": study.best_params,
            }
        }

    elif config.experiment.mode == "ensemble":
        results = run_ensemble(X, y, config)

    elif config.experiment.mode == "train":
        results = train_models(X, y, config)

    else:
        raise ValueError(f"Unknown mode: {config.experiment.mode}")

    # сохранение результатов
    save_results(results, config)

    # вывод результата
    for model, score in sorted(
        results.items(),
        key=lambda x: x[1]["rmse_mean"],
        reverse=False,  
    ):
        print(model, round(score["rmse_mean"], 4))


if __name__ == "__main__":
    main()