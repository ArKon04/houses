from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

import numpy as np


def get_cv(config):
    # ИЗМЕНЕНО: StratifiedKFold -> KFold (в регрессии нет классов)
    return KFold(
        n_splits=config.validation.cv_folds,
        shuffle=True,
        random_state=config.project.random_state,
    )


def evaluate_model(model, X, y, config, cat_features=None):
    cv = get_cv(config)
    log_target = getattr(config.data, "log_target", False)

    scores = []

    if cat_features is not None:
        for train_idx, val_idx in cv.split(X, y):
            X_train = X.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_train = y.iloc[train_idx]
            y_val = y.iloc[val_idx]

            model.fit(
                X_train,
                y_train,
                cat_features=cat_features,
            )

            preds = model.predict(X_val)


            rmse = np.sqrt(mean_squared_error(y_val, preds))
            scores.append(rmse)

        scores = np.array(scores)

    else:
        # cross_val_score с neg_root_mean_squared_error возвращает
        # отрицательный RMSE (sklearn делает "negative" для унификации)
        raw_scores = cross_val_score(
            model,
            X,
            y,
            cv=cv,
            scoring="neg_root_mean_squared_error",
        )
        scores = -raw_scores

        # Если таргет в логарифмах — оценки уже в лог-пространстве,
        # expm1 здесь не применить напрямую (нет доступа к y).
        # Если хотите RMSE в долларах для этой ветки, нужно
        # делать ручной цикл, как для catboost.

    return {
        "rmse_mean": float(np.mean(scores)),
        "rmse_std": float(np.std(scores)),
    }