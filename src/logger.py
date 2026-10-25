import os
import pandas as pd
import datetime
from datetime import datetime


def save_results(results, config):
    rows = []

    for model_name, metrics in results.items():
        rows.append({
            "date": datetime.now().strftime("%d %H:%M"),
            "version": config.project.version,
            "model": model_name,
            
            "rmse_mean": metrics["rmse_mean"],
           
            "rmse_std": metrics["rmse_std"],
            "config": metrics.get(
                "config",
                config.models.get(model_name, None),
            ),
        })

    df = pd.DataFrame(rows)
    path = config.output.experiments_path

    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.exists(path):
        old_df = pd.read_csv(path)
        df = pd.concat([old_df, df], ignore_index=True)

    df.to_csv(path, index=False)


def save_tuning_results(results, config):
    path = config.output.tuning_path

    os.makedirs(os.path.dirname(path), exist_ok=True)

    df = pd.DataFrame(results)

    
    df = df.sort_values(
        by="score",
        ascending=True,
    )

    df.to_csv(path, index=False)



def save_submission(predictions, test_df, config):
    """
    Сохраняет submission в формате Kaggle.
    predictions должны быть в исходном масштабе (долларах),
    не в логарифмах.
    """
    submission = pd.DataFrame({
        "Id": test_df[config.data.id_column],
        "SalePrice": predictions,
    })

    path = "results/submissions/submission.csv"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    submission.to_csv(path, index=False)

    return path