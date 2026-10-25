# Houses Classical ML Regressor

Пайплайн для предсказания стоимости домов (регрессия) на датасете
[Kaggle House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).

Реализовано:

- подготовка данных и feature engineering;
- preprocessing (числовой и категориальный);
- K-Fold Cross Validation;
- 6 моделей: Ridge, KNN, DecisionTree, RandomForest, CatBoost, XGBoost;
- ансамблирование: weighted average, voting, stacking;
- подбор гиперпараметров через Optuna;
- сохранение истории экспериментов.

---

## Структура проекта

```
final_houses/

├── configs/
│   └── config.yaml              # параметры эксперимента
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── notebooks/
│   └── EDA.ipynb                # исследовательский анализ данных
│
├── results/
│   ├── scores.csv               # метрики по всем запускам
│   ├── tuning_results.csv       # результаты Optuna
│   └── submissions/             # submission для Kaggle
│
├── src/
│   ├── config.py                # загрузка конфига
│   ├── data_loader.py           # загрузка train/test
│   ├── features.py              # feature engineering
│   ├── preprocessing.py         # пайплайны препроцессинга
│   ├── models.py                # фабрика моделей
│   ├── validation.py            # KFold + RMSE
│   ├── train.py                 # обучение отдельных моделей
│   ├── ensemble.py              # ансамблирование
│   ├── logger.py                # сохранение результатов
│   └── tuning/
│       ├── __init__.py
│       ├── search_space.py      # Optuna
│       └── optuna_runner.py     # оптимизация
│
├── main.py                      # точка входа
├── requirements.txt
└── README.md
```

---

## Dataset

Скачать: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

Разместить в папке `data/`:

```
data/
├── train.csv
└── test.csv
```

- Целевая переменная: `SalePrice`
- Идентификатор: `Id`

---

## Data Preprocessing

Preprocessing выполняется **внутри каждого fold**, что исключает утечку
данных между train и validation.

### Обработка пропусков

| Тип признаков | Стратегия |
|---|---|
| Числовые | `median` |
| Категориальные | `constant "None"` |

`NaN` в категориальных признаках House Prices (`PoolQC`, `Alley`, `GarageType`,
`BsmtQual`, `FireplaceQu` и др.) означает **отсутствие объекта**, а не
«неизвестно». Поэтому пропуск заменяется строкой `"None"`, и модель получает
отдельную категорию «нет бассейна / нет гаража / нет подвала».

### Кодирование категорий

One-Hot Encoding (`handle_unknown="ignore"`).

### Масштабирование

- **`standart`** пайплайн (Ridge, KNN):
  1. `SimpleImputer(strategy="median")`
  2. `log1p` — сжатие длинных хвостов числовых признаков
  3. `StandardScaler`
- **`tree`** пайплайн (DecisionTree, RandomForest, XGBoost):
  1. `SimpleImputer(strategy="median")`

Деревья инвариантны к монотонным преобразованиям, поэтому `log1p` для них
не нужен.

### CatBoost

- Категориальные признаки определяются автоматически:
  `X.select_dtypes(include=["object"]).columns`
- `NaN` в категориальных → `"None"`, приведение к `str`
- `NaN` в числовых → медиана
- Колонки из `catboost.drop_columns` удаляются из `X`

---

## Feature Engineering

| Признак | Формула |
|---|---|
| `TotalSF` | `TotalBsmtSF + 1stFlrSF + 2ndFlrSF` |
| `TotalBath` | `FullBath + 0.5·HalfBath + BsmtFullBath + 0.5·BsmtHalfBath` |
| `TotalPorch` | `OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch` |
| `Age` | `YrSold − YearBuilt` (clip lower=0) |
| `RemodAge` | `YrSold − YearRemodAdd` (clip lower=0) |
| `HasPool` | `PoolArea > 0` |
| `Has2ndFloor` | `2ndFlrSF > 0` |
| `HasGarage` | `GarageArea > 0` |
| `HasBsmt` | `TotalBsmtSF > 0` |
| `HasFireplace` | `Fireplaces > 0` |

`clip(lower=0)` защищает от отрицательных значений, если `YearBuilt` или
`YearRemodAdd` по ошибке данных оказались позже `YrSold`.

### Логарифмирование таргета

`SalePrice` имеет skewness ≈ 1.88. Модель обучается на `log1p(SalePrice)`,
метрики считаются в log-пространстве. Управляется флагом в конфиге:

```yaml
data:
  log_target: true
```

При формировании submission применяется обратное преобразование
`np.expm1(predictions)` — цены возвращаются в доллары.

---

## Models

| Модель | Класс | Ключевые параметры |
|---|---|---|
| Ridge | `sklearn.linear_model.Ridge` | `alpha`, `max_iter` |
| KNN | `sklearn.neighbors.KNeighborsRegressor` | `n_neighbors`, `weights`, `metric` |
| DecisionTree | `sklearn.tree.DecisionTreeRegressor` | `max_depth` |
| RandomForest | `sklearn.ensemble.RandomForestRegressor` | `n_estimators`, `max_depth` |
| CatBoost | `catboost.CatBoostRegressor` | `iterations`, `learning_rate`, `depth`, `l2_leaf_reg`, `loss_function=RMSE` |
| XGBoost | `xgboost.XGBRegressor` | `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `objective=reg:squarederror` |

---

## Validation

```yaml
validation:
  method: kfold
  cv_folds: 5
  shuffle: true
```

`StratifiedKFold` не применяется — в задаче регрессии нет классов.

Метрика — **RMSE** в log-пространстве (меньше — лучше).

Для CatBoost реализован отдельный цикл CV, так как `cat_features` не
совместим с `cross_val_score`.

---

## Training

- **Логарифмирование таргета** выполняется один раз в `main.py`:
  `y = np.log1p(y)`, если `config.data.log_target=True`.
- **Early stopping** для бустингов не реализован — модели обучаются на
  фиксированном числе итераций.
- **Reproducibility**: `random_state=42` фиксируется во всех моделях и в KFold.

---

## Ensemble

Три метода:

1. **weighted_average** — взвешенная смесь предсказаний с весами из конфига.
2. **voting** — простое среднее всех моделей.
3. **stacking** — Ridge поверх OOF-предсказаний базовых моделей.

Веса по умолчанию:

```yaml
ensemble:
  weights:
    ridge: 0.15
    catboost: 0.50
    xgboost: 0.35
```

OOF-предсказания получаются через `cross_val_predict` — каждая модель
обучается 5 раз на 4/5 данных, предсказания на оставшейся части становятся
входом для мета-модели.

---

## Tuning (Optuna)

```yaml
experiment:
  mode: tuning
tuning:
  enabled: true
  n_trials: 40
```

Направление оптимизации — `minimize` (RMSE).

Пространства поиска:

**CatBoost:**
- `iterations`: 300–1500
- `depth`: 4–10
- `learning_rate`: 0.01–0.3 (log)
- `l2_leaf_reg`: 1–10

**XGBoost:**
- `n_estimators`: 100–1500
- `max_depth`: 3–10
- `learning_rate`: 0.01–0.3 (log)
- `subsample`: 0.5–1.0
- `colsample_bytree`: 0.5–1.0
- `min_child_weight`: 1–10

Результаты сохраняются в `results/tuning_results.csv`.

---

## Experiment Tracking

Каждый запуск добавляет строку в `results/scores.csv`:

| Колонка | Описание |
|---|---|
| `date` | дата запуска |
| `version` | `config.project.version` |
| `model` | имя модели или метод ансамбля |
| `rmse_mean` | средний RMSE по фолдам |
| `rmse_std` | стандартное отклонение RMSE |
| `config` | гиперпараметры модели или веса ансамбля |

---

## Results

### Одиночные модели (mode: train)

| Модель | RMSE (log) |
|---|---:|
| CatBoost | **0.1243** |
| XGBoost | 0.1327 |
| Ridge | 0.1343 |
| RandomForest | 0.1458 |
| KNN | 0.1763 |
| DecisionTree | 0.1987 |

### Ансамбли (mode: ensemble)

| Метод | RMSE (log) |
|---|---:|
| voting | **0.1236** |
| weighted_average | **0.1236** |
| stacking | 0.1246 |

**Для сравнения:**

- константное предсказание (среднее): RMSE (log) ≈ 0.40
- топ Kaggle: 0.11–0.12
- средний уровень участника: 0.13–0.14

---

## Installation

Создание окружения:

```bash
python -m venv venv
```

Активация:

```bash
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate          # Windows
```

Установка зависимостей:

```bash
pip install -r requirements.txt
```

---

## Training

Все параметры запуска задаются в `configs/config.yaml`.

### Режим обучения отдельных моделей

```yaml
experiment:
  mode: train
```

```bash
python main.py
```

Будут обучены все модели с `enabled: true` в секции `models`.

### Режим ансамблирования

```yaml
experiment:
  mode: ensemble
```

```bash
python main.py
```

Будут выведены RMSE для weighted average, voting и stacking.

### Режим подбора гиперпараметров

```yaml
experiment:
  mode: tuning
tuning:
  enabled: true
  n_trials: 40
  models:
    catboost:
      enabled: true
    xgboost:
      enabled: false
```

```bash
python main.py
```

Результаты сохраняются в `results/tuning_results.csv`.

---

## Configuration

Все параметры в `configs/config.yaml`:

| Секция | Что настраивается |
|---|---|
| `project` | имя, seed, версия |
| `experiment` | режим: `train` / `ensemble` / `tuning` |
| `data` | пути, target, id_column, log_target |
| `validation` | метод, число фолдов, shuffle |
| `preprocessing` | стратегия заполнения пропусков, drop_columns |
| `features` | какие фичи включать |
| `models` | гиперпараметры каждой модели |
| `ensemble` | веса, метод, мета-модель |
| `tuning` | диапазоны Optuna |
| `output` | пути сохранения результатов |

---

## Requirements

- Python 3.10+
- pandas, numpy, scikit-learn
- catboost, xgboost
- optuna
- omegaconf

Полный список — в `requirements.txt`.

Выводы:

catboost 0.1243
xgboost 0.1327
ridge 0.1343
random_forest 0.1457
knn 0.1763
mlp 0.1802
decision_tree 0.1987
