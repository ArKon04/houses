import numpy as np
import pandas as pd


def add_total_sf(df):
    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    return df


def add_total_bath(df):
    df["TotalBath"] = (
        df["FullBath"]
        + 0.5 * df["HalfBath"]
        + df["BsmtFullBath"]
        + 0.5 * df["BsmtHalfBath"]
    )
    return df


def add_total_porch(df):
    df["TotalPorch"] = (
        df["OpenPorchSF"]
        + df["EnclosedPorch"]
        + df["3SsnPorch"]
        + df["ScreenPorch"]
    )
    return df


def add_age(df):
    # clip(lower=0) — защита от отрицательных значений,
    # если YearBuilt/YearRemodAdd позже YrSold (ошибка в данных)
    df["Age"] = (df["YrSold"] - df["YearBuilt"]).clip(lower=0)
    df["RemodAge"] = (df["YrSold"] - df["YearRemodAdd"]).clip(lower=0)
    return df


def add_has_pool(df):
    df["HasPool"] = (df["PoolArea"] > 0).astype(int)
    return df


def add_has_2nd_floor(df):
    df["Has2ndFloor"] = (df["2ndFlrSF"] > 0).astype(int)
    return df


def add_has_garage(df):
    df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
    return df


def add_has_bsmt(df):
    df["HasBsmt"] = (df["TotalBsmtSF"] > 0).astype(int)
    return df


def add_has_fireplace(df):
    df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)
    return df


def apply_features(X, config):
    if not config.features.enabled:
        return X

    X = X.copy()

    if config.features.total_sf:
        X = add_total_sf(X)

    if config.features.total_bath:
        X = add_total_bath(X)

    if config.features.total_porch:
        X = add_total_porch(X)

    if config.features.age:
        X = add_age(X)

    if config.features.has_pool:
        X = add_has_pool(X)

    if config.features.has_2nd_floor:
        X = add_has_2nd_floor(X)

    if config.features.has_garage:
        X = add_has_garage(X)

    if config.features.has_bsmt:
        X = add_has_bsmt(X)

    if config.features.has_fireplace:
        X = add_has_fireplace(X)

    return X