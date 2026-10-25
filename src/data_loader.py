import pandas as pd
import numpy as np


def load_data(config):

    train = pd.read_csv(config.data.train_path)
    test = pd.read_csv(config.data.test_path)

    return train, test