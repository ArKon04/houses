from omegaconf import OmegaConf
from pathlib import Path


def load_config(path):
    return OmegaConf.load(path)