"""
src/utils/config_loader.py
Loads the central YAML config and exposes it as a dict/dotdict.
"""
import yaml
from pathlib import Path


def load_config(config_path: str | Path = None) -> dict:
    """Load project config from YAML. Defaults to configs/config.yaml."""
    if config_path is None:
        # Walk up from this file to find project root
        root = Path(__file__).resolve().parents[2]
        config_path = root / "configs" / "config.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


class DotDict(dict):
    """Allows dot-notation access: config.modeling.xgboost.n_estimators"""
    def __getattr__(self, key):
        try:
            val = self[key]
            if isinstance(val, dict):
                return DotDict(val)
            return val
        except KeyError:
            raise AttributeError(f"Config has no key '{key}'")

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def get_config(config_path: str | Path = None) -> DotDict:
    return DotDict(load_config(config_path))
