import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict


def load_config(path: str) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    return json.loads(config_path.read_text())


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base


def resolve_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    cfg = deepcopy(config)
    exp_cfg = cfg.get("experiment", {})
    active = exp_cfg.get("active")
    definitions = exp_cfg.get("definitions", {})
    if active not in definitions:
        known = ", ".join(sorted(definitions))
        raise KeyError(f"Unknown experiment '{active}'. Known: {known}")
    cfg = _deep_update(cfg, definitions.get(active, {}))
    overrides = exp_cfg.get("overrides") or {}
    if overrides:
        cfg = _deep_update(cfg, overrides)
    cfg.setdefault("experiment", {})
    cfg["experiment"]["name"] = active
    return cfg


def load_and_resolve(path: str) -> Dict[str, Any]:
    return resolve_experiment(load_config(path))


def get_data_source(cfg: Dict[str, Any]) -> str:
    return cfg.get("data", {}).get("source", "synthetic").lower().strip()


def get_condition_label(cfg: Dict[str, Any]) -> str:
    source = get_data_source(cfg)
    if source == "ctr":
        return cfg.get("ctr", {}).get("condition_col", "Condition")
    return "Condition"


def loss_label(loss_mode: str) -> str:
    return "LE" if loss_mode.lower().strip() == "localized_entropy" else "BCE"
