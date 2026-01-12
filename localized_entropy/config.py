import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional


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


def resolve_loss_modes(loss_mode) -> list:
    def normalize(mode: str) -> Optional[str]:
        if mode is None:
            return None
        text = str(mode).lower().strip()
        if text in {"le", "localized_entropy"}:
            return "localized_entropy"
        if text == "bce":
            return "bce"
        return None

    def expand(mode: str) -> list:
        text = str(mode).lower().strip()
        if text in {"both", "bce+le", "le+bce", "bce_le", "le_bce"}:
            return ["bce", "localized_entropy"]
        if "," in text:
            items = []
            for part in text.split(","):
                norm = normalize(part)
                if norm is not None:
                    items.append(norm)
            return items
        norm = normalize(text)
        return [norm] if norm is not None else []

    modes = []
    if isinstance(loss_mode, (list, tuple)):
        for item in loss_mode:
            if item is None:
                continue
            if isinstance(item, str) and item.lower().strip() in {
                "both",
                "bce+le",
                "le+bce",
                "bce_le",
                "le_bce",
            }:
                modes.extend(["bce", "localized_entropy"])
                continue
            norm = normalize(item)
            if norm is not None:
                modes.append(norm)
    else:
        modes.extend(expand(loss_mode))

    seen = set()
    ordered = []
    for mode in modes:
        if mode in seen:
            continue
        seen.add(mode)
        ordered.append(mode)
    return ordered
