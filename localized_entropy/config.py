import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional


def load_config(path: str) -> Dict[str, Any]:
    """Load a JSON config file from disk."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    return json.loads(config_path.read_text())


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge updates into a base dict."""
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base


def resolve_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply experiment definitions and overrides to a config payload."""
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
    """Load config from disk and apply experiment overrides."""
    return resolve_experiment(load_config(path))


def get_data_source(cfg: Dict[str, Any]) -> str:
    """Return the normalized data source name."""
    return cfg.get("data", {}).get("source", "synthetic").lower().strip()


def get_condition_label(cfg: Dict[str, Any]) -> str:
    """Return a display label for condition IDs."""
    source = get_data_source(cfg)
    if source == "ctr":
        return cfg.get("ctr", {}).get("condition_col", "Condition")
    return "Condition"


def loss_label(loss_mode: str) -> str:
    """Map a loss mode string to a short label."""
    text = loss_mode.lower().strip()
    if text == "localized_entropy":
        return "LE"
    if text == "bce":
        return "BCE"
    if text == "focal":
        return "Focal"
    return text.upper()


def _normalize_loss_mode(mode: Optional[str]) -> Optional[str]:
    """Normalize a loss mode string to a canonical value."""
    if mode is None:
        return None
    text = str(mode).lower().strip()
    if text in {"le", "localized_entropy"}:
        return "localized_entropy"
    if text == "bce":
        return "bce"
    if text in {"focal", "focal_loss", "focal-loss", "fl"}:
        return "focal"
    return None


def resolve_loss_modes(loss_mode) -> list:
    """Normalize loss mode configuration into a list of modes."""

    def expand(mode: str) -> list:
        """Expand combined loss-mode strings into a list."""
        text = str(mode).lower().strip()
        if text in {"both", "bce+le", "le+bce", "bce_le", "le_bce"}:
            return ["bce", "localized_entropy"]
        if text in {
            "all",
            "bce+le+focal",
            "bce+focal+le",
            "le+bce+focal",
            "le+focal+bce",
            "focal+bce+le",
            "focal+le+bce",
        }:
            return ["bce", "localized_entropy", "focal"]
        if "," in text or "+" in text:
            clean = text.replace("+", ",")
            items = []
            for part in clean.split(","):
                norm = _normalize_loss_mode(part)
                if norm is not None:
                    items.append(norm)
            return items
        norm = _normalize_loss_mode(text)
        return [norm] if norm is not None else []

    modes = []
    if isinstance(loss_mode, (list, tuple)):
        for item in loss_mode:
            if item is None:
                continue
            if isinstance(item, str):
                expanded = expand(item)
                if expanded:
                    modes.extend(expanded)
                    continue
            norm = _normalize_loss_mode(item)
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


def resolve_training_cfg(cfg: Dict[str, Any], loss_mode: Optional[str] = None) -> Dict[str, Any]:
    """Resolve training config, including optional per-loss/per-source overrides."""
    training_cfg = cfg.get("training")
    if not isinstance(training_cfg, dict):
        return {}
    resolved = deepcopy(training_cfg)
    if loss_mode is None:
        return resolved
    by_loss = training_cfg.get("by_loss")
    if not isinstance(by_loss, dict):
        return resolved
    normalized = {}
    for key, value in by_loss.items():
        norm = _normalize_loss_mode(key)
        if norm is None or not isinstance(value, dict):
            continue
        normalized[norm] = value
    overrides = normalized.get(_normalize_loss_mode(loss_mode))
    if overrides:
        overrides = deepcopy(overrides)
        source_overrides = overrides.pop("by_source", None)
        resolved = _deep_update(resolved, overrides)
        if isinstance(source_overrides, dict):
            normalized_sources = {}
            for key, value in source_overrides.items():
                if not isinstance(value, dict):
                    continue
                normalized_sources[str(key).lower().strip()] = value
            source = get_data_source(cfg)
            source_cfg = normalized_sources.get(source)
            if source_cfg:
                resolved = _deep_update(resolved, source_cfg)
    return resolved
