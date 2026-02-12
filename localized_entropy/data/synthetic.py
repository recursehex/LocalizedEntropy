import warnings
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.stats import norm

REWEIGHTING_DEPRECATION_MESSAGE = (
    "synthetic.reweighting is deprecated and will be removed in a future release. "
    "Disable synthetic.reweighting.enabled to silence this warning."
)


def _sigmoid(x: np.ndarray, mu: float, s: float) -> np.ndarray:
    """Compute a shifted/scaled sigmoid."""
    return 1.0 / (1.0 + np.exp(-(x - mu) / s))


def _interest_prob(ages: np.ndarray, mu_age: float, sigma_age: float, interest_scale: float) -> np.ndarray:
    """Compute a normalized age-based interest curve."""
    if interest_scale <= 0:
        raise ValueError("interest_scale must be > 0.")
    denom = norm.pdf(50, loc=mu_age, scale=sigma_age)
    denom = denom if denom > 0 else 1.0
    interest_prob = norm.pdf(ages, loc=mu_age, scale=sigma_age) / denom
    return interest_prob / interest_scale


def _ranked_normal_probs(
    base_probs: np.ndarray,
    target_mean: float,
    band_lo_log: float,
    band_hi_log: float,
    sigma_log10: float,
    tol: float = 1e-8,
    max_iter: int = 60,
) -> Tuple[np.ndarray, bool]:
    """Map base-prob ranks to a log10-normal band with target mean."""
    n = base_probs.size
    if n == 0:
        return base_probs.astype(np.float32), False
    if sigma_log10 <= 0:
        return np.full_like(base_probs, target_mean, dtype=np.float32), False
    if not np.isfinite(band_lo_log) or not np.isfinite(band_hi_log) or band_hi_log <= band_lo_log:
        return np.full_like(base_probs, target_mean, dtype=np.float32), False

    order = np.argsort(base_probs, kind="mergesort")
    q = np.empty(n, dtype=np.float64)
    q[order] = (np.arange(n) + 0.5) / n
    q = np.clip(q, 1e-6, 1.0 - 1e-6)
    z = norm.ppf(q)

    def mean_for(mu_log10: float) -> float:
        log10p = mu_log10 + sigma_log10 * z
        log10p = np.clip(log10p, band_lo_log, band_hi_log)
        return float(np.mean(10.0 ** log10p))

    low = float(band_lo_log)
    high = float(band_hi_log)
    mean_low = mean_for(low)
    mean_high = mean_for(high)
    if target_mean <= mean_low:
        mu_log10 = low
    elif target_mean >= mean_high:
        mu_log10 = high
    else:
        mu_log10 = 0.5 * (low + high)
        for _ in range(max_iter):
            mu_mid = 0.5 * (low + high)
            mean_mid = mean_for(mu_mid)
            mu_log10 = mu_mid
            if abs(mean_mid - target_mean) <= tol:
                break
            if mean_mid < target_mean:
                low = mu_mid
            else:
                high = mu_mid

    log10p = mu_log10 + sigma_log10 * z
    clipped = bool(np.any(log10p < band_lo_log) or np.any(log10p > band_hi_log))
    log10p = np.clip(log10p, band_lo_log, band_hi_log)
    probs = (10.0 ** log10p).astype(np.float32, copy=False)
    return probs, clipped


def _ranked_log10_normal_probs(
    base_probs: np.ndarray,
    mu_log10: float,
    sigma_log10: float,
    log10_min: Optional[float] = None,
    log10_max: Optional[float] = 0.0,
) -> Tuple[np.ndarray, bool]:
    """Map base-prob ranks to a log10-normal distribution with given mean/std."""
    n = base_probs.size
    if n == 0:
        return base_probs.astype(np.float32), False

    order = np.argsort(base_probs, kind="mergesort")
    q = np.empty(n, dtype=np.float64)
    q[order] = (np.arange(n) + 0.5) / n
    q = np.clip(q, 1e-6, 1.0 - 1e-6)
    z = norm.ppf(q)

    if sigma_log10 <= 0:
        log10p = np.full(n, mu_log10, dtype=np.float64)
    else:
        log10p = mu_log10 + sigma_log10 * z

    clipped = False
    if log10_max is not None:
        clipped = clipped or bool(np.any(log10p > log10_max))
        log10p = np.minimum(log10p, log10_max)
    if log10_min is not None:
        clipped = clipped or bool(np.any(log10p < log10_min))
        log10p = np.maximum(log10p, log10_min)

    probs = (10.0 ** log10p).astype(np.float32, copy=False)
    probs = np.clip(probs, 0.0, 1.0)
    return probs, clipped


def _solve_mu_log10_for_target(target_mean: float, sigma_log10: float) -> float:
    """Solve for log10 mean of a log10-normal."""
    if sigma_log10 <= 0:
        return float(np.log10(max(target_mean, 1e-12)))
    ln10 = np.log(10.0)
    return float(np.log10(max(target_mean, 1e-12)) - 0.5 * (sigma_log10 ** 2) * ln10)


def _solve_mu_log10_for_target_centered(
    target_mean: float,
    band_width: float,
    sigma_log10: float,
    tol: float = 1e-8,
    max_iter: int = 60,
    grid_size: int = 4096,
) -> float:
    """Solve for log10 mean with a band centered on the mean."""
    if sigma_log10 <= 0:
        return float(np.log10(max(target_mean, 1e-12)))
    q = (np.arange(grid_size) + 0.5) / grid_size
    z = norm.ppf(np.clip(q, 1e-6, 1.0 - 1e-6))

    def mean_for(mu_log10: float) -> float:
        log10p = mu_log10 + sigma_log10 * z
        if band_width > 0:
            half = 0.5 * band_width
            log10p = np.clip(log10p, mu_log10 - half, mu_log10 + half)
        return float(np.mean(10.0 ** log10p))

    mu_center = float(np.log10(max(target_mean, 1e-12)))
    low = mu_center - 6.0 * sigma_log10 - 1.0
    high = mu_center + 6.0 * sigma_log10 + 1.0
    mean_low = mean_for(low)
    mean_high = mean_for(high)
    while mean_low > target_mean:
        high = low
        low -= 2.0
        mean_low = mean_for(low)
    while mean_high < target_mean:
        low = high
        high += 2.0
        mean_high = mean_for(high)

    mu_log10 = mu_center
    for _ in range(max_iter):
        mu_mid = 0.5 * (low + high)
        mean_mid = mean_for(mu_mid)
        mu_log10 = mu_mid
        if abs(mean_mid - target_mean) <= tol:
            break
        if mean_mid < target_mean:
            low = mu_mid
        else:
            high = mu_mid
    return mu_log10


def _sample_truncated_log10_normal(
    rng: np.random.Generator,
    size: int,
    mu_log10: float,
    sigma_log10: float,
    band_lo_log: float,
    band_hi_log: float,
    max_iter: int = 20,
) -> Tuple[np.ndarray, bool]:
    """Sample log10 probabilities from a truncated normal distribution."""
    if size <= 0:
        return np.empty((0,), dtype=np.float32), False
    if sigma_log10 <= 0:
        return np.full(size, mu_log10, dtype=np.float32), False
    samples = rng.normal(loc=mu_log10, scale=sigma_log10, size=size)
    if band_hi_log > band_lo_log:
        mask = (samples < band_lo_log) | (samples > band_hi_log)
        for _ in range(max_iter):
            if not np.any(mask):
                break
            samples[mask] = rng.normal(loc=mu_log10, scale=sigma_log10, size=int(mask.sum()))
            mask = (samples < band_lo_log) | (samples > band_hi_log)
        clipped = bool(np.any(mask))
        if clipped:
            samples = np.clip(samples, band_lo_log, band_hi_log)
    else:
        clipped = False
    return samples.astype(np.float32, copy=False), clipped


def generate_probs(
    num_samples: int,
    mu_ln: float,
    sigma_ln: float,
    sig_mu: float,
    sig_s: float,
    mu_age: float,
    sigma_age: float,
    interest_scale: float,
    min_age: int = 10,
    max_age: int = 100,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample synthetic net worth, age, and click probabilities."""
    if rng is None:
        rng = np.random.default_rng()

    net_worth = rng.lognormal(mean=mu_ln, sigma=sigma_ln, size=num_samples)
    probs = _sigmoid(np.log10(net_worth + 1.0), mu=sig_mu, s=sig_s)

    ages = rng.integers(min_age, max_age, size=num_samples)
    probs = probs * _interest_prob(ages, mu_age, sigma_age, interest_scale)
    probs = np.clip(probs, 0.0, 1.0)
    return net_worth.astype(np.float32), ages.astype(np.float32), probs.astype(np.float32)


def _sample_condition_params(cfg: Dict, rng: np.random.Generator) -> Tuple[float, float, float, float, float, float, float]:
    """Sample per-condition parameters for synthetic data generation."""
    sig_mu_range = cfg["sigmoid_mu_range"]
    sig_s_range = cfg["sigmoid_s_range"]
    age_mu_range = cfg["age_mu_range"]
    age_sigma_range = cfg["age_sigma_range"]
    interest_range = cfg["interest_scale_log10_range"]

    mu_ln = cfg["base_mu_ln"]
    sigma_ln = cfg["base_sigma_ln"]
    sig_mu = rng.choice(np.linspace(sig_mu_range[0], sig_mu_range[1], sig_mu_range[2]))
    sig_s = rng.choice(np.linspace(sig_s_range[0], sig_s_range[1], sig_s_range[2]))
    mu_age = rng.choice(np.linspace(age_mu_range[0], age_mu_range[1], age_mu_range[2]))
    sigma_age = rng.choice(np.linspace(age_sigma_range[0], age_sigma_range[1], age_sigma_range[2]))
    interest_scale = 10.0 ** (rng.choice(np.linspace(interest_range[0], interest_range[1], interest_range[2])))
    return mu_ln, sigma_ln, sig_mu, sig_s, mu_age, sigma_age, interest_scale


def make_dataset(
    cfg: Dict,
    seed: int,
) -> Dict[str, np.ndarray]:
    """Generate a synthetic dataset with per-condition distributions."""
    rng = np.random.default_rng(seed)
    num_conditions = int(cfg["num_conditions"])
    min_samples = int(cfg["min_samples_per_condition"])
    max_samples = int(cfg["max_samples_per_condition"])
    mode = str(cfg.get("condition_mode", "random")).lower().strip()
    if mode not in {"random", "uniform_mean", "uniform_log10", "uniform"}:
        raise ValueError(f"Unknown synthetic condition_mode '{mode}'.")

    uniform_log10_means = None
    uniform_sigma_log10 = None
    uniform_log10_max = 0.0
    if mode in {"uniform_mean", "uniform_log10", "uniform"}:
        log10_means_cfg = cfg.get("uniform_log10_means")
        if log10_means_cfg is None:
            raise ValueError("uniform_log10_means must be set when condition_mode is uniform.")
        if not isinstance(log10_means_cfg, (list, tuple)):
            raise ValueError("uniform_log10_means must be a list of log10 values.")
        if len(log10_means_cfg) != num_conditions:
            raise ValueError(
                "uniform_log10_means length must match num_conditions "
                f"({len(log10_means_cfg)} != {num_conditions})."
            )
        uniform_log10_means = np.array([float(v) for v in log10_means_cfg], dtype=np.float64)

        uniform_sigma_log10 = cfg.get("uniform_log10_std")
        if uniform_sigma_log10 is None:
            raise ValueError("uniform_log10_std must be set when condition_mode is uniform.")
        uniform_sigma_log10 = float(uniform_sigma_log10)

    ages_all, nw_all, conds_all, labels_all, probs_all = [], [], [], [], []
    clip_conditions = 0
    for cond in range(num_conditions):
        n = int(rng.integers(min_samples, max_samples + 1))
        if mode in {"uniform_mean", "uniform_log10", "uniform"}:
            mu_ln, sigma_ln, sig_mu, sig_s, mu_age, sigma_age, interest_scale = _sample_condition_params(cfg, rng)
            net_worth, ages, base_probs = generate_probs(
                n,
                mu_ln,
                sigma_ln,
                sig_mu,
                sig_s,
                mu_age,
                sigma_age,
                interest_scale=interest_scale,
                min_age=int(cfg["age_min"]),
                max_age=int(cfg["age_max"]),
                rng=rng,
            )
            mu_log10 = float(uniform_log10_means[cond])
            probs, clipped = _ranked_log10_normal_probs(
                base_probs,
                mu_log10=mu_log10,
                sigma_log10=uniform_sigma_log10,
                log10_min=None,
                log10_max=uniform_log10_max,
            )
            if clipped:
                clip_conditions += 1
        else:
            params = _sample_condition_params(cfg, rng)
            net_worth, ages, probs = generate_probs(
                n,
                *params,
                min_age=int(cfg["age_min"]),
                max_age=int(cfg["age_max"]),
                rng=rng,
            )
        labels = rng.binomial(n=1, p=probs).astype(np.float32)
        ages_all.append(ages)
        nw_all.append(net_worth)
        conds_all.append(np.full_like(ages, fill_value=cond, dtype=np.float32))
        labels_all.append(labels)
        probs_all.append(probs)

    ages = np.concatenate(ages_all, axis=0)
    net_worth = np.concatenate(nw_all, axis=0)
    conds = np.concatenate(conds_all, axis=0)
    labels = np.concatenate(labels_all, axis=0)
    probs = np.concatenate(probs_all, axis=0)

    if mode in {"uniform_mean", "uniform_log10", "uniform"} and clip_conditions > 0:
        print(
            "[WARN] Uniform log10 sampling clipped probabilities for "
            f"{clip_conditions}/{num_conditions} conditions. "
            "Consider adjusting uniform_log10_means or uniform_log10_std."
        )

    return {
        "ages": ages,
        "net_worth": net_worth,
        "conds": conds.astype(np.int64),
        "labels": labels,
        "probs": probs,
        "num_conditions": num_conditions,
        "rng": rng,
    }


def build_features(dataset: Dict[str, np.ndarray], cfg: Dict) -> Dict[str, np.ndarray]:
    """Build numeric feature matrix for synthetic data."""
    ages = dataset["ages"]
    net_worth = dataset["net_worth"]
    log10_nw = np.log10(np.clip(net_worth, 1e-12, None))
    base_features = {
        "age": ages.astype(np.float32),
        "net_worth": net_worth.astype(np.float32),
        "log10_net_worth": log10_nw.astype(np.float32),
    }

    feature_list = cfg.get("numeric_features")
    if feature_list is None:
        num_features = int(cfg.get("num_numeric_features", 3))
        if num_features < 1:
            raise ValueError("num_numeric_features must be >= 1 for synthetic data.")
        base_names = list(base_features.keys())
        base_count = min(num_features, len(base_names))
        feature_list = base_names[:base_count]
        extra_count = num_features - base_count
        feature_list += [f"noise_{i + 1}" for i in range(extra_count)]
    elif not isinstance(feature_list, (list, tuple)):
        raise ValueError("synthetic.numeric_features must be a list of feature names.")

    features = []
    feature_names = []
    mean = float(cfg["extra_feature_dist"]["mean"])
    std = float(cfg["extra_feature_dist"]["std"])
    rng = dataset.get("rng", np.random.default_rng())
    noise_idx = 1
    for name in feature_list:
        key = str(name).strip()
        key_lower = key.lower()
        if key_lower in base_features:
            features.append(base_features[key_lower])
            feature_names.append(key_lower)
            continue
        if key_lower == "noise" or key_lower.startswith("noise_"):
            noise = rng.normal(loc=mean, scale=std, size=(ages.shape[0],)).astype(np.float32)
            features.append(noise)
            if key_lower == "noise":
                feature_names.append(f"noise_{noise_idx}")
            else:
                feature_names.append(key_lower)
            noise_idx += 1
            continue
        raise ValueError(
            "Unsupported synthetic numeric feature "
            f"'{name}'. Allowed: age, net_worth, log10_net_worth, noise."
        )

    if not features:
        raise ValueError("synthetic.numeric_features produced no features.")
    xnum = np.stack(features, axis=1).astype(np.float32)

    return {
        "xnum": xnum,
        "feature_names": feature_names,
    }


def compute_negative_reweighting(
    labels: np.ndarray,
    conds: np.ndarray,
    cfg: Dict,
    *,
    probs: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Downsample negative samples per condition and return keep indices + weights."""
    reweight_cfg = cfg.get("reweighting", {}) if isinstance(cfg, dict) else {}
    enabled = bool(reweight_cfg.get("enabled", False))
    n_total = int(labels.shape[0])
    keep_all = np.arange(n_total, dtype=np.int64)
    if not enabled:
        weights = np.ones((n_total,), dtype=np.float32)
        return keep_all, weights, {"enabled": False}
    print(f"[WARN] {REWEIGHTING_DEPRECATION_MESSAGE}")
    warnings.warn(REWEIGHTING_DEPRECATION_MESSAGE, FutureWarning, stacklevel=2)

    if rng is None:
        rng = np.random.default_rng()

    mode = str(reweight_cfg.get("mode", "fixed")).lower().strip()
    if mode not in {"fixed", "adjustable"}:
        raise ValueError(f"Unknown synthetic reweighting mode '{mode}'.")

    base_n = float(reweight_cfg.get("negative_removal_n", 0.0))
    if base_n < 0:
        raise ValueError("synthetic.reweighting.negative_removal_n must be >= 0.")

    base_rate_floor = float(reweight_cfg.get("base_rate_log10_floor", 1e-6))
    base_rate_floor = max(base_rate_floor, 1e-12)

    labels_flat = np.asarray(labels, dtype=np.float32).reshape(-1)
    conds_flat = np.asarray(conds, dtype=np.int64).reshape(-1)
    num_conditions = int(conds_flat.max() + 1) if conds_flat.size else 0

    base_rates = None
    if mode == "adjustable":
        if probs is None:
            values = labels_flat
        else:
            values = np.asarray(probs, dtype=np.float32).reshape(-1)
        sums = np.bincount(conds_flat, weights=values, minlength=num_conditions)
        counts = np.bincount(conds_flat, minlength=num_conditions)
        base_rates = np.divide(
            sums,
            counts,
            out=np.zeros_like(sums, dtype=np.float64),
            where=counts > 0,
        )
        base_rates = np.clip(base_rates, base_rate_floor, 1.0)

    keep_idx = []
    weights = np.ones((n_total,), dtype=np.float32)
    removal_by_condition = np.zeros((num_conditions,), dtype=np.int64)
    kept_neg_by_condition = np.zeros((num_conditions,), dtype=np.int64)
    for cond_id in range(num_conditions):
        cond_mask = (conds_flat == cond_id)
        if not np.any(cond_mask):
            continue
        neg_idx = np.flatnonzero(cond_mask & (labels_flat <= 0.0))
        pos_idx = np.flatnonzero(cond_mask & (labels_flat > 0.0))

        removal_n = base_n
        if mode == "adjustable":
            base_rate = float(base_rates[cond_id]) if base_rates is not None else 0.0
            removal_n = base_n * abs(np.log10(max(base_rate, base_rate_floor)))
        removal_n = int(round(removal_n))
        removal_n = max(removal_n, 0)
        removal_by_condition[cond_id] = removal_n

        keep_neg = neg_idx
        if removal_n > 0 and neg_idx.size > 0:
            group_size = removal_n + 1
            keep_count = max(1, int(np.ceil(neg_idx.size / group_size)))
            if keep_count < neg_idx.size:
                keep_neg = rng.choice(neg_idx, size=keep_count, replace=False)
            weights[keep_neg] = float(removal_n)
        kept_neg_by_condition[cond_id] = int(keep_neg.size)
        if pos_idx.size > 0:
            keep_idx.append(pos_idx)
        if keep_neg.size > 0:
            keep_idx.append(keep_neg)

    if keep_idx:
        keep_idx = np.concatenate(keep_idx).astype(np.int64, copy=False)
        keep_idx = rng.permutation(keep_idx)
    else:
        keep_idx = np.array([], dtype=np.int64)

    return keep_idx, weights[keep_idx], {
        "enabled": True,
        "mode": mode,
        "negative_removal_n": base_n,
        "removal_by_condition": removal_by_condition,
        "kept_negatives_by_condition": kept_neg_by_condition,
    }
