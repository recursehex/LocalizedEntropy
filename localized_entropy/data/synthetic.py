from typing import Dict, Optional, Tuple

import numpy as np
from scipy.stats import norm


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


def _rescale_probs_to_mean_with_bounds(
    probs: np.ndarray,
    target_mean: float,
    prob_min: float,
    prob_max: float,
    tol: float = 1e-8,
    max_iter: int = 60,
) -> Tuple[np.ndarray, float, bool]:
    """Scale probabilities to hit a target mean while staying within bounds."""
    if not (0.0 < target_mean <= 1.0):
        raise ValueError("target_mean must be in (0, 1].")
    if not (0.0 <= prob_min < prob_max <= 1.0):
        raise ValueError("prob_min/prob_max must satisfy 0 <= min < max <= 1.")
    current_mean = float(np.mean(probs))
    if current_mean <= 0:
        clipped = prob_min > 0
        return np.full_like(probs, prob_min, dtype=np.float32), 0.0, clipped

    target = float(np.clip(target_mean, prob_min, prob_max))

    def mean_for(scale: float) -> float:
        return float(np.mean(np.clip(probs * scale, prob_min, prob_max)))

    scale_low = 0.0
    scale_high = 1.0
    mean_high = mean_for(scale_high)
    while mean_high < target and scale_high < 1e12:
        scale_high *= 2.0
        mean_high = mean_for(scale_high)

    if mean_high < target:
        scaled = np.clip(probs * scale_high, prob_min, prob_max)
        return scaled.astype(np.float32, copy=False), scale_high, True

    scale = scale_high
    for _ in range(max_iter):
        scale_mid = 0.5 * (scale_low + scale_high)
        mean_mid = mean_for(scale_mid)
        scale = scale_mid
        if abs(mean_mid - target) <= tol:
            break
        if mean_mid < target:
            scale_low = scale_mid
        else:
            scale_high = scale_mid

    scaled = np.clip(probs * scale, prob_min, prob_max)
    clipped = bool(np.any((probs * scale) < prob_min) or np.any((probs * scale) > prob_max))
    return scaled.astype(np.float32, copy=False), scale, clipped


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
    uniform_shape_mode = None
    uniform_band_width = None
    uniform_log10_min = None
    uniform_log10_max = 0.0
    if mode in {"uniform_mean", "uniform_log10", "uniform"}:
        log10_means_cfg = cfg.get("uniform_log10_means")
        if log10_means_cfg is not None:
            if not isinstance(log10_means_cfg, (list, tuple)):
                raise ValueError("uniform_log10_means must be a list of log10 values.")
            if len(log10_means_cfg) != num_conditions:
                raise ValueError(
                    "uniform_log10_means length must match num_conditions "
                    f"({len(log10_means_cfg)} != {num_conditions})."
                )
            uniform_log10_means = np.array([float(v) for v in log10_means_cfg], dtype=np.float64)
        else:
            mean_range = cfg.get("uniform_mean_log10_range", [-1.0, -10.0])
            if not isinstance(mean_range, (list, tuple)) or len(mean_range) != 2:
                raise ValueError("uniform_mean_log10_range must be a 2-item list.")
            start, end = float(mean_range[0]), float(mean_range[1])
            uniform_log10_means = np.linspace(start, end, num_conditions)

        uniform_sigma_log10 = cfg.get("uniform_log10_std")
        if uniform_sigma_log10 is None:
            if log10_means_cfg is not None:
                raise ValueError("uniform_log10_std must be set when using uniform_log10_means.")
            sigma_fraction = float(cfg.get("uniform_log10_sigma_fraction", 0.3))
            if num_conditions > 1:
                span = float(np.max(uniform_log10_means) - np.min(uniform_log10_means))
                spacing = span / (num_conditions - 1)
            else:
                spacing = 1.0
            uniform_sigma_log10 = spacing * max(sigma_fraction, 0.0)
        else:
            uniform_sigma_log10 = float(uniform_sigma_log10)

        uniform_shape_mode = str(cfg.get("uniform_log10_shape", "rank_normal")).lower().strip()
        band_fraction = cfg.get("uniform_log10_band_fraction", 0.0)
        band_fraction = float(band_fraction) if band_fraction is not None else 0.0
        if num_conditions > 1:
            span = float(np.max(uniform_log10_means) - np.min(uniform_log10_means))
            spacing = span / (num_conditions - 1)
        else:
            spacing = 1.0
        if band_fraction >= 1.0:
            print("[WARN] uniform_log10_band_fraction >= 1; bands will overlap.")
        band_fraction = max(band_fraction, 0.0)
        uniform_band_width = spacing * band_fraction if band_fraction > 0 else None

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
            if uniform_shape_mode in {"rank_normal", "rank_log10", "bell", "gaussian", "normal"}:
                probs, clipped = _ranked_log10_normal_probs(
                    base_probs,
                    mu_log10=mu_log10,
                    sigma_log10=uniform_sigma_log10,
                    log10_min=uniform_log10_min,
                    log10_max=uniform_log10_max,
                )
            else:
                # Legacy fallback: rescale base probs to a target mean in probability space.
                target_mean = float(10.0 ** mu_log10)
                if uniform_band_width is not None:
                    band_lo_log = mu_log10 - 0.5 * uniform_band_width
                    band_hi_log = mu_log10 + 0.5 * uniform_band_width
                    prob_min = float(10.0 ** band_lo_log)
                    prob_max = float(10.0 ** band_hi_log)
                else:
                    prob_min = 0.0
                    prob_max = 1.0
                probs, _, clipped = _rescale_probs_to_mean_with_bounds(
                    base_probs,
                    target_mean,
                    prob_min,
                    prob_max,
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

    if mode == "uniform_mean" and clip_conditions > 0:
        print(
            "[WARN] uniform_mean scaling clipped probabilities for "
            f"{clip_conditions}/{num_conditions} conditions. "
            "Consider lowering uniform_mean_log10_range or adjusting base shape parameters."
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
