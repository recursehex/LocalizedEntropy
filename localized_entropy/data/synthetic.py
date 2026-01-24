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
    if mode not in {"random", "uniform_mean"}:
        raise ValueError(f"Unknown synthetic condition_mode '{mode}'.")

    uniform_targets = None
    shared_params = None
    if mode == "uniform_mean":
        mean_range = cfg.get("uniform_mean_log10_range", [-1.0, -10.0])
        if not isinstance(mean_range, (list, tuple)) or len(mean_range) != 2:
            raise ValueError("uniform_mean_log10_range must be a 2-item list.")
        start, end = float(mean_range[0]), float(mean_range[1])
        log10_means = np.linspace(start, end, num_conditions)
        uniform_targets = 10.0 ** log10_means
        band_fraction = cfg.get("uniform_log10_band_fraction", 0.6)
        band_fraction = float(band_fraction) if band_fraction is not None else 0.0
        shape_mode = str(cfg.get("uniform_log10_shape", "scaled")).lower().strip()
        sigma_fraction = float(cfg.get("uniform_log10_sigma_fraction", 0.3))
        span = abs(start - end)
        spacing = span / (num_conditions - 1) if num_conditions > 1 else 1.0
        if band_fraction >= 1.0:
            print("[WARN] uniform_log10_band_fraction >= 1; bands will overlap.")
        band_fraction = max(band_fraction, 0.0)
        band_width = spacing * band_fraction if band_fraction > 0 else None
        mu_ln, sigma_ln, sig_mu, sig_s, mu_age, sigma_age, _ = _sample_condition_params(cfg, rng)
        shared_params = (mu_ln, sigma_ln, sig_mu, sig_s, mu_age, sigma_age)

    ages_all, nw_all, conds_all, labels_all, probs_all = [], [], [], [], []
    clip_conditions = 0
    for cond in range(num_conditions):
        n = int(rng.integers(min_samples, max_samples + 1))
        if mode == "uniform_mean":
            mu_ln, sigma_ln, sig_mu, sig_s, mu_age, sigma_age = shared_params
            net_worth, ages, base_probs = generate_probs(
                n,
                mu_ln,
                sigma_ln,
                sig_mu,
                sig_s,
                mu_age,
                sigma_age,
                interest_scale=1.0,
                min_age=int(cfg["age_min"]),
                max_age=int(cfg["age_max"]),
                rng=rng,
            )
            target_mean = float(uniform_targets[cond])
            if band_width is not None:
                band_lo_log = log10_means[cond] - 0.5 * band_width
                band_hi_log = log10_means[cond] + 0.5 * band_width
                prob_min = float(10.0 ** band_lo_log)
                prob_max = float(10.0 ** band_hi_log)
            else:
                band_lo_log = None
                band_hi_log = None
                prob_min = 0.0
                prob_max = 1.0
            if shape_mode in {"normal", "gaussian"}:
                sigma_log10 = spacing * max(sigma_fraction, 0.0)
                if band_width is None:
                    mu_log10 = _solve_mu_log10_for_target(target_mean, sigma_log10)
                    log10p = rng.normal(loc=mu_log10, scale=sigma_log10, size=n).astype(
                        np.float32, copy=False
                    )
                    probs = np.power(10.0, log10p).astype(np.float32, copy=False)
                    clipped = bool(np.any(probs > 1.0))
                    if clipped:
                        probs = np.clip(probs, 0.0, 1.0)
                else:
                    mu_log10 = _solve_mu_log10_for_target_centered(
                        target_mean,
                        band_width,
                        sigma_log10,
                    )
                    band_lo_log = mu_log10 - 0.5 * band_width
                    band_hi_log = mu_log10 + 0.5 * band_width
                    log10p, clipped = _sample_truncated_log10_normal(
                        rng,
                        n,
                        mu_log10,
                        sigma_log10,
                        band_lo_log,
                        band_hi_log,
                    )
                    probs = (10.0 ** log10p).astype(np.float32, copy=False)
            elif shape_mode == "rank_normal":
                sigma_log10 = spacing * max(sigma_fraction, 0.0)
                if band_lo_log is None or band_hi_log is None:
                    band_lo_log = log10_means[cond] - 0.5 * spacing
                    band_hi_log = log10_means[cond] + 0.5 * spacing
                probs, clipped = _ranked_normal_probs(
                    base_probs,
                    target_mean,
                    band_lo_log,
                    band_hi_log,
                    sigma_log10,
                )
            else:
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
    base_features = np.stack([ages, log10_nw], axis=1).astype(np.float32)

    num_features = int(cfg["num_numeric_features"])
    if num_features < 2:
        raise ValueError("num_numeric_features must be >= 2 for synthetic data.")

    feature_names = ["age", "log10_net_worth"]
    if num_features > 2:
        extra_count = num_features - 2
        mean = float(cfg["extra_feature_dist"]["mean"])
        std = float(cfg["extra_feature_dist"]["std"])
        rng = dataset.get("rng", np.random.default_rng())
        extra = rng.normal(loc=mean, scale=std, size=(base_features.shape[0], extra_count)).astype(np.float32)
        feature_names += [f"noise_{i + 1}" for i in range(extra_count)]
        xnum = np.concatenate([base_features, extra], axis=1)
    else:
        xnum = base_features

    return {
        "xnum": xnum,
        "feature_names": feature_names,
    }
