from typing import Dict, Optional, Tuple

import numpy as np
from scipy.stats import norm


def _sigmoid(x: np.ndarray, mu: float, s: float) -> np.ndarray:
    """Compute a shifted/scaled sigmoid."""
    return 1.0 / (1.0 + np.exp(-(x - mu) / s))


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
    denom = norm.pdf(50, loc=mu_age, scale=sigma_age)
    denom = denom if denom > 0 else 1.0
    interest_prob = norm.pdf(ages, loc=mu_age, scale=sigma_age) / denom
    interest_prob = interest_prob / interest_scale

    probs = probs * interest_prob
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

    ages_all, nw_all, conds_all, labels_all, probs_all = [], [], [], [], []
    for cond in range(num_conditions):
        n = int(rng.integers(min_samples, max_samples + 1))
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
