# Localized Entropy
## Capstone Faculty Research Project

This repository implements and evaluates Localized Entropy (LE), a normalized cross-entropy loss designed for mixed-scale event probabilities (rare and common events in the same model). This work is a 10-week capstone research project with a faculty advisor, focused on building a reproducible training pipeline, running controlled experiments, and documenting results clearly for review.

**Project goals**
- Implement Localized Entropy in PyTorch and compare it to BCE baselines.
- Evaluate rare-event performance, calibration, and per-class gradient balance.
- Keep the pipeline modular and configuration-driven for repeatable experiments.

## Repository layout
- `localized_entropy.ipynb` - Main notebook (uses modular Python files + config).
- `configs/default.json` - All run configuration (data source, model, training, plots).
- `localized_entropy/` - Reusable modules:
  - `config.py` (config loader)
  - `data/` (synthetic + CTR preprocessing, normalization, loaders)
  - `models.py` (network)
  - `losses.py` (LE + BCE helpers)
  - `training.py` (train/eval loops)
  - `analysis.py` (summaries + LE stats)
  - `plotting.py` (all charts/plots)
- `contract.md` - Capstone contract and evaluation criteria.

## Requirements
Python 3.10+ with:
- `numpy`, `pandas`, `matplotlib`, `scipy`, `torch`

## How to run
1. Open `localized_entropy.ipynb`.
2. Make sure `configs/default.json` points at your data files and the experiment you want.
3. Run all cells.

The notebook stays small and delegates everything to the modules in `localized_entropy/`.

## Config overview (`configs/default.json`)
- `experiment`
  - `active` selects an experiment profile.
  - `definitions` holds per-experiment overrides (e.g., `bce_baseline`, `small_net`).
- `data.source`: `"ctr"` or `"synthetic"`.
- `ctr` section: file paths, numeric feature columns, condition column, filtering rules, and preprocessing flags.
- `synthetic` section: number of conditions, sample counts, parameter ranges, and `numeric_features` (ordered feature list).
- `model`: hidden sizes, embedding dimension, activation/norm, dropout.
- `training`: epochs, learning rate, batch size, loss mode, and loss comparisons.
- `plots`: toggles for every chart and summary table.
- `evaluation`: bins/thresholds for BCE + calibration summaries (including small-probability calibration, aka ECE computed only on low-probability predictions).

**Notes**
- Normalization uses training-set mean/std for both real and synthetic data to keep scaling consistent.
- For Avazu CTR experiments, `C14` is treated as the ad identifier ("ad id") for conditioning and per-ad analysis.
- To change the number of input features:
  - For CTR, edit `ctr.numeric_cols`.
  - For synthetic, edit `synthetic.numeric_features` (append `noise` entries for Gaussian noise features).
- CTR filtering is applied in `localized_entropy/data/ctr.py` via `ctr.filter` (id lists or top/bottom-k by impressions or click rate).
- If `ctr.filter.cache.enabled` is true, the pipeline writes filtered CSVs to disk before loading to reduce memory use.
- CTR distribution plots use a sample size from `ctr.plot_sample_size` and toggles in `plots.ctr_data_distributions` and `plots.ctr_label_rates`. Set `plots.ctr_use_density` to `true` if you want density curves instead of counts.

## Data sources
- Real data currently uses the Avazu CTR dataset from Kaggle located in `data/`. The repo does not distribute the dataset.
  - The Avazu dataset comes as `.gz` files, convert them to `.csv` with `gunzip -c NAME.gz > NAME.csv`
  - It should be structured like this:
    - `data/train.csv` - training data with the following fields:
      - `id`: ad identifier
      - `click`: 0/1 for non-click/click
      - `hour`: format is YYMMDDHH, so 14091123 means 23:00 on Sept. 11, 2014 UTC.
      - `C1`: anonymized categorical variable
      - `banner_pos`
      - `site_id`
      - `site_domain`
      - `site_category`
      - `app_id`
      - `app_domain`
      - `app_category`
      - `device_id`
      - `device_ip`
      - `device_model`
      - `device_type`
      - `device_conn_type`
      - `C14-C21`: anonymized categorical variables
    - `data/test.csv` - test data with the same structure.
- Synthetic data is generated with log-normal net-worth and age features, plus optional extra numeric features.

## Outputs and evaluation
The notebook produces:
- Training/evaluation loss curves.
- Prediction distributions on a log10 scale.
- Per-condition LE numerator/denominator stats.
- Total BCE log loss, total ECE, and per-ad BCE/ECE (including low-probability calibration).
- Total ROC-AUC and PR-AUC for CTR baseline comparison.
- Optional CTR filter statistics plots.

These outputs support the capstone evaluation criteria in my contract (software quality, empirical improvement, and research merit).

## Reproducibility
- Fixed random seeds (NumPy + PyTorch) are set in the notebook.
- All experiment settings are in `configs/default.json`.
