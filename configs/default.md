# configs/default.json reference

This document explains each setting in `configs/default.json`, how it is
resolved, and how it affects the pipeline.

## How config resolution works

`localized_entropy/config.py` loads JSON, then:

1) Reads `experiment.active` to pick a definition from
   `experiment.definitions`.
2) Deep-merges the active definition into the base config.
3) Deep-merges `experiment.overrides` last.
4) Sets `experiment.name` to the active key for reporting.

This means you can keep defaults in the top-level config and override
small subsets per experiment.

Example:

```json
"experiment": {
  "active": "bce_baseline",
  "definitions": {
    "bce_baseline": {
      "training": {"loss_mode": "bce"}
    }
  }
}
```

Template model definitions included in `configs/default.json`:
- `default_net`: Mirrors the base `model` settings.
- `small_net`: Smaller MLP + smaller embeddings.
- `wide_net`: Wider MLP with fewer layers.
- `large_net`: Wider + deeper MLP with larger embeddings.

## Settings reference

### project
- `project.seed`: RNG seed used for numpy, torch, and dataset splitting.

### experiment
- `experiment.active`: Name of the active experiment definition.
- `experiment.definitions`: Map of experiment overrides to merge.
- `experiment.overrides`: Ad-hoc overrides applied after definitions.

### device
- `device.use_mps`: If true and MPS is available, use Apple Silicon GPU (with float32);
  set to false to force CPU (the notebook then builds models in float64).
- `device.move_dataset_to_cuda`: If true and CUDA is available, stage
  datasets on GPU and use `TensorBatchLoader`.
- `device.allow_dataloader_workers`: If true, allow multiprocessing
  DataLoader workers (false in notebooks by default).
- `device.num_workers_env`: Env var name used to override worker count.

### training
- `training.epochs`: Number of training epochs.
- `training.lr`: Adam learning rate.
- `training.lr_category`: Optional learning rate applied only to the
  condition embedding table. Accepts `LRCategory` as an alias
  in JSON configs.
- `training.lr_zero_after_epochs`: Optional epoch count after which the
  base learning rate is set to 0.
- `training.batch_size`: Train/eval batch size.
- `training.loss_mode`: `localized_entropy`, `bce`, `focal`, `both` (train
  BCE + LE sequentially), `all` (train BCE + LE + focal sequentially), or
  a comma/plus-delimited list (for example `bce,localized_entropy,focal`).
- `training.eval_every_n_batches`: If > 0, run mid-epoch eval callbacks
  at that interval and record train/eval loss by batch in the loss curve.
- `training.eval_compare_losses`: List of loss modes to evaluate after
  training for comparison (ignored when training multiple losses).
- `training.focal`: Focal loss configuration for `loss_mode=focal`.
  - `training.focal.alpha`: Positive-class weight (set `null` to disable
    alpha reweighting).
  - `training.focal.gamma`: Focusing parameter (higher = more focus on hard
    examples).
- `training.debug_gradients`: If true, print raw per-parameter gradient
  tensors every batch. Recommended batch size of 3. WARNING: this is extremely performance intensive and will generate massive output!
- `training.print_embedding_table`: If true, print the full condition
  embedding table after each epoch.
- `training.by_loss`: Optional per-loss overrides keyed by `bce`,
  `localized_entropy` (`le` is accepted), or `focal`. Values in this block
  override the top-level training fields and apply independently per loss mode.
- `training.by_loss.<loss>.by_source`: Optional per-dataset overrides
  keyed by `data.source` (`ctr` or `synthetic`). These apply after the
  per-loss overrides to select hyperparameters for a specific loss +
  data source combination.

Example:

```json
"training": {
  "loss_mode": "all",
  "focal": {"alpha": 0.25, "gamma": 2.0},
  "eval_compare_losses": ["localized_entropy", "bce", "focal"],
  "eval_every_n_batches": 200,
  "by_loss": {
    "bce": {
      "by_source": {
        "ctr": {
          "epochs": 2,
          "batch_size": 10000
        },
        "synthetic": {
          "epochs": 5,
          "batch_size": 50000
        }
      }
    },
    "focal": {
      "by_source": {
        "ctr": {
          "epochs": 2,
          "batch_size": 10000
        },
        "synthetic": {
          "epochs": 5,
          "batch_size": 50000
        }
      }
    },
    "localized_entropy": {
      "by_source": {
        "ctr": {
          "epochs": 4,
          "batch_size": 20000,
          "lr": 0.001
        },
        "synthetic": {
          "epochs": 8,
          "batch_size": 10000,
          "lr": 0.0005,
          "lr_category": 0.00025,
          "lr_zero_after_epochs": 2
        }
      }
    }
  }
}
```

### model
- `model.hidden_sizes`: MLP hidden layer sizes.
- `model.embed_dim`: Condition embedding dimension.
- `model.cat_embed_dim`: Categorical embedding dimension
  (defaults to `embed_dim` if absent).
- `model.activation`: Hidden-layer activation (`relu`, `gelu`, `silu`,
  `tanh`, `leaky_relu`, `elu`, `selu`, `sigmoid`, or `none`). Supports a
  list (length must match `hidden_sizes`) to set per-layer activations.
- `model.norm`: Normalization per hidden layer (`batch_norm` or
  `layer_norm`; `none`/`null` disables). The base config defaults to
  `null`. Supports a list (length must match `hidden_sizes`) to set
  per-layer norms.
- `model.dropout`: Dropout probability in the MLP. Can be a list (length
  must match `hidden_sizes`) to set per-layer dropouts.

### data
- `data.source`: `ctr` or `synthetic`.
- `data.train_split`: Fraction of samples used for training.
- `data.standardize`: If true, standardize numeric features using train
  mean/std.
- `data.standardize_eps`: Small std floor to avoid divide-by-zero.
- `data.shuffle_test`: If true, shuffle test arrays after load.

### ctr
These settings are used when `data.source` is `ctr`.

- `ctr.train_path` / `ctr.test_path`: CSV paths.
- `ctr.read_rows`: Max rows to read (null/0 = all rows; default null).
- `ctr.numeric_cols`: Numeric feature columns.
- `ctr.categorical_cols`: Categorical feature columns.
- `ctr.categorical_max_values`: Cap per categorical vocab (top-k + other).
- `ctr.derived_time`: Derive `wd` and `wd_hour` from `hour`.
- `ctr.device_counters`: Add capped device_id/device_ip counts.
- `ctr.device_counter_cap`: Cap for device counters.
- `ctr.condition_col`: Condition column (e.g., ad id).
- `ctr.label_col`: Label column (binary click).
- `ctr.weight_col`: Optional weight column (currently unused; reserved for future weighting).
- `ctr.max_conditions`: Cap condition vocab; others map to a single id.
- `ctr.filter`: Optional filtering block for selecting a subset of values.
  - `ctr.filter.enabled`: Enable filtering (default true if the block is set).
  - `ctr.filter.mode`: `ids`, `top_k`, `bottom_k`, or `none`.
  - `ctr.filter.col`: Column to filter (defaults to `condition_col`).
- `ctr.filter.ids`: List of values to keep (mode `ids`).
- `ctr.filter.k`: Number of values to keep (top/bottom-k modes).
- `ctr.filter.metric`: `count` (impressions) or `mean` (click rate).
- `ctr.filter.order`: Optional override for sort order (`asc`/`desc`).
- `ctr.filter.min_count`: Optional min count threshold before ranking.
- `ctr.filter.apply_to_test`: Apply the same filter to the test set.
- `ctr.filter.cache`: Optional on-disk cache for filtered CSVs.
  - `ctr.filter.cache.enabled`: If true, write filtered CSVs before load.
  - `ctr.filter.cache.train_path`: Output train CSV path.
  - `ctr.filter.cache.test_path`: Output test CSV path.
  - `ctr.filter.cache.overwrite`: If true, regenerate cached CSVs.
  - `ctr.filter.cache.chunksize`: Chunk size for streaming filter pass.
- `ctr.filter_col`: Legacy top-k filter column (still supported).
- `ctr.filter_top_k`: Legacy top-k values to keep (0/empty = disabled).
- `ctr.drop_na`: Drop rows with NA in selected columns.
- `ctr.plot_filter_stats`: If true, show filter stats plot.
- `ctr.filter_test`: Legacy toggle for applying filters to the test set
  when `ctr.filter.apply_to_test` is unset.
- `ctr.test_has_labels`: If true, treat test CSV as labeled.
- `ctr.plot_sample_size`: Sample size for distribution plots (0 = off).
- `ctr.balance_by_condition`: If true, downsample training to the
  minimum condition count.

Example: keep the top 50 ads by impressions and balance the training set:

```json
"ctr": {
  "filter": {
    "mode": "top_k",
    "col": "C14",
    "k": 50,
    "metric": "count"
  },
  "balance_by_condition": true
}
```

Example: keep the bottom 10 ads by click rate (mean) with at least 1,000
impressions:

```json
"ctr": {
  "filter": {
    "mode": "bottom_k",
    "col": "C14",
    "k": 10,
    "metric": "mean",
    "min_count": 1000
  }
}
```

Example: keep a fixed list of ad ids (mirrors the default config):

```json
"ctr": {
  "filter": {
    "mode": "ids",
    "col": "C14",
    "ids": [20093, 21768, 21191],
    "cache": {
      "enabled": true,
      "train_path": "data/train_filtered.csv",
      "test_path": "data/test_filtered.csv"
    }
  }
}
```

Note: when `ctr.filter.cache.enabled=true`, the data pipeline writes
filtered CSVs before loading to reduce memory usage.

### synthetic
These settings are used when `data.source` is `synthetic`.

- `synthetic.num_conditions`: Number of synthetic conditions.
- `synthetic.min_samples_per_condition` / `synthetic.max_samples_per_condition`:
  Samples per condition.
- `synthetic.numeric_features`: Ordered list of numeric features to emit.
  Supported values: `age`, `net_worth`, `log10_net_worth`, and `noise`
  (or `noise_1`, `noise_2`, ...). Any `noise*` entries draw from
  `synthetic.extra_feature_dist`.
- `synthetic.condition_mode`: `random` (default) or `uniform`/`uniform_log10`.
  - `random`: each condition samples its own shape parameters.
  - `uniform`/`uniform_log10`: generate per-condition probabilities from
    log10-normal bell curves while preserving feature ranks.
- `synthetic.uniform_log10_means`: List of log10 probability means (one
  per condition) used in uniform mode.
- `synthetic.uniform_log10_std`: Standard deviation in log10 space for
  the bell curve used in uniform mode.
  Both `uniform_log10_means` and `uniform_log10_std` are required when
  using uniform modes.
- `synthetic.use_true_base_rates_for_le`: If true, LE denominators use
  per-condition mean probabilities (from synthetic `probs`) instead of
  label-derived rates to avoid zero-positive collapse at very low means.
- `synthetic.base_mu_ln` / `synthetic.base_sigma_ln`: Lognormal base for
  net worth.
- `synthetic.sigmoid_mu_range` / `synthetic.sigmoid_s_range`: Sigmoid
  parameters for net-worth response (range + steps).
- `synthetic.age_mu_range` / `synthetic.age_sigma_range`: Age response
  parameters (range + steps).
- `synthetic.interest_scale_log10_range`: Log10 scale for interest curve.
  (used in `condition_mode=random`).
- To fix the synthetic shape across conditions, set the range triplets to
  `[value, value, 1]` so only one parameter value is sampled.
- `synthetic.age_min` / `synthetic.age_max`: Age bounds.
- `synthetic.extra_feature_dist`: Mean/std for extra noise features.
- `synthetic.reweighting`: Optional negative downsampling + weighting (training only).
  - `synthetic.reweighting.enabled`: Enable negative downsampling/weights (default false).
  - `synthetic.reweighting.mode`: `fixed` or `adjustable`.
    - `fixed`: remove `negative_removal_n` negatives per kept negative.
    - `adjustable`: remove `negative_removal_n * abs(log10(base_rate))` negatives per kept negative,
      where `base_rate` is the per-condition mean probability (from synthetic `probs`) clamped by
      `base_rate_log10_floor`.
  - `synthetic.reweighting.negative_removal_n`: Base N used for removal ratios (rounded to int).
  - `synthetic.reweighting.base_rate_log10_floor`: Minimum base rate used before `log10` to avoid
    `log10(0)` (default `1e-6`).
  - Positives are never dropped and always keep weight 1; kept negatives carry weight `N`.

Example: small synthetic dataset with 4 conditions and 3 features:

```json
"synthetic": {
  "num_conditions": 4,
  "min_samples_per_condition": 50000,
  "max_samples_per_condition": 50000,
  "numeric_features": ["age", "net_worth", "log10_net_worth"]
}
```

### plots
- `plots.data_before_training`: Plot data distributions before training.
- `plots.data_after_training`: Plot data distributions after training.
- `plots.eval_hist_epochs`: Plot log10(p) histogram each epoch.
- `plots.loss_curves`: Plot train/eval loss curves.
- `plots.eval_pred_hist`: Plot log10(p) histogram after training.
- `plots.eval_pred_by_condition`: Plot per-condition predictions.
- `plots.eval_calibration_ratio`: Plot calibration ratio vs condition
  base rate after training.
- `plots.eval_pred_value_range`: Log10(p) plot range as [min, max].
- `plots.le_stats`: Plot LE numerator/denominator stats per condition.
- `plots.grad_sq_by_condition`: Track and plot per-condition mean-squared
  gradient (MSE) for BCE vs LE during training.
- `plots.grad_sq_top_k`: If > 0, plot only the top-k conditions by
  gradient MSE (0 = all).
- `plots.grad_sq_log10`: If true, plot log10 of gradient MSE values.
- `plots.print_eval_summary`: Print eval prediction summary.
- `plots.print_le_stats_table`: Print LE stats table.
- `plots.ctr_data_distributions`: Enable CTR feature plots.
- `plots.ctr_label_rates`: Plot per-condition label rates (CTR).
- `plots.ctr_max_features`: Max numeric features to plot.
- `plots.ctr_log10_features`: Numeric feature names to log10-transform.
- `plots.ctr_use_density`: Use density for CTR feature plots.

### logging
- `logging.print_loader_note`: Print DataLoader configuration summary.

### evaluation
- `evaluation.use_test_labels`: If true and `ctr.test_has_labels`, use
  test labels for metrics.
- `evaluation.split`: `train`, `eval`, or `test` for evaluation.
- `evaluation.ece_bins`: Number of calibration bins.
- `evaluation.ece_min_count`: Minimum samples per ECE bin.
- `evaluation.small_prob_max`: Threshold for "small p" calibration.
- `evaluation.small_prob_quantile`: Quantile fallback if no preds are
  below `small_prob_max`.
- `evaluation.per_ad_top_k`: Top-k conditions to print for metrics.
- `evaluation.print_per_ad`: Print per-condition metrics table.
- `evaluation.print_calibration_table`: Print full ECE bin table.

Example: evaluate on test labels when available:

```json
"evaluation": {
  "split": "test",
  "use_test_labels": true,
  "per_ad_top_k": 20
}
```

### repeats
Optional repeated-run significance testing when training BCE + LE.

- `repeats.enabled`: If true, run repeated BCE vs LE training runs.
- `repeats.num_runs`: Number of paired runs per loss.
- `repeats.seed_stride`: Seed increment per run (base = `project.seed`).
- `repeats.include_base_run`: If true, include the initial run results.
- `repeats.wilcoxon_zero_method`: `zero_method` argument for
  `scipy.stats.wilcoxon`.
- `repeats.wilcoxon_alternative`: `alternative` hypothesis for
  `scipy.stats.wilcoxon`.
- `repeats.per_condition_top_k`: Rows to print for per-condition
  calibration tests.
- `repeats.per_condition_sort_by`: Sort key for per-condition calibration
  tests (`p_value`, `delta_mean`, `abs_delta_mean`, `count`, `base_rate`).
- `repeats.per_condition_min_count`: Minimum condition sample count to
  include in per-condition calibration tests.
- Repeated-run summaries include `ece_small` using
  `evaluation.small_prob_max` / `evaluation.small_prob_quantile`.

Example: run 10 paired repeats and report Wilcoxon p-values:

```json
"repeats": {
  "enabled": true,
  "num_runs": 10,
  "seed_stride": 1
}
```

### comparison
Used when BCE + LE are both trained to compare per-condition performance
calibration and LE ratio terms.

- `comparison.enabled`: If true, run the per-condition comparison.
- `comparison.top_k`: Number of rows to include in the table plot.
- `comparison.sort_by`: Sort key for the comparison table
  (`count`, `delta_calibration`, `abs_delta_calibration`,
  `delta_le_ratio`, `abs_delta_le_ratio`).
- `comparison.print_table`: Print a tab-separated comparison table.
