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
- `avazu_top10_ads_by_count`: Sets `data.source=ctr` with `data.ctr_dataset=avazu`; keeps top 10 ads by impression count.
- `avazu_top50_ads_by_count`: Sets `data.source=ctr` with `data.ctr_dataset=avazu`; keeps top 50 ads by impression count.
- `avazu_top200_count_median_mix_30`: Sets `data.source=ctr` with `data.ctr_dataset=avazu`; from the top 200 ads by count,
  selects 10 high-mean, 10 median-band, and 10 low-mean ads.

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
- `training.lr`: Adam learning rate for the base optimizer group
  (all parameters except the condition embedding table).
- `training.lr_decay`: Multiplicative per-batch learning-rate decay
  factor applied after each optimizer step to the base group
  (`1.0` disables decay).
- `training.lr_category_decay`: Multiplicative per-batch decay factor
  for the condition-embedding optimizer group only (`1.0` disables decay;
  no-op if `training.lr_category` is unset).
- `training.lr_category`: Optional learning rate applied only to the
  condition embedding table (`model.embedding`). Accepts `LRCategory` as an
  alias in JSON configs.
- `training.lr_zero_after_epochs`: Optional epoch count after which the
  base learning rate group is set to `0`.
  - With `training.lr_category` set, this freezes non-condition parameters
    while allowing condition embeddings to continue updating.
  - Without `training.lr_category`, all trainable parameters are frozen.
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
- `training.by_loss.localized_entropy`: Localized Entropy-specific tuning
  options applied when the resolved loss mode is LE.
  - `training.by_loss.localized_entropy.by_source.<source>.cross_batch.enabled`:
    If true, maintain a moving-window history of labels per condition/label
    to stabilize LE denominators across batches for that source.
  - `training.by_loss.localized_entropy.by_source.<source>.cross_batch.amplification_rate`:
    Scalar used to compute per-condition window sizes as
    `N = amplification_rate / base_rate`. The window is a single FIFO
    queue per condition (total length capped to `N`). `amplification_factor`
    is accepted as a legacy alias.
- `training.debug_gradients`: If true, print raw per-parameter gradient
  tensors every batch. Recommended batch size of 3. WARNING: this is extremely performance intensive and will generate massive output!
- `training.debug_le_inputs`: If true, print per-batch LE debug summaries
  (inputs in the training loop plus tensor summaries inside
  `localized_entropy`). Defaults to `false`; can also be overridden under
  `training.by_loss.<loss>` and `training.by_loss.<loss>.by_source.<source>`.
- `training.print_embedding_table`: If true, print the full condition
  embedding table after each epoch.
- `training.by_loss`: Optional per-loss overrides keyed by `bce`,
  `localized_entropy` (`le` is accepted), or `focal`. Values in this block
  override the top-level training fields and apply independently per loss mode.
- `training.by_loss.<loss>.by_source`: Optional per-dataset overrides
  keyed by source key (`synthetic`, or CTR dataset key such as `avazu`,
  `criteo`, `yambda`; legacy `ctr` is also accepted). These apply after the
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
        "avazu": {
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
        "avazu": {
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
        "avazu": {
          "epochs": 4,
          "batch_size": 20000,
          "lr": 0.001,
          "cross_batch": {
            "enabled": false,
            "amplification_rate": 0.85
          }
        },
        "synthetic": {
          "epochs": 8,
          "batch_size": 25000,
          "lr": 0.006,
          "lr_category": null,
          "lr_zero_after_epochs": 2,
          "cross_batch": {
            "enabled": true,
            "amplification_rate": 1.35
          }
        }
      }
    }
  }
}
```

### model
- `model` input composition (`ConditionedLogitMLP`): for each sample, the model
  concatenates numeric features, one condition embedding lookup from
  `model.embedding` using the encoded condition ID, and optional categorical
  embedding lookups from `model.cat_embeddings`.
  - Forward pass:
    1. `cond` indexes `model.embedding` to fetch one condition vector.
    2. Each categorical column in `x_cat` indexes its corresponding table in
       `model.cat_embeddings` (if configured).
    3. The model concatenates `[x_num, cond_embedding, cat_embeddings...]`.
    4. The concatenated vector is passed through `model.net` and mapped to one
       scalar logit (probabilities are obtained later with sigmoid).
  - Input width to first linear layer:
    `num_numeric + embed_dim + (#categorical_columns * cat_embed_dim)`.
- `model.hidden_sizes`: MLP hidden layer sizes.
- `model.embed_dim`: Condition embedding dimension for
  `model.embedding` (condition ID lookup table).
- `model.cat_embed_dim`: Categorical embedding dimension
  for `model.cat_embeddings` (defaults to `embed_dim` if absent).
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
- `data.ctr_dataset`: Active CTR dataset key when `data.source=ctr`
  (`avazu`, `criteo`, or `yambda`).
- `data.train_split`: Fraction of samples used for training.
- `data.use_test_set`: If true, expose a test split/loader when available.
  - CTR: uses `ctr.datasets.<data.ctr_dataset>.test_path` as test data.
  - Synthetic: holds out `synthetic.test_split` from generated rows.
- `data.standardize`: If true, standardize numeric features using train
  mean/std.
- `data.standardize_eps`: Small std floor to avoid divide-by-zero.
- `data.shuffle_test`: If true, shuffle test arrays after load.

### ctr
These settings are used when `data.source` is `ctr`.

- `ctr.dataset`: Default CTR dataset key if `data.ctr_dataset` is unset.
- `ctr.warn_root_csv`: If true, print a warning when CSV files are found
  directly under `data/` instead of dataset subfolders.
- `ctr.data_root`: Root folder scanned for misplaced root-level CSVs.
- `ctr.defaults`: Shared defaults applied to all CTR datasets.
- `ctr.datasets.<name>`: Per-dataset overrides (for `avazu`, `criteo`,
  `yambda`) merged on top of `ctr.defaults`.
- `ctr.datasets.<name>.train_path` / `test_path`: Dataset-specific CSV paths.
- `ctr.datasets.<name>.numeric_cols`: Numeric feature columns.
- `ctr.datasets.<name>.categorical_cols`: Categorical feature columns.
- `ctr.datasets.<name>.categorical_max_values`: Cap per categorical vocab
  (top-k + other).
- `ctr.datasets.<name>.derived_time`: Derive `wd` and `wd_hour` from `hour`.
- `ctr.datasets.<name>.device_counters`: Add capped device_id/device_ip counts.
- `ctr.datasets.<name>.device_counter_cap`: Cap for device counters.
- `ctr.datasets.<name>.condition_col`: Condition column (for example ad id).
- `ctr.datasets.<name>.label_col`: Label column (binary click).
- `ctr.datasets.<name>.weight_col`: Optional weight column (currently unused).
- `ctr.datasets.<name>.max_conditions`: Cap condition vocab; others map to an `other` ID.
- `ctr.datasets.<name>.filter`: Optional filtering block (`ids`, `top_k`,
  `bottom_k`, `top_count_rate_mix`, or `none`), including optional cache paths.
- `ctr.datasets.<name>.filter_col` / `filter_top_k`: Legacy filtering keys (still supported).
- `ctr.datasets.<name>.drop_na`: Drop rows with NA in selected columns.
- `ctr.datasets.<name>.plot_filter_stats`: Show filter stats plot.
- `ctr.datasets.<name>.filter_test`: Legacy toggle for applying filters to test when
  `filter.apply_to_test` is unset.
- `ctr.datasets.<name>.test_has_labels`: If true, treat test CSV as labeled.
- `ctr.datasets.<name>.plot_sample_size`: Sample size for distribution plots.
- `ctr.datasets.<name>.balance_by_condition`: If true, downsample training
  to the minimum condition count.

Example: switch to Criteo and set dataset-specific filtering:

```json
"data": {
  "source": "ctr",
  "ctr_dataset": "criteo"
},
"ctr": {
  "datasets": {
    "criteo": {
      "condition_col": "C1",
      "filter": {
        "enabled": false,
        "mode": "none"
      }
    }
  }
}
```

Note: when `ctr.datasets.<name>.filter.cache.enabled=true`, the data
pipeline writes filtered CSVs before loading to reduce memory usage.

### synthetic
These settings are used when `data.source` is `synthetic`.

- `synthetic.num_conditions`: Number of synthetic conditions.
- `synthetic.min_samples_per_condition` / `synthetic.max_samples_per_condition`:
  Samples per condition.
- `synthetic.test_split`: Fraction of generated synthetic rows reserved
  for the test split when `data.use_test_set=true` (must be in `[0, 1)`).
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
  the bell curve used in uniform mode (default `0.4` in base config).
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
- `synthetic.reweighting`: Deprecated negative downsampling + weighting (training only).
  - Deprecated: when `synthetic.reweighting.enabled=true`, the pipeline prints a warning that
    this feature is deprecated and will be removed in a future release.
  - `synthetic.reweighting.enabled`: Enable deprecated negative downsampling/weights (default false).
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
- `evaluation.use_test_labels`: If true and test labels are available,
  use test labels for metrics (`ctr.datasets.<active>.test_has_labels=true`
  for CTR;
  synthetic test labels are available when `synthetic.test_split>0` and
  `data.use_test_set=true`).
- `evaluation.split`: `train`, `eval`, or `test` for evaluation.
- `evaluation.ece_bins`: Number of calibration bins.
- `evaluation.ece_min_count`: Minimum samples per ECE bin.
- `evaluation.ece_method`: ECE backend used for metrics and repeats:
  `custom` (fixed-width bins), `adaptive` (in-repo equal-mass bins),
  `smooth` (in-repo Gaussian-smoothed reliability), `adaptive_lib`
  (pandas `qcut` equal-mass bins), or `smooth_lib` (SciPy
  `gaussian_filter1d` smoothing).
- `evaluation.ece_smooth_bandwidth`: Bandwidth for `smooth` ECE.
- `evaluation.ece_smooth_grid_bins`: Histogram grid bins for `smooth` ECE.
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
  `evaluation.small_prob_max` / `evaluation.small_prob_quantile`, and use
  the configured `evaluation.ece_method` backend.

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
  (`base_rate`, `count`, `delta_calibration`, `abs_delta_calibration`,
  `delta_le_ratio`, `abs_delta_le_ratio`).
  Default is `base_rate` (descending).
- `comparison.print_table`: Print a tab-separated comparison table.
