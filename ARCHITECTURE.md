# Localized Entropy Architecture Overview

This document describes the end-to-end pipeline and modules used by the
`localized_entropy.ipynb` notebook, plus the supporting Python scripts.

## Project goal

Train a per-condition click-through-rate model and compare the Localized
Entropy (LE) loss against standard BCE, with diagnostics and per-condition
calibration analysis.

## Main execution flow (notebook)

Notebook entry point: `localized_entropy.ipynb`.

Step-by-step pipeline:

1) Load config + set seed and device
- Uses `localized_entropy/config.py` to load and resolve
  `configs/default.json`.
- Per-loss `training.by_loss` overrides are resolved when each loss is
  trained (BCE vs LE), including `by_source` settings nested under each
  loss for the active data source.
- Chooses CUDA vs CPU via `localized_entropy/utils.py`.

2) Optional CTR filtering + caching (config-driven)
- `localized_entropy/data/ctr.py` applies `ctr.filter` (or legacy
  `ctr.filter_col`/`ctr.filter_top_k`) to select a subset of conditions
  by id list or by top/bottom-k ranking on impressions or click rate.
- If `ctr.filter.cache.enabled` is true, the pipeline writes filtered
  CSVs to disk before loading, reducing memory pressure for large
  datasets.
- Filter stats are computed from the filtered training data and passed
  through `plot_ctr_filter_stats` when enabled.

3) Data preparation
- `localized_entropy/data/pipeline.py` drives the data pipeline.
- It branches on `data.source`:
  - CTR: `localized_entropy/data/ctr.py` loads CSVs, adds derived
    features (time, device counters), encodes conditions and
    categoricals, and builds numeric/categorical arrays.
  - Synthetic: `localized_entropy/data/synthetic.py` generates
    synthetic net-worth and age distributions per condition and
    converts to numeric features.
- Training/eval split uses `train_split` with a deterministic RNG.
- Optional per-condition balancing occurs when
  `ctr.balance_by_condition=true`.
- Optional feature standardization happens via
  `localized_entropy/data/common.py`.
- Dataloaders:
  - If CUDA is available and `device.move_dataset_to_cuda=true`,
    `TensorBatchLoader` is used to stage tensors on GPU.
  - Otherwise, standard PyTorch `DataLoader` is used with a
    worker fallback strategy.

4) Diagnostics before training
- Feature stats, condition stats, and label stats are printed via
  `localized_entropy/analysis.py`.
- If plots are enabled, the notebook produces pre-training
  distribution plots from sampled data, skipping device count features
  (`device_ip_count`, `device_id_count`) for CTR runs.

5) Model construction
- `localized_entropy/models.py` defines `ConditionProbNet`:
  - Condition embedding.
  - Optional categorical embeddings.
  - Feed-forward MLP to a single logit, with configurable hidden sizes,
    activation, normalization, and dropout (per-layer settings supported
    via `configs/default.json`).

6) Initial prediction diagnostics (untrained)
- One batch is passed through the model to report logit and
  probability ranges.

7) Pre-training per-condition predictions
- If enabled, the notebook runs `predict_probs` and renders
  per-condition prediction histograms for the untrained model.

8) Training loop
- `localized_entropy/training.py` implements training and eval.
- Supports two loss modes:
  - Localized Entropy (`localized_entropy/losses.py`).
  - BCE (`torch.nn.BCEWithLogitsLoss`).
- When configured, BCE/LE use per-loss training overrides for
  `epochs`, `lr`, and `batch_size`, rebuilding dataloaders per loss to
  honor different batch sizes.
- For LE, per-condition base rates are computed once from the training data
  and reused as fixed normalization factors during training (streaming
  base rates are only used as a fallback when precomputed rates are absent).
- Optional mid-epoch eval callbacks can plot prediction histograms.
- When `training.eval_every_n_batches > 0`, train/eval loss is tracked by
  batch for additional diagnostics in the loss curve plot.
- Loss curves include an initial epoch/batch 0 train/eval loss before updates.
- When `plots.grad_sq_by_condition=true`, the loop also accumulates
  per-condition mean-squared logits gradients for BCE vs LE, plus
  per-class (label 0/1) gradient MSE to report a class MSE ratio.
  The notebook prints a per-condition LE/BCE gradient MSE ratio table.
- The notebook can enable raw per-parameter gradient debug prints per
  batch via the `debug_gradients` training flag (WARNING: extremely performance
  intensive!).
- When `debug_le_inputs=true`, the training loop prints per-batch input
  feature summaries (x, x_cat, conditions, targets) and forwards a debug
  flag into `localized_entropy` to summarize logits/targets/conditions and
  optional base-rate/weight tensors.
- If `training.print_embedding_table=true`, the loop prints the full
  condition embedding table (`model.embedding.weight`) after each epoch.
- If `training.loss_mode` is set to `both`, the notebook trains BCE and
  LE sequentially and stores results for comparison.

9) Post-training evaluation
- Evaluates on the configured split (`evaluation.split`).
- Computes summary stats, histograms, and per-condition prediction
  plots.
- If labels are available, computes:
  - BCE log loss, ECE, ROC-AUC, PR-AUC.
  - Accuracy/F1 at 0.5 (global) plus per-condition accuracy/F1.
  - Per-condition ECE and base rates.
- Optionally compares loss values under alternate loss modes
  (`training.eval_compare_losses`).
- If both losses are trained, the notebook builds a per-condition
  comparison table (calibration ratio + LE ratio deltas) using
  `localized_entropy/compare.py`, printing an aligned text table plus a
  BCE-vs-LE summary (accuracy, logloss, brier, ECE) and a per-condition
  abs(1 - calibration) table for quick closeness-to-1 checks.

10) Optional repeated-run significance testing
- When `repeats.enabled=true`, the notebook re-trains BCE/LE across
  multiple seeds and computes paired Wilcoxon signed-rank tests over
  logloss/brier/ece/ece_small/accuracy deltas to report p-values, with
  `ece_small` focused on low-probability predictions.
- If evaluation conditions are available, it also computes per-condition
  calibration Wilcoxon tests on absolute gaps between per-condition
  prediction means and base rates.

11) Per-condition train vs eval rate diagnostics
- `summarize_per_ad_train_eval_rates` compares per-condition
  train click rates vs eval mean prediction.

12) Optional test-set inference
- If a test loader exists and `evaluation.split != test`,
  the notebook runs predictions on the test set and plots
  the same diagnostics (without label-based metrics).

13) Post-training data plots and LE stats
- Optionally re-plot data distributions (CTR plots skip device count
  features).
- If labels are available, collects per-condition LE numerator/
  denominator stats and plots them.

## Data pipeline details

CTR source (`localized_entropy/data/ctr.py`):
- Reads CSV columns from config (`ctr.*`).
- Optional on-disk cache:
  - When `ctr.filter.cache.enabled=true`, filtered train/test CSVs are
    generated in streaming chunks and paths are updated for loading.
- Optional preprocessing:
  - `derived_time`: extract day-of-week and hour features.
  - `device_counters`: add capped counts for device_ip/device_id.
- Optional filtering:
  - Uses `ctr.filter` (or legacy `filter_col`/`filter_top_k`) to keep
    specific condition ids or top/bottom-k by impressions/click rate.
  - Optionally applies the same filter to the test set.
- Encodes:
  - Condition column into integer IDs (with optional top-k mapping).
  - Categorical features with per-column vocab caps.
- Produces:
  - Numeric feature matrix `xnum`.
  - Categorical feature matrix `xcat`.
  - Labels, conditions, and optional test labels.
  - Note: `ctr.weight_col` is currently ignored; net-worth arrays are not
    included in the training loaders.
  - `num_conditions` set from the max encoded condition across train/test
    (unused "other" buckets are trimmed).

Synthetic source (`localized_entropy/data/synthetic.py`):
- Generates net-worth and age distributions per condition.
- Produces probability targets from a sigmoid + interest curve.
- Builds numeric features based on `synthetic.numeric_features`
  (supported values: `age`, `net_worth`, `log10_net_worth`, and
  `noise*`).
- Supports `synthetic.condition_mode=uniform_mean` to keep a shared
  shape across conditions and rescale to per-condition mean
  probabilities (log10-uniform range), with optional non-overlapping
  or overlapping log10 bands via `synthetic.uniform_log10_band_fraction`.
- Optional `synthetic.uniform_log10_shape=normal` samples a truncated
  log10-normal bell curve per condition (band centered on the solved
  mean); `rank_normal` maps probability ranks to a bell-shaped log10
  distribution within each band.
- When `synthetic.use_true_base_rates_for_le=true`, LE uses the true
  per-condition mean probabilities (from synthetic `probs`) instead of
  label-derived rates to avoid zero-positive collapse.

## Losses and metrics

- `localized_entropy/losses.py`:
  - `localized_entropy`: novel per-condition normalized BCE.
  - `binary_cross_entropy`: custom loop-based BCE (currently not used in
    the notebook training loop).
- `localized_entropy/analysis.py`:
  - Summary stats (label, condition, prediction).
  - ECE, ROC-AUC, PR-AUC.
  - Binary accuracy/F1 metrics and per-condition diagnostics.
  - Per-condition metrics and LE numerator/denominator diagnostics.
  - Per-condition calibration ratios (pred mean / label mean) and helpers
    for LE ratio tables.
- `localized_entropy/experiments.py`: experiment helpers for building
  models, resolving eval splits, and training single-loss runs.
- `localized_entropy/compare.py`: per-condition BCE vs LE comparison
  table builder and plot/save utilities.

## Plotting and outputs

- `localized_entropy/plotting.py` renders:
  - Feature distributions by condition.
  - Prediction histograms (log10 p).
  - Loss curves.
  - Eval loss by batch (when mid-epoch eval is enabled).
  - Per-condition LE diagnostics.
  - Table plots for BCE vs LE per-condition comparisons.
- Outputs are shown inline in the notebook, and `localized_entropy.ipynb`
  now saves key plots plus text logs under `output/` via
  `localized_entropy/outputs.py`.
- Output folder structure: `output/{bce|le}/{ctr|synthetic}/{nn_type}/{filter_mode}/`
  with `avg.png`, `loss.png`, `preds.png`, `calibration.png`, and
  `notebook_output.txt`.

## Key entry points

- `localized_entropy.ipynb`: end-to-end execution.
- `configs/default.json`: experiment configuration.
- `localized_entropy/`: reusable pipeline, model, training, and analysis.
- `ad_id_compare_bce_le.py`: trains BCE/LE models and writes per-condition
  comparison tables (calibration + LE ratio deltas) to `results/`.
