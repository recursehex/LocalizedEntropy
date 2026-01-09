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
- Chooses CUDA vs CPU via `localized_entropy/utils.py`.

2) Optional CTR top-k filtering (notebook-only)
- If `ctr.filter_col` and `ctr.filter_top_k` are set, the notebook
  precomputes top-k values, writes filtered CSVs under `data/`, and
  updates `cfg['ctr']['train_path']`/`cfg['ctr']['test_path']`.
- It also writes per-value stats under `results/` for inspection.
- This is separate from the `localized_entropy/data/ctr.py` filtering
  and is designed to cache a smaller dataset on disk.

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
  distribution plots from sampled data.

5) Model construction
- `localized_entropy/models.py` defines `ConditionProbNet`:
  - Condition embedding.
  - Optional categorical embeddings.
  - Feed-forward MLP to a single logit.

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
- For LE, a streaming per-condition base rate is updated each batch.
- Optional mid-epoch eval callbacks can plot prediction histograms.

9) Post-training evaluation
- Evaluates on the configured split (`evaluation.split`).
- Computes summary stats, histograms, and per-condition prediction
  plots.
- If labels are available, computes:
  - BCE log loss, ECE, ROC-AUC, PR-AUC.
  - Per-condition ECE and base rates.
- Optionally compares loss values under alternate loss modes
  (`training.eval_compare_losses`).

10) Per-condition train vs eval rate diagnostics
- `summarize_per_ad_train_eval_rates` compares per-condition
  train click rates vs eval mean prediction.

11) Optional test-set inference
- If a test loader exists and `evaluation.split != test`,
  the notebook runs predictions on the test set and plots
  the same diagnostics (without label-based metrics).

12) Post-training data plots and LE stats
- Optionally re-plot data distributions.
- If labels are available, collects per-condition LE numerator/
  denominator stats and plots them.

## Data pipeline details

CTR source (`localized_entropy/data/ctr.py`):
- Reads CSV columns from config (`ctr.*`).
- Optional preprocessing:
  - `derived_time`: extract day-of-week and hour features.
  - `device_counters`: add capped counts for device_ip/device_id.
- Encodes:
  - Condition column into integer IDs (with optional top-k mapping).
  - Categorical features with per-column vocab caps.
- Produces:
  - Numeric feature matrix `xnum`.
  - Categorical feature matrix `xcat`.
  - Labels, conditions, and optional test labels.

Synthetic source (`localized_entropy/data/synthetic.py`):
- Generates net-worth and age distributions per condition.
- Produces probability targets from a sigmoid + interest curve.
- Adds extra noise features if configured.

## Losses and metrics

- `localized_entropy/losses.py`:
  - `localized_entropy`: novel per-condition normalized BCE.
  - `binary_cross_entropy`: custom loop-based BCE (currently not used in
    the notebook training loop).
- `localized_entropy/analysis.py`:
  - Summary stats (label, condition, prediction).
  - ECE, ROC-AUC, PR-AUC.
  - Per-condition metrics and LE numerator/denominator diagnostics.

## Plotting and outputs

- `localized_entropy/plotting.py` renders:
  - Feature distributions by condition.
  - Prediction histograms (log10 p).
  - Loss curves.
  - Per-condition LE diagnostics.
- Outputs are shown inline in the notebook and optionally saved by
  ad-hoc scripts under `results/`.

## Key entry points

- `localized_entropy.ipynb`: end-to-end execution.
- `configs/default.json`: experiment configuration.
- `localized_entropy/`: reusable pipeline, model, training, and analysis.
