# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A 10-week capstone research project implementing and evaluating **Localized Entropy (LE)**, a per-condition normalized cross-entropy loss for mixed-scale event probabilities (rare and common events in the same model). The core research question: does LE improve rare-event calibration and per-class gradient balance versus a BCE baseline? Everything is configuration-driven so experiments are reproducible.

Two long-form docs already exist and are authoritative — read them before making non-trivial changes:
- `README.md` — usage, config surface, data source setup (Avazu/Criteo/Yambda/synthetic).
- `ARCHITECTURE.md` — the full 13-step notebook pipeline and per-module behavior. This is unusually detailed; consult it rather than re-deriving pipeline flow from code.
- `invention.md` — the LE concept/rationale.

## Commands

```bash
pip install -r requirements.txt        # numpy, pandas, pyarrow, huggingface_hub, matplotlib, seaborn, scipy, torch, pytest

pytest localized_entropy/tests/                          # run all tests
pytest localized_entropy/tests/test_losses.py            # single file
pytest localized_entropy/tests/test_losses.py::test_name # single test

# Prepare real datasets ahead of notebook runs (also auto-prepared on demand):
python scripts/prepare_avazu_dataset.py  --config configs/default.json   # Kaggle (kagglehub) — needs ~/.kaggle/kaggle.json + rules accepted
python scripts/prepare_criteo_dataset.py --config configs/default.json   # Hugging Face
python scripts/prepare_yambda_dataset.py --config configs/default.json   # Hugging Face
```

There is no build step, linter config, or packaging (`setup.py`/`pyproject.toml`). `conftest.py` inserts the repo root on `sys.path`, so tests import `localized_entropy` without installation. The notebook `localized_entropy.ipynb` is the primary run surface — it delegates everything to the modules and is meant to stay thin.

## Architecture essentials

The notebook is orchestration only; all logic lives in `localized_entropy/`:

- `config.py` — loads/resolves `configs/default.json`. Config is deeply nested with layered overrides: an active `experiment` profile, `training.by_loss.<loss>.by_source.<source>` overrides (loss ∈ bce/localized_entropy/focal; source ∈ synthetic or the active CTR dataset key), and `ctr.defaults` merged with `ctr.datasets.<active>`. When editing behavior, prefer changing config over hardcoding.
- `data/` — `pipeline.py` branches on `data.source` (`ctr` vs `synthetic`). `ctr.py` loads/filters/encodes CTR CSVs; `synthetic.py` generates log-normal features; `yambda.py`/`criteo.py` auto-prepare CSVs from Hugging Face; `common.py` standardizes features (train-set mean/std). `datasets.py` builds loaders (`TensorBatchLoader` when staging tensors on GPU/MPS).
- `models.py` — `ConditionedLogitMLP`. Each sample is `(x_num, x_cat, c, y, w)`. Produces one logit from `concat(x_num, condition_embedding[c], categorical_embeddings...)`. **Only condition-embedding rows whose IDs appear in a batch get gradients that step**; the shared MLP gets gradients from all samples.
- `losses.py` — the research core. `localized_entropy()` normalizes each class's CE numerator by a constant-base-rate CE denominator: `LE = (Σ_j CE_j(y,ŷ) / CE_j(y,p_j)^α) / Σ N_j`. The denominator depends on labels only, so it is constant w.r.t. logits by design. Key knobs: `norm_strength` (α; 1.0=full LE, 0.0≈BCE), `CrossBatchHistory` (moving per-condition label window to stabilize denominators across batches), and `passive_weight`/`passive_mode` (Active-Passive-Loss RCE/BCE regularizer — currently disabled per recent commits). Also holds `focal_loss_with_logits` and a loop-based `binary_cross_entropy` (not used in the training loop).
- `training.py` — train/eval loops. Supports loss modes bce/localized_entropy/focal, plus `both`/`all`/comma-list to train several sequentially. Two optimizer param groups: base (`training.lr`, everything except `model.embedding`) and condition-embedding (`training.lr_category`); supports per-group decay and `lr_zero_after_epochs` (freeze base, keep embeddings training). For LE, per-condition base rates are precomputed once from training data and reused as fixed normalization.
- `analysis.py` — summary stats + metrics: ECE (`custom`/`adaptive`/`smooth`/`adaptive_lib`/`smooth_lib` via `evaluation.ece_method`), ROC/PR-AUC, accuracy/F1, per-condition calibration ratios and LE numerator/denominator diagnostics.
- `experiments.py` — helpers for building models and running single-loss repeated runs.
- `compare.py` — per-condition BCE-vs-LE comparison tables, repeated-run Wilcoxon summaries, grad-MSE comparison.
- `plotting.py` — all charts; `outputs.py` saves plots/logs under `output/{loss}/{dataset}/{nn_type}/{filter_mode}/`.

## Hyperparameter search

`notebooks/hyper_search/*.ipynb` sweep individual hyperparameters. Each has a top-level `LOSS_MODE` flag (`bce`/`le`/`fl`) in the first cell, resolves `REPO_ROOT` dynamically, loads `configs/hyper.json`, and routes through `localized_entropy.hyper_search.run_hyper_search` (reports `test_loss`).

## Conventions and gotchas

- **Numerical dtype**: when MPS is disabled the notebook uses float64 on CPU; float64 is unsupported on MPS, so gradient-accumulation diagnostics fall back to float32 there. Keep dtype-aware epsilons (see `focal_loss_with_logits`) when touching loss numerics.
- **Deprecated paths**: synthetic reweighting (`synthetic.reweighting`) and per-sample weights are DEPRECATED and emit warnings; unit weights are mathematically equivalent to the unweighted formulation. Don't build new features on them.
- **Datasets are not distributed** — they live under `data/<dataset>/` (gitignored). Each has an auto-prepare path (guarded by `auto_prepare`/`download_if_missing`): Criteo/Yambda download from Hugging Face; Avazu downloads the Kaggle competition via `kagglehub` (needs `~/.kaggle/kaggle.json` + accepted rules) and splits the labeled `train` file into train/test. Avazu can also be prepared manually (`gunzip -c NAME.gz > NAME.csv`). For Avazu, `C14` is the ad-id condition.
- Reproducibility relies on fixed NumPy/PyTorch seeds set in the notebook and deterministic hash splits in data prep; keep RNG usage deterministic.
- `README.md` still references a `contract.md` that no longer exists — trust the actual tree over such mentions.
