from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from localized_entropy.config import load_and_resolve
from localized_entropy.data.yambda import maybe_prepare_yambda_dataset


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for Yambda preparation."""
    parser = argparse.ArgumentParser(
        description=(
            "Prepare yandex/yambda flat/50m multi_event parquet into "
            "train/test CSV files for pandas loading."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/default.json",
        help="Path to a config JSON file (default: configs/default.json).",
    )
    return parser.parse_args()


def main() -> None:
    """Load config and trigger Yambda auto-preparation."""
    args = parse_args()
    cfg = load_and_resolve(args.config)
    cfg.setdefault("data", {})
    cfg["data"]["source"] = "ctr"
    cfg["data"]["ctr_dataset"] = "yambda"
    yambda_cfg = cfg["ctr"]["datasets"]["yambda"]
    yambda_cfg["dataset_name"] = "yambda"
    yambda_cfg["auto_prepare"] = True
    maybe_prepare_yambda_dataset(yambda_cfg)
    train_path = Path(str(yambda_cfg["train_path"]))
    test_path = Path(str(yambda_cfg["test_path"]))
    print(f"[INFO] Yambda train CSV: {train_path}")
    print(f"[INFO] Yambda test CSV:  {test_path}")


if __name__ == "__main__":
    main()
