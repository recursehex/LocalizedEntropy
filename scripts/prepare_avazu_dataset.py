from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from localized_entropy.config import load_and_resolve
from localized_entropy.data.avazu import maybe_prepare_avazu_dataset


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for Avazu preparation."""
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the Avazu CTR competition into local train/test CSVs "
            "(downloads via kagglehub when configured)."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/default.json",
        help="Path to a config JSON file (default: configs/default.json).",
    )
    return parser.parse_args()


def main() -> None:
    """Load config and trigger Avazu auto-preparation."""
    args = parse_args()
    cfg = load_and_resolve(args.config)
    cfg.setdefault("data", {})
    cfg["data"]["source"] = "ctr"
    cfg["data"]["ctr_dataset"] = "avazu"
    avazu_cfg = cfg["ctr"]["datasets"]["avazu"]
    avazu_cfg["dataset_name"] = "avazu"
    avazu_cfg["auto_prepare"] = True
    maybe_prepare_avazu_dataset(avazu_cfg)
    train_path = Path(str(avazu_cfg["train_path"]))
    test_path = Path(str(avazu_cfg["test_path"]))
    print(f"[INFO] Avazu train CSV: {train_path}")
    print(f"[INFO] Avazu test CSV:  {test_path}")


if __name__ == "__main__":
    main()
