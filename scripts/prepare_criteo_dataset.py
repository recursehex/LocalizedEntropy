from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from localized_entropy.config import load_and_resolve
from localized_entropy.data.criteo import maybe_prepare_criteo_dataset


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for Criteo preparation."""
    parser = argparse.ArgumentParser(
        description=(
            "Prepare reczoo/Criteo_x1 files into local train/test CSVs "
            "for pandas loading."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/default.json",
        help="Path to a config JSON file (default: configs/default.json).",
    )
    return parser.parse_args()


def main() -> None:
    """Load config and trigger Criteo auto-preparation."""
    args = parse_args()
    cfg = load_and_resolve(args.config)
    cfg.setdefault("data", {})
    cfg["data"]["source"] = "ctr"
    cfg["data"]["ctr_dataset"] = "criteo"
    criteo_cfg = cfg["ctr"]["datasets"]["criteo"]
    criteo_cfg["dataset_name"] = "criteo"
    criteo_cfg["auto_prepare"] = True
    maybe_prepare_criteo_dataset(criteo_cfg)
    train_path = Path(str(criteo_cfg["train_path"]))
    test_path = Path(str(criteo_cfg["test_path"]))
    print(f"[INFO] Criteo train CSV: {train_path}")
    print(f"[INFO] Criteo test CSV:  {test_path}")


if __name__ == "__main__":
    main()
