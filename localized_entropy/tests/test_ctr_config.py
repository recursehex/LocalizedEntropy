from localized_entropy.config import get_training_source, resolve_ctr_config, resolve_training_cfg
from localized_entropy.data.ctr import warn_if_root_csvs


def _cfg_template() -> dict:
    return {
        "data": {"source": "ctr", "ctr_dataset": "avazu"},
        "ctr": {
            "dataset": "avazu",
            "defaults": {
                "label_col": "click",
                "condition_col": "Condition",
                "test_has_labels": False,
            },
            "datasets": {
                "avazu": {"condition_col": "C14", "test_has_labels": False},
                "criteo": {"condition_col": "C1", "test_has_labels": True},
            },
        },
        "training": {
            "batch_size": 32,
            "by_loss": {
                "bce": {
                    "by_source": {
                        "avazu": {"batch_size": 111},
                        "criteo": {"batch_size": 222},
                        "ctr": {"batch_size": 333},
                    }
                }
            },
        },
    }


def test_resolve_ctr_config_uses_selected_dataset_from_data():
    cfg = _cfg_template()
    cfg["data"]["ctr_dataset"] = "criteo"

    resolved = resolve_ctr_config(cfg)
    assert resolved["dataset_name"] == "criteo"
    assert resolved["condition_col"] == "C1"
    assert resolved["test_has_labels"] is True


def test_resolve_training_cfg_uses_ctr_dataset_source_key():
    cfg = _cfg_template()
    cfg["data"]["ctr_dataset"] = "criteo"

    train_cfg = resolve_training_cfg(cfg, "bce")
    assert get_training_source(cfg) == "criteo"
    assert train_cfg["batch_size"] == 222


def test_resolve_training_cfg_falls_back_to_legacy_ctr_key():
    cfg = _cfg_template()
    cfg["data"]["ctr_dataset"] = "yambda"

    train_cfg = resolve_training_cfg(cfg, "bce")
    assert get_training_source(cfg) == "yambda"
    assert train_cfg["batch_size"] == 333


def test_warn_if_root_csvs_warns_once_per_root(tmp_path, capsys):
    root = tmp_path / "data"
    root.mkdir(parents=True)
    (root / "leftover.csv").write_text("x\n1\n", encoding="utf-8")
    (root / "subdir").mkdir()
    (root / "subdir" / "inside.csv").write_text("x\n1\n", encoding="utf-8")

    warn_if_root_csvs(root)
    out1 = capsys.readouterr().out
    assert "[WARN]" in out1
    assert "leftover.csv" in out1
    assert "inside.csv" not in out1

    warn_if_root_csvs(root)
    out2 = capsys.readouterr().out
    assert out2 == ""
