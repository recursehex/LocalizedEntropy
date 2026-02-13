from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union, TextIO

from localized_entropy.config import get_data_source


def _normalize_filter_mode(mode: Optional[str]) -> str:
    """Normalize filter mode strings to a canonical token."""
    if not mode:
        return ""
    text = str(mode).strip().lower()
    if text in ("ids", "id_list", "list", "include"):
        return "ids"
    if text in ("top", "top_k", "highest", "most"):
        return "top_k"
    if text in ("bottom", "bottom_k", "lowest", "least"):
        return "bottom_k"
    if text in (
        "top_count_rate_mix",
        "count_rate_mix",
        "count_then_rate_mix",
        "top200_mix",
        "mixed_30",
    ):
        return "top_count_rate_mix"
    if text in ("none", "off", "disabled"):
        return "none"
    return text


def resolve_loss_dir(loss_mode: str) -> str:
    """Map a loss mode to its output directory name."""
    text = str(loss_mode).strip().lower()
    if text in {"localized_entropy", "le"}:
        return "le"
    if text == "focal":
        return "focal"
    return "bce"


def resolve_nn_type(cfg: Dict) -> str:
    """Infer the network size label from the experiment name."""
    exp_cfg = cfg.get("experiment", {})
    name = exp_cfg.get("name") or exp_cfg.get("active") or ""
    text = str(name).strip().lower()
    if "small" in text:
        return "small"
    if "large" in text:
        return "large"
    if "wide" in text:
        return "wide"
    return "default"


def resolve_filter_mode(cfg: Dict) -> str:
    """Resolve the active filter mode for output paths."""
    data_source = get_data_source(cfg)
    if data_source != "ctr":
        return "ids"
    ctr_cfg = cfg.get("ctr", {})
    filter_cfg = ctr_cfg.get("filter") or {}
    mode = filter_cfg.get("mode")
    if not mode:
        if filter_cfg.get("ids") or filter_cfg.get("values"):
            mode = "ids"
        elif filter_cfg.get("pool_k") or filter_cfg.get("preselect_k") or filter_cfg.get("candidate_k"):
            mode = "top_count_rate_mix"
        elif filter_cfg.get("k") or filter_cfg.get("top_k") or ctr_cfg.get("filter_top_k"):
            mode = "top_k"
        else:
            mode = "none"
    mode = _normalize_filter_mode(mode)
    filter_enabled = filter_cfg.get("enabled")
    if filter_enabled is None:
        filter_enabled = mode not in ("", "none")
    if not filter_enabled:
        mode = "none"
    if mode not in {"ids", "top_k", "bottom_k", "top_count_rate_mix"}:
        mode = "ids"
    return mode


def build_output_dir(cfg: Dict, loss_mode: str, *, root: Union[Path, str] = "output") -> Path:
    """Build the base output directory for a run."""
    data_source = get_data_source(cfg)
    nn_type = resolve_nn_type(cfg)
    filter_mode = resolve_filter_mode(cfg)
    return Path(root) / resolve_loss_dir(loss_mode) / data_source / nn_type / filter_mode


def build_output_paths(
    cfg: Dict,
    loss_mode: str,
    *,
    root: Union[Path, str] = "output",
) -> Dict[str, Path]:
    """Build output file paths for plots and logs."""
    base = build_output_dir(cfg, loss_mode, root=root)
    return {
        "pred_to_train_rate": base / "avg.png",
        "loss_curves": base / "loss.png",
        "post_training_eval_predictions": base / "preds.png",
        "calibration_ratio": base / "calibration.png",
        "notebook_output": base / "notebook_output.txt",
    }


def _normalize_text(value: str) -> str:
    """Normalize newlines to Unix-style."""
    return value.replace("\r\n", "\n").replace("\r", "\n")


def _stringify_output(output: dict) -> str:
    """Convert a notebook output dict into a text block."""
    output_type = output.get("output_type")
    if output_type == "stream":
        return _normalize_text(output.get("text", ""))
    if output_type in {"display_data", "execute_result"}:
        data = output.get("data", {})
        if "text/plain" in data:
            return _normalize_text("".join(data["text/plain"]))
        return json.dumps(data, indent=2, sort_keys=True)
    if output_type == "error":
        traceback = output.get("traceback", [])
        if traceback:
            return _normalize_text("\n".join(traceback))
        return f"{output.get('ename', '')}: {output.get('evalue', '')}".strip()
    return ""


def _ensure_cell_ids(nb: dict) -> dict:
    """Ensure each notebook cell has a stable ID."""
    import uuid

    for cell in nb.get("cells", []):
        if "id" not in cell:
            cell["id"] = uuid.uuid4().hex
    return nb


def _normalize_notebook(nb: dict, version: int) -> dict:
    """Normalize notebook structure while ensuring cell IDs."""
    try:
        from nbformat import validator
    except Exception:
        return _ensure_cell_ids(nb)

    try:
        nb = _ensure_cell_ids(nb)
        _, normalized = validator.normalize(nb, version=version)
    except Exception:
        return _ensure_cell_ids(nb)
    return normalized


def _read_notebook(notebook_path: Path, as_version: int) -> dict:
    """Read and normalize a notebook as a specific version."""
    import nbformat

    raw = notebook_path.read_text(encoding="utf-8")
    nb = nbformat.reader.reads(raw)
    if as_version is not nbformat.NO_CONVERT:
        nb = nbformat.convert(nb, as_version)
    return _normalize_notebook(nb, version=as_version)


def collect_notebook_outputs(notebook_path: Path) -> List[str]:
    """Collect text outputs from code cells in a notebook."""
    nb = _read_notebook(notebook_path, as_version=4)
    blocks: List[str] = []
    for idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        outputs = cell.get("outputs", [])
        if not outputs:
            continue
        header = f"--- Cell {idx} ---"
        content_parts = [_stringify_output(output) for output in outputs]
        content = "\n".join(part for part in content_parts if part)
        if content:
            blocks.append("\n".join([header, content]))
    return blocks


def save_notebook_outputs(notebook_path: Path, output_path: Path) -> bool:
    """Write notebook outputs to a text file."""
    if not notebook_path.exists():
        print(f"[WARN] Notebook not found: {notebook_path}")
        return False
    try:
        blocks = collect_notebook_outputs(notebook_path)
    except Exception as exc:
        print(f"[WARN] Failed to collect notebook outputs: {exc}")
        return False
    output_text = "\n\n".join(blocks).rstrip() + "\n" if blocks else ""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output_text, encoding="utf-8")
    print(f"[INFO] Wrote notebook outputs to {output_path}")
    return True


class _TeeStream:
    def __init__(self, primary, extras: List[TextIO]):
        """Mirror writes to a primary stream and extra targets."""
        self._primary = primary
        self._extras = extras

    def write(self, text: str) -> int:
        """Write text to all streams."""
        written = self._primary.write(text)
        for stream in self._extras:
            stream.write(text)
        return written

    def writelines(self, lines) -> None:
        """Write multiple lines to all streams."""
        for line in lines:
            self.write(line)

    def flush(self) -> None:
        """Flush all wrapped streams."""
        self._primary.flush()
        for stream in self._extras:
            stream.flush()

    @property
    def encoding(self):
        """Expose the primary stream encoding."""
        return getattr(self._primary, "encoding", "utf-8")

    def isatty(self) -> bool:
        """Proxy terminal detection to the primary stream."""
        if hasattr(self._primary, "isatty"):
            return bool(self._primary.isatty())
        return False

    def __getattr__(self, name):
        """Delegate unknown attributes to the primary stream."""
        return getattr(self._primary, name)


class NotebookOutputCapture:
    def __init__(self, stdout, stderr, handles: List[TextIO]):
        """Capture stdout/stderr and tee output to files."""
        self._stdout = stdout
        self._stderr = stderr
        self._handles = handles

    def stop(self) -> None:
        """Restore stdout/stderr and close output handles."""
        sys.stdout = self._stdout
        sys.stderr = self._stderr
        for handle in self._handles:
            handle.flush()
            handle.close()


def start_notebook_output_capture(output_paths: Dict[str, Dict[str, Path]]) -> NotebookOutputCapture:
    """Start teeing notebook output to configured log files."""
    targets = []
    for paths in output_paths.values():
        target = paths.get("notebook_output")
        if target is not None:
            targets.append(Path(target))
    unique_targets = []
    seen = set()
    for path in targets:
        key = path.absolute()
        if key in seen:
            continue
        seen.add(key)
        unique_targets.append(path)

    handles = []
    for path in unique_targets:
        path.parent.mkdir(parents=True, exist_ok=True)
        handles.append(path.open("w", encoding="utf-8", buffering=1))

    stdout = sys.stdout
    stderr = sys.stderr
    sys.stdout = _TeeStream(stdout, handles)
    sys.stderr = _TeeStream(stderr, handles)
    return NotebookOutputCapture(stdout, stderr, handles)
