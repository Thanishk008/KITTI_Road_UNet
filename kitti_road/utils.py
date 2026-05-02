from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

try:
    import yaml
except ImportError:  # pragma: no cover - load_config gives a clearer message
    yaml = None


def load_config(path: str | Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required for config files. Install dependencies with: pip install -r requirements.txt")
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_hash(data: Any) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def append_csv_row(path: str | Path, row: Dict[str, Any], fieldnames: list[str]) -> None:
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def device_from_config(cpu: bool = False) -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
