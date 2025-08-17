import os
from typing import Any
from dataclasses import is_dataclass, asdict
from enum import Enum


def _ensure_data_dirs() -> None:
    base_dir = _project_root()
    os.makedirs(os.path.join(base_dir, "data/inputs"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "data/processed"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "data/outputs"), exist_ok=True)


def _project_root() -> str:
    """Return absolute path to the project root (one level up from this file's directory)."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def _resolve_path(path: str) -> str:
    """Resolve relative paths to the project root if they don't exist as given."""
    if not path:
        return path
    if os.path.isabs(path) and os.path.exists(path):
        return path
    if os.path.exists(path):
        return path
    candidate = os.path.join(_project_root(), path)
    return candidate


def _to_json_compatible(obj: Any) -> Any:
    """Recursively convert dataclasses, Enums, sets, and other containers to JSON-compatible primitives."""
    if is_dataclass(obj):
        # Convert dataclass to dict and recurse
        return {k: _to_json_compatible(v) for k, v in asdict(obj).items()}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, set):
        # Sort for stable output
        return sorted(_to_json_compatible(v) for v in obj)
    if isinstance(obj, (list, tuple)):
        return [_to_json_compatible(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _to_json_compatible(v) for k, v in obj.items()}
    return obj
